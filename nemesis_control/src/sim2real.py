"""
Converts /control/channels to PPM for the Radiomaster Zorro's trainer port.

    channel 1, 2   left and right tread, always
    channel 3, 4   blade (-1 stopped .. +1 full) and arm motor (+1 forward, -1 back,
                   0 no torque: the policy node only powers it while the arm travels),
                   only with NEMESIS_PPM_WEAPON=true (the launch file's weapon:=true)

With no fresh command (see control_listener.STALE_SEC) the treads and the arm
go to centre (1500 us) and the blade to stopped (1000 us).

Which transmitter input each PPM channel drives is set on the Zorro (its
trainer channel mapping). Check the blade and arm mapping with the weapon
disconnected before switching weapon output on.
"""

# ────────────────────────────────────────────────────────────────────────────
# ── Imports ─────────────────────────────────────────────────────────────────

import numpy as np
import os
import sounddevice as sd
import threading
import time

from control_listener import get_channels, main

# ── Configuration ───────────────────────────────────────────────────────────

SAMPLE_HZ       = 44100     # Hz — standard audio sample rate                       # To be tuned
NUM_CHANNELS    = 8         # PPM carries 8 channels 
FRAME_US        = 22500     # Total PPM frame length in microseconds
PULSE_US        = 300       # PPM pulse length in microseconds
BLOCKSIZE       = 512       # Audio buffer size in samples

HIGH_SAMPLE_VAL = 1.0       # High audio samples correspond to pulse
LOW_SAMPLE_VAL  = -1.0      # Low audio samples correspond to gap

FULL_QUEUE_SLEEP_TIME  = 0.001  # Time to sleep when queue is full, in seconds      # To be tuned
FULL_QUEUE_SAMPLES_COUNT = 50    # Number of frames in a full queue                  # To be tuned


# ── Calculated Constants (Automatically updated by configuration) ───────────

SAMPLES_PER_US  = SAMPLE_HZ / 1_000_000   
NUM_HIGH_SAMPLES_PER_PULSE = round(PULSE_US * SAMPLES_PER_US)   # 13


# ── Shared Channel State ────────────────────────────────────────────────────

channels_us = [1500] * NUM_CHANNELS     # List containing the length of each channel, array is intended to be modified
channels_lock = threading.Lock()


def float_to_us(v: float) -> int:
    """Convert normalized float [-1, 1] to PPM microseconds [1000, 2000]"""
    v = min(max(-1.0, float(v)), 1.0)
    return round(500 * v + 1500)


def generate_ppm_frame(channels_us: list) -> np.ndarray:
    """
    Build one complete PPM frame as a 1D numpy array of audio samples.\n
    Each sample is a float: +1.0 (high) or -1.0 (low).\n
    High samples represent pulses.\n
    Low samples represent gaps.
    """
    samples = []

    # Generate pulse-gap-pulse-gap wave based on channel lengths 
    for channel_us in channels_us:
        number_of_high_samples = NUM_HIGH_SAMPLES_PER_PULSE
        samples.extend([HIGH_SAMPLE_VAL] * number_of_high_samples) 

        number_of_low_samples = round((channel_us - PULSE_US) * SAMPLES_PER_US)
        samples.extend([LOW_SAMPLE_VAL] * number_of_low_samples) 

    # Generate final pulse to mark end of last channel's gap
    samples.extend([HIGH_SAMPLE_VAL] * NUM_HIGH_SAMPLES_PER_PULSE)

    # Generate gap to fill remaining frame time (FRAME_US)
    used_us = sum(channels_us)
    number_of_low_samples = round((FRAME_US - used_us - PULSE_US) * SAMPLES_PER_US)
    samples.extend([LOW_SAMPLE_VAL] * number_of_low_samples)

    samples = np.array(samples, dtype=np.float32)
    return samples


# ── Generate queue and buffer of PPM frames ─────────────────────────────────
# audio_callback feeds frames from sample_queue to the buffer
# This ensures there are always fresh samples ready for the audio driver.

sample_queue = [] 
queue_lock = threading.Lock()

def extend_sample_queue():
    """
    Reads channel values, converts to PPM samples, pauses briefly if buffer is full.
    Mutates sample_queue directly.
    """
    while True:
        with channels_lock:
            channels_snapshot = channels_us[:]
        samples = generate_ppm_frame(channels_snapshot)

        with queue_lock:
            sample_queue.extend(samples.tolist())

        while len(sample_queue) > FULL_QUEUE_SAMPLES_COUNT * len(samples):
            time.sleep(FULL_QUEUE_SLEEP_TIME)


def audio_callback(outdata: np.ndarray, number_of_requested_samples: int, time_info, status):
    """
    Called by sounddevice whenever buffer needs refilling.\n
    Reads from sample_queue.
    """
    with queue_lock:
        number_of_queued_samples = len(sample_queue)

        if number_of_queued_samples >= number_of_requested_samples:
            outdata[:, 0] = sample_queue[:number_of_requested_samples]
            del sample_queue[:number_of_requested_samples]
        else:
            # Caller requested more samples than we have currently in sample_queue,
            # so send what we have, and pad the rest of outdata with low-sample values
            outdata[:number_of_queued_samples, 0] = sample_queue[:]
            outdata[number_of_queued_samples:, 0] = LOW_SAMPLE_VAL    
            del sample_queue[:]         


# ── Main float to ppm conversion block (lots of threads lol) ────────────────

# A thread
extend_sample_queue_thread = threading.Thread(target=extend_sample_queue, daemon=True)
extend_sample_queue_thread.start()

# Another thread
stream = sd.OutputStream(
    samplerate=SAMPLE_HZ,
    channels=1,
    dtype="float32",
    blocksize=BLOCKSIZE,
    callback=audio_callback,
)
stream.start()

# And another thread
control_channels_subscriber_thread = threading.Thread(target=main, daemon=True)  
control_channels_subscriber_thread.start()

# ── Main loop: /control/channels -> PPM channels 1 and 2 ────────────────────

NEUTRAL_US = 1500
BLADE_STOPPED_US = 1000
WEAPON = os.environ.get("NEMESIS_PPM_WEAPON", "").lower() in ("1", "true", "yes")
print("PPM out: treads on channels 1-2" + (", blade on 3, arm on 4" if WEAPON else "; weapon channels off"), flush=True)

try: 
    dt = 0.02
    last_print = 0.0
    while True:
        wL, wR, blade, arm = get_channels()
        live = (wL is not None) and (wR is not None)
        with channels_lock:
            # Failsafe: with no fresh command on /control/channels (policy
            # stopped, vision lost, node crashed) send neutral sticks instead
            # of repeating the last command.
            channels_us[0] = float_to_us(wL) if live else NEUTRAL_US
            channels_us[1] = float_to_us(wR) if live else NEUTRAL_US
            if WEAPON:
                channels_us[2] = float_to_us(blade) if live else BLADE_STOPPED_US
                channels_us[3] = float_to_us(arm) if live else NEUTRAL_US
        if time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            text = f"PPM out: wL={wL:+.2f} wR={wR:+.2f}" + (f" blade={blade:+.2f} arm={arm:+.2f}" if WEAPON else "") if live else "PPM out: no command -> neutral"
            print(text, flush=True)
        time.sleep(dt)
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    stream.stop()
    print("Stopped.")