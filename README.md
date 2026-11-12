# nemesis_auton: Nemesis hammer-saw sim (2026-11-12)

A copy of the current working code only: the saw-v9 sim, the trainer, the
yardstick, the web viewer and two checkpoints. Old generations, logs, audits
and backups stay in the parent folder.

## Layout

    config.py              every tunable number
    environment.py         base sim and env factory
    saw_env.py             hammer-saw rules (saw-v9)
    saw_field.py           opponent mix, per-fight scenario
    saw_gate.py            arm gate
    saw_walls.py           smooth solid walls
    opponent_library.py    scripted opponent drivers
    opponent_selfplay.py   self-play opponents (classic rules)
    agent.py               policy network, checkpoint loading
    train_ppo.py           trainer
    eval/poc180.py         yardstick: share of 3-minute fights with a solid legal strike
    web/                   viewer: serve.py, play.py (fight the policy), export_replays.py, static/
    tests/                 sim tests
    battlebot_models/      arena map used by the classic rules
    ppo_models/            saw_v8a_traves_hold/SAW_upd375.pt (best vs Traves-like),
                           saw_v8a_body_priv/SAW_upd950.pt (best general)

## Use

    uv sync
    uv run python tests/test_saw.py
    uv run python eval/poc180.py ppo_models/saw_v8a_traves_hold/SAW_upd375.pt --lite
    uv run python web/serve.py            # then open http://127.0.0.1:8765

Set `DODGE_DEVICE=cpu` to keep off the GPU.

## On another machine (Linux or Windows)

Needs [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU with a
current driver. `uv sync` installs Python packages only, including PyTorch
built for CUDA 12.8; on a machine without a GPU everything still runs on the CPU.

    git clone <your repository> && cd <folder>
    uv sync
    uv run python tests/test_saw.py

Train (about 4 GB of GPU memory per run at 32,768 fights at once; lower
`--n-envs` on a smaller card):

    mkdir -p logs/v9
    uv run python train_ppo.py --init ppo_models/saw_v8a_traves_hold/SAW_upd375.pt --init-std 0.25 \
        --name my_run/SAW --n-envs 32768 --epochs 2 --minibatch-envs 1024 --target-kl 0.02 --lr 1e-4 \
        --critic-warmup 10 --p-range 0.4 0.85 --gate-rule body --gate-persist 4 --priv-critic \
        --traves-share 1.0 --weapon-shares 0.1 0.1 0.8 --swings-3min 10 --foul 0.05 --save-every 25 \
        > logs/v9/train_my_run.log 2>&1

Checkpoints land in `ppo_models/my_run/`. The viewer listens on 127.0.0.1
only; to see it from another computer, forward the port over SSH:

    ssh -L 8765:127.0.0.1:8765 user@linux-machine

Both checkpoints were trained under the earlier saw-v8 rules; the saw-v9
runs are still training in the parent folder.
