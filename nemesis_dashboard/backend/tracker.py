"""Constant-velocity tracker in arena (homography) coordinates."""

from __future__ import annotations

from typing import Optional

import numpy as np

# =========================
# Tracker config (arena-coordinate interpolation)
# =========================
VELOCITY_DECAY = 0.9    # per-step velocity bias when coasting (< 1 → slight deceleration)
ASSOC_MAX_DIST = 100      # max arena-unit distance to match a detection to a predicted track
TRACK_TIMEOUT = 2.0       # seconds without a matched detection before dropping a track


class RobotTrack:
    """Single-robot constant-velocity track in arena (homography) coordinates."""

    def __init__(self, pos, t):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.t = t
        self.last_det_t = t

    def predict(self, t_now):
        dt = t_now - self.t
        if dt <= 0:
            return self.pos.copy()
        return self.pos + self.vel * dt

    def update_with_detection(self, measurement, t_now):
        dt = t_now - self.t
        if dt > 0:
            self.vel = (np.array(measurement) - self.pos) / dt
        self.pos = np.array(measurement, dtype=np.float64)
        self.t = t_now
        self.last_det_t = t_now

    def coast(self, t_now):
        dt = t_now - self.t
        if dt > 0:
            self.pos = self.pos + self.vel * dt
            self.vel *= VELOCITY_DECAY
        self.t = t_now

    def is_expired(self, t_now):
        return (t_now - self.last_det_t) > TRACK_TIMEOUT


class DualTracker:
    """Tracks nemesis and opponent in arena coordinates using constant-velocity interpolation."""

    def __init__(self):
        self.nem: Optional[RobotTrack] = None
        self.opp: Optional[RobotTrack] = None

    def step(self, arena_positions, t_now):
        """
        arena_positions: list of (x, y) arena-coord positions for each detection.

        Returns (nem_info, opp_info) where each is:
            (arena_pos_tuple, det_index) if track exists, else None.
            det_index is int if matched to a detection, None if coasting.
        """
        # Drop expired tracks
        if self.nem and self.nem.is_expired(t_now):
            self.nem = None
        if self.opp and self.opp.is_expired(t_now):
            self.opp = None

        nem_pred = self.nem.predict(t_now) if self.nem else None
        opp_pred = self.opp.predict(t_now) if self.opp else None

        n_dets = len(arena_positions)

        # No tracks yet — initialise from detections
        if self.nem is None and self.opp is None:
            if n_dets == 0:
                return (None, None)
            if n_dets == 1:
                self.nem = RobotTrack(arena_positions[0], t_now)
                return ((tuple(arena_positions[0]), 0), None)
            # >= 2 detections: leftmost = nemesis
            indices = sorted(range(n_dets), key=lambda i: arena_positions[i][0])
            self.nem = RobotTrack(arena_positions[indices[0]], t_now)
            self.opp = RobotTrack(arena_positions[indices[1]], t_now)
            return (
                (tuple(arena_positions[indices[0]]), indices[0]),
                (tuple(arena_positions[indices[1]]), indices[1]),
            )

        # Build greedy nearest-neighbour assignment (tracks ↔ detections)
        tracks = []
        track_names = []
        preds = []
        if self.nem is not None:
            tracks.append(self.nem)
            track_names.append('nem')
            preds.append(nem_pred)
        if self.opp is not None:
            tracks.append(self.opp)
            track_names.append('opp')
            preds.append(opp_pred)

        assignments = {}  # track_name -> det_index
        if n_dets > 0 and len(tracks) > 0:
            dists = np.zeros((len(tracks), n_dets))
            for ti, pred in enumerate(preds):
                for di in range(n_dets):
                    dists[ti, di] = np.linalg.norm(pred - np.array(arena_positions[di]))

            used_tracks = set()
            used_dets = set()
            for flat_idx in np.argsort(dists, axis=None):
                ti = int(flat_idx // n_dets)
                di = int(flat_idx % n_dets)
                if ti in used_tracks or di in used_dets:
                    continue
                if dists[ti, di] > ASSOC_MAX_DIST:
                    continue
                assignments[track_names[ti]] = di
                used_tracks.add(ti)
                used_dets.add(di)

        # Update matched tracks / coast unmatched
        nem_info = None
        opp_info = None

        if self.nem is not None:
            if 'nem' in assignments:
                di = assignments['nem']
                self.nem.update_with_detection(arena_positions[di], t_now)
                nem_info = (tuple(self.nem.pos), di)
            else:
                self.nem.coast(t_now)
                nem_info = (tuple(self.nem.pos), None)

        if self.opp is not None:
            if 'opp' in assignments:
                di = assignments['opp']
                self.opp.update_with_detection(arena_positions[di], t_now)
                opp_info = (tuple(self.opp.pos), di)
            else:
                self.opp.coast(t_now)
                opp_info = (tuple(self.opp.pos), None)

        # Create new tracks from unassigned detections
        assigned_dets = set(assignments.values())
        for di in range(n_dets):
            if di not in assigned_dets:
                if self.nem is None:
                    self.nem = RobotTrack(arena_positions[di], t_now)
                    nem_info = (tuple(arena_positions[di]), di)
                elif self.opp is None:
                    self.opp = RobotTrack(arena_positions[di], t_now)
                    opp_info = (tuple(arena_positions[di]), di)

        return (nem_info, opp_info)

    def step_known(self, arena_positions, nem_idx, opp_idx, t_now):
        """step() for a model that tells the robots apart itself: nem_idx and
        opp_idx are the detection index of each robot (None = not detected).
        Same return value, same coasting and expiry."""
        infos = []
        for name, di in (('nem', nem_idx), ('opp', opp_idx)):
            track = getattr(self, name)
            if track and track.is_expired(t_now):
                track = None
            if di is None:
                if track:
                    track.coast(t_now)
            elif track:
                track.update_with_detection(arena_positions[di], t_now)
            else:
                track = RobotTrack(arena_positions[di], t_now)
            setattr(self, name, track)
            infos.append((tuple(track.pos), di) if track else None)
        return tuple(infos)
