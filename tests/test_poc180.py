"""
The PoC-180 yardstick: fixed panels, a checkpoint in, a report out.

    uv run python tests/test_poc180.py
"""
import _runner  # noqa: F401  (puts the repo root on sys.path)
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import poc180 as yard
from agent import Agent, save_checkpoint
from config import SAW_OBS_DIM, SAW_ACT_DIM


def _checkpoint(folder, obs_dim=SAW_OBS_DIM, **meta):
    torch.manual_seed(0)
    path = Path(folder) / "run_x" / "SAW_upd7.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, Agent(obs_dim, SAW_ACT_DIM), update=7, p=0.5, stage="combat", rules="saw", **meta)
    return path


def test_the_panels_are_fixed_and_differ():
    assert set(yard.PANELS) == {"T", "F", "L"}
    assert len({v["seed"] for v in yard.PANELS.values()}) == 3
    assert yard.COMMON == dict(from_bell=True, arm_reading=False, blade_reading=False, weapon_fail_share=0.0)


def test_a_report_comes_out_whole_and_repeats():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp, arm_gate=True, gate_rule="body")
        kw = dict(n=24, seconds=6.0, out_dir=Path(tmp) / "out", quiet=True)
        rep = yard.poc180(path, **kw)
        again = yard.poc180(path, write=False, **kw)
        assert set(rep["panels"]) == {"T", "F", "L"}
        for name, r in rep["panels"].items():
            assert r["n_fights"] == 24 and r["arm_gate"] is True and r["gate_rule"] == "body"
            for key in ("poc_share", "clean_share", "ge2_share", "attack_in_share", "land_legal", "land_side", "land_floor"):
                assert 0.0 <= r[key] <= 1.0, (name, key, r[key])
            assert r["clean_share"] <= r["poc_share"] and r["ge2_share"] <= r["poc_share"]
            assert r["swings_used"] <= 10.0 + 1e-6
            same = {k: v for k, v in r.items()}
            assert same == again["panels"][name], f"panel {name} did not repeat"
        # T is all Traves-like horizontals; the others hold none
        assert rep["panels"]["T"]["n_traves"] == rep["panels"]["T"]["n_horiz"] == 24
        assert rep["panels"]["F"]["n_traves"] == rep["panels"]["L"]["n_traves"] == 0
        assert rep["panels"]["L"]["scenario"]["legacy_field"] is True
        written = Path(rep["written"])
        assert written.name == "run_x_SAW_upd7_u7_gckpt-body_s10_r0.5_none.json"
        assert json.loads(written.read_text(encoding="utf-8"))["panels"]["T"]["seed"] == 1000
        text = written.with_suffix(".txt").read_text(encoding="utf-8")
        assert "poc_share" in text and len(text.splitlines()) == 5


def test_a_saw_v7_checkpoint_runs_with_and_without_the_gate():
    """A 30-input policy from before the gate drives the arm itself unless
    told otherwise, and the gate can be put in front of it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp, obs_dim=30, physics="saw-v7")
        kw = dict(n=16, seconds=6.0, panels=("T",), write=False, quiet=True)
        own = yard.poc180(path, **kw)
        assert own["panels"]["T"]["arm_gate"] is False and own["panels"]["T"]["gate_rule"] is None
        gated = yard.poc180(path, gate="on", gate_rule="body", **kw)
        assert gated["panels"]["T"]["arm_gate"] is True and gated["panels"]["T"]["gate_rule"] == "body"
        assert gated["panels"]["T"]["fires"] <= 10.0
        lit = yard.poc180(path, heading="axis_nose", swings=20, reach=1.0, **kw)
        assert lit["panels"]["T"]["scenario"]["heading_mode"] == "axis_nose" and lit["settings"]["swings"] == 20


def test_a_classic_checkpoint_is_refused():
    from config import OBS_DIM, ACT_DIM
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Z.pt"
        save_checkpoint(path, Agent(OBS_DIM, ACT_DIM), update=1, rules="classic")
        try:
            yard.poc180(path, write=False, quiet=True)
        except SystemExit:
            return
        raise AssertionError("a classic checkpoint was put through the saw yardstick")


if __name__ == "__main__":
    _runner.run(globals())
