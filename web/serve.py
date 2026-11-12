"""
The dodge_master web viewer: training curves, yardstick scores and fight
replays in the browser.

    uv run python web/serve.py              then open http://127.0.0.1:8765
    uv run python web/serve.py --port 9000

Apart from "Fight the policy" (web/play.py, a live fight on the CPU) it only reads: the training logs in logs/v8/ and logs/v9/, the yardstick reports in
eval/poc180/ and the replays in web/replays/ (made by web/export_replays.py).
It listens on this machine only and needs nothing beyond the standard library.
"""
import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
REPLAYS = ROOT / "web" / "replays"
LOGS = ROOT / "logs" / "v8"
LOGS_V9 = ROOT / "logs" / "v9"            # runs under the saw-v9 rules; shown as "v9_<name>"
YARD = ROOT / "eval" / "poc180"

UPD = re.compile(r"^\[\s*([\d,]+)\] upd=\s*(\d+) sps=\s*(\d+) win=([\d.]+) loss=([\d.]+) draw=([\d.]+)"
                 r"(?: hit=([\d.]+))? p=([\d.]+) lr=\S+ std=([\d.]+) kl=([\d.]+) ret=([+-][\d.]+)")
LEN = re.compile(r"^\s+len180: n=(\d+) solid=([\d.]+) atk=([\d.]+) dealt=([\d.]+) taken=([\d.]+) swings=([\d.]+) "
                 r"land L/W/F=([\d.]+)/([\d.]+)/([\d.]+)")
KEYS = ("update", "steps", "sps", "win", "loss", "draw", "hit", "ret",
        "solid", "attack_in", "dealt", "taken", "swings", "legal", "side", "floor")
_cache = {}


def cached(path, build):
    """build(path) once per version of a file."""
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    value = build(path)
    _cache[path] = (stamp, value)
    return value


def parse_log(path):
    rows, info, row = [], [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = UPD.match(line)
            if m:
                g = m.groups()
                row = dict(update=int(g[1]), steps=int(g[0].replace(",", "")), sps=int(g[2]), win=float(g[3]),
                           loss=float(g[4]), draw=float(g[5]), hit=float(g[6]) if g[6] else None, ret=float(g[10]))
                rows.append(row)
                continue
            m = LEN.match(line)
            if m and row is not None:
                g = [float(x) for x in m.groups()]
                if g[0] > 0:
                    row.update(solid=g[1], attack_in=g[2], dealt=g[3], taken=g[4], swings=g[5],
                               legal=g[6], side=g[7], floor=g[8])
                continue
            if line.startswith(("[INFO]", "[INIT]", "[RESUME]")):
                info.append(line.strip())
    # A run fine-tuned from another v8a run carries on from that run's update count on the charts
    offset = 0
    for line in info:
        m = re.match(r"\[INIT\] combat starts from \S*saw_v8a\S*_upd(\d+)\.pt\s*$", line)
        if m:
            offset = int(m.group(1))
    crashed = "Traceback" in path.read_text(encoding="utf-8", errors="replace")[-6000:]
    return {"series": {k: [r.get(k) for r in rows] for k in KEYS}, "info": info, "crashed": crashed, "offset": offset}


def training():
    runs = []
    for path in sorted(LOGS.glob("train_*.log")) + sorted(LOGS_V9.glob("train_*.log")):
        doc = cached(path, parse_log)
        if not doc["series"]["update"]:
            continue
        age = time.time() - path.stat().st_mtime
        runs.append({"name": ("v9_" if path.parent == LOGS_V9 else "") + path.stem[len("train_"):], "log": path.name, "seconds_since_write": round(age),
                     "live": age < 240 and not doc["crashed"], **doc})
    return {"runs": runs, "now": time.time()}


def yardstick():
    out = []
    for path in sorted(YARD.glob("*.json")):
        try:
            doc = cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))
            ckpt = Path(doc["meta"]["checkpoint"])
            out.append({"file": path.name, "run": ckpt.parent.name, "checkpoint": ckpt.name,
                        "update": doc["meta"]["update"], "settings": doc["settings"],
                        "arm_gate": doc["meta"]["arm_gate"], "gate_rule": doc["meta"]["gate_rule"],
                        "panels": {k: {m: v for m, v in r.items() if m != "scenario"}
                                   for k, r in doc["panels"].items()}})
        except (ValueError, KeyError):
            continue
    return {"reports": out}


def replay_head(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in doc.items() if k not in ("frames", "events", "hazards", "columns")}


def replays():
    if not REPLAYS.exists():
        return {"replays": []}
    return {"replays": [{"file": p.name, **cached(p, replay_head)} for p in sorted(REPLAYS.glob("*.json"))]}


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml"}
API = {"/api/training": training, "/api/yardstick": yardstick, "/api/replays": replays}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Fight the policy: /api/play/start and /api/play/step. The sim is loaded on first use."""
        path = self.path.split("?", 1)[0]
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if path not in ("/api/play/start", "/api/play/step") or size > 4096:
                return self.send(404, b"not found", "text/plain")
            origin = self.headers.get("Origin")
            if origin and origin.split("//")[-1].split(":")[0] not in ("127.0.0.1", "localhost"):
                return self.send(403, b"forbidden", "text/plain")
            body = json.loads(self.rfile.read(size) or b"{}")
            import play                                          # torch and the sim: a few seconds the first time
            out = play.start(body) if path.endswith("start") else play.step(body)
            self.send(200, json.dumps(out, separators=(",", ":")).encode("utf-8"))
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as e:
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in API:
                return self.send(200, json.dumps(API[path]()).encode("utf-8"))
            if path.startswith("/replays/"):
                base, name = REPLAYS, path[len("/replays/"):]
            else:
                base, name = STATIC, path.lstrip("/") or "index.html"
            target = (base / name).resolve()
            if base.resolve() not in target.parents or not target.is_file() or target.suffix not in TYPES:
                return self.send(404, b"not found", "text/plain")
            self.send(200, target.read_bytes(), TYPES[target.suffix])
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as e:                                  # one bad file must not take the page down
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))    # web/play.py
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"dodge_master viewer: http://127.0.0.1:{a.port}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
