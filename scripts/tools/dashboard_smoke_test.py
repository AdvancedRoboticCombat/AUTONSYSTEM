#!/usr/bin/env python3
"""Hardware-free check of the dashboard backend (nemesis_dashboard/main.py).

Plays a generated video through `main.py --no-ros --vision none` (no camera,
model, GPU or ROS needed) and checks over HTTP and the WebSocket:

  1. the page is served, with its security headers,
  2. a request addressed to another host name is refused (DNS rebinding),
  3. a WebSocket opened by another site's page is refused,
  4. state, vision and JPEG frames arrive,
  5. video waits for "ready": one frame, then none until the page asks,
  6. OVERRIDE sets the state asked for, also when asked twice, and is logged,
  7. bad messages are ignored and the connection stays usable,
  8. more than 10 actions a second from one page are dropped,
  9. an oversized message closes only that connection,
 10. a ninth page is refused,
 11. Ctrl-C stops the backend cleanly.

Run through scripts/test.sh, or with any Python that has aiohttp, numpy and cv2.
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import cv2
import numpy as np

DASHBOARD = Path(__file__).resolve().parents[2] / "nemesis_dashboard"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}", flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_video(path, seconds=40, fps=15):
    """A square moving across a grey field, as MJPG (any OpenCV build can play it)."""
    w, h = 320, 240
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not out.isOpened():
        sys.exit(f"FAIL  OpenCV cannot write {path}")
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), 90, np.uint8)
        x = int((i * 4) % (w - 40))
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (0, 200, 0), -1)
        out.write(frame)
    out.release()


async def receive_until(ws, want, timeout):
    """Messages until want(message) is true; returns that message or None."""
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        try:
            msg = await ws.receive(timeout=left)
        except asyncio.TimeoutError:
            return None
        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            return msg
        if want(msg):
            return msg
    return None


def is_text(kind):
    def want(msg):
        return msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("type") == kind
    return want


def is_frame(msg):
    return msg.type == aiohttp.WSMsgType.BINARY


async def first_state(session, ws_url, origin):
    """override as a brand-new page sees it."""
    async with session.ws_connect(ws_url, origin=origin) as ws:
        msg = await receive_until(ws, is_text("state"), 3)
        return None if msg is None else json.loads(msg.data)["data"]["override"]


async def run_checks(base, origin):
    ws_url = base + "/ws"
    async with aiohttp.ClientSession() as session:
        # 1. The page and its headers
        async with session.get(base + "/") as r:
            body = await r.text()
            built = (DASHBOARD / "dist" / "index.html").is_file()
            page_ok = r.status == 200 and ('id="root"' in body if built else "not built" in body)
            check("page served" + ("" if built else " (not built: the notice page)"), page_ok, f"HTTP {r.status}")
            csp = r.headers.get("Content-Security-Policy", "")
            check("security headers",
                  "frame-ancestors 'none'" in csp and r.headers.get("X-Frame-Options") == "DENY"
                  and r.headers.get("X-Content-Type-Options") == "nosniff",
                  csp[:60] + "...")
        async with session.get(base + "/assets/missing.js") as r:
            check("missing asset is a 404, not the page", r.status == 404, f"HTTP {r.status}")

        # 2. DNS rebinding: the right address, another name
        async with session.get(base + "/", headers={"Host": "evil.example:80"}) as r:
            check("request for another host name refused", r.status == 421, f"HTTP {r.status}")

        # 3. Cross-site WebSocket
        try:
            async with session.ws_connect(ws_url, origin="http://evil.example"):
                status = 101
        except aiohttp.WSServerHandshakeError as error:
            status = error.status
        check("WebSocket from another site refused", status == 403, f"HTTP {status}")

        async with session.ws_connect(ws_url, origin=origin) as ws:
            # 4. Messages
            msg = await receive_until(ws, is_text("state"), 3)
            state = json.loads(msg.data)["data"] if msg is not None and msg.type == aiohttp.WSMsgType.TEXT else None
            check("state on connect", state is not None and state.get("override") is False)
            msg = await receive_until(ws, is_text("vision"), 10)
            check("vision messages", msg is not None and msg.type == aiohttp.WSMsgType.TEXT
                  and json.loads(msg.data).get("detW") == 320)

            # 5. Credits
            msg = await receive_until(ws, is_frame, 10)
            check("first JPEG frame", msg is not None and msg.type == aiohttp.WSMsgType.BINARY
                  and msg.data[:2] == b"\xff\xd8")
            extra = await receive_until(ws, is_frame, 1.0)
            check("no second frame before 'ready'", extra is None)
            await ws.send_json({"type": "ready"})
            msg = await receive_until(ws, is_frame, 3)
            check("next frame after 'ready'", msg is not None and msg.type == aiohttp.WSMsgType.BINARY)

            # 6. OVERRIDE
            await ws.send_json({"type": "override", "manual": True})
            msg = await receive_until(ws, lambda m: is_text("state")(m) and json.loads(m.data)["data"]["override"], 3)
            check("override -> manual reaches the page", msg is not None)
            await ws.send_json({"type": "override", "manual": True})
            await asyncio.sleep(0.3)
            check("asking for manual twice stays manual", await first_state(session, ws_url, origin) is True)
            await ws.send_json({"type": "override", "manual": False})
            await asyncio.sleep(0.3)
            check("override -> autonomous", await first_state(session, ws_url, origin) is False)

            # 7. Garbage
            await asyncio.sleep(1.1)  # a fresh one-second action budget
            for bad in ("not json", "[1, 2]", json.dumps({"type": "nope"}),
                        json.dumps({"type": "canvas-click", "x": "a", "y": 1}),
                        json.dumps({"type": "canvas-click", "x": 1e400, "y": 1}),
                        json.dumps({"type": "action", "key": "rm -rf /"}),
                        json.dumps({"type": "override", "manual": "yes"})):
                await ws.send_str(bad)
            await ws.send_bytes(b"\x00\x01")
            await asyncio.sleep(0.3)
            check("bad messages ignored, connection open", not ws.closed and
                  await first_state(session, ws_url, origin) is False)

            # 8. Rate limit: the 11th action within a second is dropped
            await asyncio.sleep(1.1)
            for _ in range(10):
                await ws.send_json({"type": "override", "manual": False})
            await ws.send_json({"type": "override", "manual": True})
            await asyncio.sleep(0.3)
            check("11th action in a second dropped", await first_state(session, ws_url, origin) is False)
            await asyncio.sleep(1.1)
            await ws.send_json({"type": "override", "manual": True})
            await asyncio.sleep(0.3)
            check("actions accepted again a second later", await first_state(session, ws_url, origin) is True)
            await ws.send_json({"type": "override", "manual": False})

            # 9. Oversized message
            async with session.ws_connect(ws_url, origin=origin) as big:
                await big.send_str("x" * 100_000)
                msg = await receive_until(big, lambda m: False, 3)
                closed = msg is not None and msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
                check("oversized message closes that connection", closed, f"close code {big.close_code}")
            check("other connections unaffected", not ws.closed and
                  await first_state(session, ws_url, origin) is False)

            # 10. Page limit (this one is open already)
            others = [await session.ws_connect(ws_url, origin=origin) for _ in range(7)]
            try:
                async with session.ws_connect(ws_url, origin=origin):
                    status = 101
            except aiohttp.WSServerHandshakeError as error:
                status = error.status
            check("ninth page refused", status == 503, f"HTTP {status}")
            for other in others:
                await other.close()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "fight.avi"
        make_video(video)
        port = free_port()
        log_dir = tmp / "logs"
        env = {**os.environ, "NEMESIS_LOG_DIR": str(log_dir)}
        env.pop("NEMESIS_START_MANUAL", None)
        output = open(tmp / "backend.out", "w")
        backend = subprocess.Popen(
            [sys.executable, "main.py", "--no-ros", "--vision", "none", "--fight-video", str(video),
             "--port", str(port)],
            cwd=DASHBOARD, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            up = False
            while time.monotonic() < deadline and backend.poll() is None:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        up = True
                        break
                except OSError:
                    time.sleep(0.2)
            check("backend starts", up, f"see {tmp / 'backend.out'}" if not up else "")
            if up:
                asyncio.run(run_checks(base, f"http://localhost:{port}"))
        except Exception as error:
            check("checks ran to the end", False, repr(error))
        finally:
            if backend.poll() is None:
                os.killpg(backend.pid, signal.SIGINT)
            try:
                code = backend.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(backend.pid, signal.SIGKILL)
                code = None
            output.close()
            check("Ctrl-C stops the backend cleanly", code == 0, f"exit code {code}")
            log = (log_dir / "dashboard.log").read_text() if (log_dir / "dashboard.log").exists() else ""
            check("OVERRIDE changes are logged", "OVERRIDE -> MANUAL" in log and "OVERRIDE -> AUTONOMOUS" in log)
            if not all(results):
                print("\n--- backend output ---")
                print((tmp / "backend.out").read_text()[-4000:])

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
