"""The page and its live link: one aiohttp server in its own thread.

    GET /ws     the page's WebSocket (messages below)
    GET /...    the built page, from nemesis_dashboard/dist/

Server -> page:
    text    {"type": "state", "data": {...}}    dashboard numbers: on connect, then at most
                                                STATE_HZ times a second, only when changed
    text    {"type": "vision", ...}             robot positions and outlines (newest only)
    text    {"type": "arc", "points": [...]}    the policy's predicted path (newest only)
    binary  one JPEG camera frame               only while the page holds the credit

Page -> server:
    {"type": "ready"}                           frame shown: send the next one (the credit)
    {"type": "override", "manual": true|false}
    {"type": "action", "key": "swap_btn_pressed" | "fight_video_btn_pressed"}
    {"type": "canvas-click", "x": .., "y": ..}  pixels of the camera frame

Video is credit-based: a page gets one frame, then the next only after it has
said "ready". Nothing queues behind a slow browser, and no frame is encoded
while no page is watching. Every other message is "newest wins" per page.

The page carries the OVERRIDE switch and has no password, so:
  * the server listens on this computer only (config.HOST),
  * every request must name an allowed host (stops DNS rebinding),
  * a WebSocket opened by a page of any other site is refused,
  * the page may not be framed by another site (clickjacking).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional
from urllib.parse import urlsplit

from aiohttp import WSCloseCode, WSMsgType, web

from .config import DIST_DIR, STATUS_PRINT_S

if TYPE_CHECKING:
    from .app import Dashboard

log = logging.getLogger("nemesis.server")
audit = logging.getLogger("nemesis.audit")

STATE_HZ = 10                  # most dashboard-state messages per second
MAX_MESSAGE_BYTES = 64 * 1024  # longest message a page may send
MAX_PAGES = 8                  # open WebSockets at once
ACTIONS_PER_S = 10             # override / action / canvas-click messages per page per second
CREDIT_TIMEOUT_S = 2.0         # a page that never said "ready" gets a new frame after this
ACTIONS = ("swap_btn_pressed", "fight_video_btn_pressed")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'{ws}; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'"
)

NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Nemesis Dashboard</title></head>
<body style="background:#111;color:#eee;font:16px monospace;padding:2em">
<h1>The dashboard page is not built</h1>
<p>Vision and ROS are running. To get the page, run <code>scripts/setup_env.sh</code>
(or <code>npm run build</code> in <code>nemesis_dashboard/</code>) and reload.</p>
</body></html>
"""


def host_name(value: Optional[str]) -> Optional[str]:
    """The name in a Host header: 'localhost:8080' -> 'localhost', '[::1]:8080' -> '::1'.
    None for anything malformed."""
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return None
        name, rest = value[1:end], value[end + 1:]
    else:
        name, colon, port = value.partition(":")
        rest = colon + port
    if rest and not (rest[0] == ":" and rest[1:].isdigit()):
        return None
    return name or None


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Page:
    """One open WebSocket and what is waiting to be sent to it."""

    def __init__(self, ws: web.WebSocketResponse, peer: str):
        self.ws = ws
        self.peer = peer
        self.wake = asyncio.Event()
        self.texts: Dict[str, str] = {}   # kind -> newest message of that kind
        self.frame: Optional[bytes] = None
        self.credit = True                # may receive one frame
        self.frame_given_at = 0.0
        self._actions = deque()
        self._bad = 0
        self._bad_noted = 0.0

    def allow_action(self, now: float) -> bool:
        while self._actions and now - self._actions[0] > 1.0:
            self._actions.popleft()
        if len(self._actions) >= ACTIONS_PER_S:
            return False
        self._actions.append(now)
        return True

    def bad(self, reason: str):
        """An unusable message: ignored, and noted at most every few seconds."""
        self._bad += 1
        now = time.monotonic()
        if now - self._bad_noted >= STATUS_PRINT_S:
            log.warning("Ignored %d message(s) from the page at %s; last: %s", self._bad, self.peer, reason)
            self._bad = 0
            self._bad_noted = now


class WebServer:
    def __init__(self, app: "Dashboard"):
        self.app = app
        cfg = app.config
        self.host = cfg.host
        self.port = cfg.port
        self.allowed = cfg.allowed_hosts()

        # Set while some page holds a credit: the stream loop encodes a frame only then
        self.frame_wanted = threading.Event()

        self.pages = set()
        self.page_count = 0  # read from other threads
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._error: Optional[BaseException] = None
        self._refused = 0
        self._refused_noted = 0.0

    # ---------- start / stop (main thread) ----------

    def start(self):
        self._thread = threading.Thread(target=self._run, name="web", daemon=True)
        self._thread.start()
        self._started.wait()
        if self._error is not None:
            raise SystemExit(f"Cannot start the dashboard on {self.host}:{self.port}: {self._error} "
                             "(is another dashboard still running?)")

    def stop(self):
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        done = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), loop)
        try:
            done.result(timeout=5)
        except Exception as error:
            log.warning("Web server did not stop cleanly: %r", error)
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=5)

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._setup())
        except OSError as error:  # port taken, address not on this computer...
            self._error = error
            self._started.set()
            loop.close()
            return
        self._started.set()
        tick = loop.create_task(self._tick())
        loop.run_forever()
        tick.cancel()
        loop.run_until_complete(asyncio.gather(tick, return_exceptions=True))
        loop.close()

    async def _setup(self):
        app = web.Application(middlewares=[self._check_host], client_max_size=MAX_MESSAGE_BYTES)
        app.on_response_prepare.append(self._add_headers)
        app.on_shutdown.append(self._close_pages)
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/{path:.*}", self._static)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    async def _close_pages(self, _app):
        for page in list(self.pages):
            await page.ws.close(code=WSCloseCode.GOING_AWAY, message=b"dashboard stopping")

    # ---------- what other threads call ----------

    def publish_vision(self, payload: dict):
        if self.page_count:
            self._call(self._deliver_text, "vision", json.dumps({"type": "vision", **payload}))

    def publish_arc(self, points: list):
        if self.page_count:
            self._call(self._deliver_text, "arc", json.dumps({"type": "arc", "points": points}))

    def publish_frame(self, jpeg: bytes):
        self._call(self._deliver_frame, jpeg)

    def push_state(self):
        """Send the dashboard state now if it changed, without waiting for the next tick."""
        self._call(self._push_state)

    def _call(self, function, *args):
        loop = self._loop
        if loop is None or not self.page_count:
            return
        try:
            loop.call_soon_threadsafe(function, *args)
        except RuntimeError:  # the loop closed while the dashboard was stopping
            pass

    # ---------- event loop thread ----------

    def _deliver_text(self, kind: str, text: str):
        for page in self.pages:
            page.texts[kind] = text
            page.wake.set()

    def _deliver_frame(self, jpeg: bytes):
        now = time.monotonic()
        for page in self.pages:
            if page.credit:
                page.credit = False
                page.frame = jpeg
                page.frame_given_at = now
                page.wake.set()
        self._update_frame_wanted()

    def _push_state(self):
        text = self.app.state.changed_json()
        if text is not None:
            self._deliver_text("state", text)

    def _update_frame_wanted(self):
        if any(page.credit for page in self.pages):
            self.frame_wanted.set()
        else:
            self.frame_wanted.clear()

    async def _tick(self):
        """Dashboard state at STATE_HZ, the silent-component watchdog, and
        credits for pages that never answered."""
        while True:
            await asyncio.sleep(1.0 / STATE_HZ)
            try:
                self.app.state.refresh_alerts(time.time())
                self._push_state()
                now = time.monotonic()
                stuck = [p for p in self.pages if not p.credit and now - p.frame_given_at > CREDIT_TIMEOUT_S]
                for page in stuck:
                    page.credit = True
                if stuck:
                    self._update_frame_wanted()
            except Exception:
                log.exception("Dashboard state update failed")

    # ---------- HTTP ----------

    def _refuse(self, request: web.Request, reason: str):
        """Logs a refused request, at most every few seconds."""
        self._refused += 1
        now = time.monotonic()
        if now - self._refused_noted >= STATUS_PRINT_S:
            audit.warning("Refused %d request(s); last from %s: %s", self._refused, request.remote, reason)
            self._refused = 0
            self._refused_noted = now

    @web.middleware
    async def _check_host(self, request: web.Request, handler):
        if host_name(request.headers.get("Host")) not in self.allowed:
            self._refuse(request, f"addressed to {request.headers.get('Host')!r}")
            raise web.HTTPMisdirectedRequest(text="This dashboard only answers on this computer's own address.\n")
        return await handler(request)

    async def _add_headers(self, request: web.Request, response: web.StreamResponse):
        host = request.headers.get("Host", "")
        ws = f" ws://{host}" if host_name(host) in self.allowed else ""
        headers = response.headers
        headers.setdefault("Content-Security-Policy", CSP.format(ws=ws))
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

    async def _static(self, request: web.Request):
        dist = DIST_DIR.resolve()
        path = request.match_info.get("path", "")
        if path:
            try:
                target = (dist / path).resolve()
                found = target.is_relative_to(dist) and target.is_file()
            except (ValueError, OSError):
                found = False
            if found:
                response = web.FileResponse(target)
                hashed = path.startswith("assets/")  # Vite puts a content hash in these names
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if hashed else "no-cache"
                return response
            if "." in path.rsplit("/", 1)[-1]:
                raise web.HTTPNotFound()
        index = dist / "index.html"
        if index.is_file():
            response = web.FileResponse(index)
            response.headers["Cache-Control"] = "no-cache"
            return response
        return web.Response(text=NOT_BUILT, content_type="text/html", headers={"Cache-Control": "no-store"})

    # ---------- WebSocket ----------

    def _origin_allowed(self, origin: str) -> bool:
        try:
            parts = urlsplit(origin)
            return parts.scheme in ("http", "https") and parts.hostname in self.allowed
        except ValueError:
            return False

    async def _websocket(self, request: web.Request):
        origin = request.headers.get("Origin")
        # Browsers always send Origin; local tools (smoke test, scripts) may not
        if origin is not None and not self._origin_allowed(origin):
            self._refuse(request, f"WebSocket from a page of another site ({origin})")
            raise web.HTTPForbidden(text="Only the dashboard's own page may connect.\n")
        if len(self.pages) >= MAX_PAGES:
            self._refuse(request, f"already {MAX_PAGES} pages open")
            raise web.HTTPServiceUnavailable(text="Too many dashboard pages are open.\n")

        ws = web.WebSocketResponse(max_msg_size=MAX_MESSAGE_BYTES, compress=False, heartbeat=5.0)
        if not ws.can_prepare(request).ok:
            raise web.HTTPBadRequest(text="WebSocket only.\n")
        await ws.prepare(request)

        page = Page(ws, request.remote or "?")
        page.texts["state"] = self.app.state.to_json()
        page.wake.set()
        self.pages.add(page)
        self.page_count = len(self.pages)
        self._update_frame_wanted()
        audit.info("Page connected from %s (%d open)", page.peer, self.page_count)

        sender = asyncio.ensure_future(self._send_to(page))
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    self._on_text(page, message.data)
                elif message.type == WSMsgType.ERROR:
                    log.warning("Closed the page at %s: %s", page.peer, ws.exception())
                    break
                else:
                    page.bad("binary message")
        finally:
            sender.cancel()
            self.pages.discard(page)
            self.page_count = len(self.pages)
            self._update_frame_wanted()
            audit.info("Page disconnected from %s (%d open)", page.peer, self.page_count)
        return ws

    async def _send_to(self, page: Page):
        ws = page.ws
        try:
            while not ws.closed:
                await page.wake.wait()
                page.wake.clear()
                texts, page.texts = page.texts, {}
                frame, page.frame = page.frame, None
                for kind in ("state", "vision", "arc"):
                    if kind in texts:
                        await ws.send_str(texts[kind])
                if frame is not None:
                    await ws.send_bytes(frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the page went away mid-send
            log.debug("Stopped sending to %s: %r", page.peer, error)

    def _on_text(self, page: Page, text: str):
        try:
            message = json.loads(text)
        except ValueError:
            return page.bad("not JSON")
        if not isinstance(message, dict):
            return page.bad("not a JSON object")

        kind = message.get("type")
        if kind == "ready":
            if not page.credit:
                page.credit = True
                self.frame_wanted.set()
            return
        if kind not in ("override", "action", "canvas-click"):
            return page.bad(f"unknown type {str(kind)[:40]!r}")
        if not page.allow_action(time.monotonic()):
            return page.bad(f"more than {ACTIONS_PER_S} actions a second")

        try:
            if kind == "override":
                manual = message.get("manual")
                if not isinstance(manual, bool):
                    return page.bad("override without a true/false 'manual'")
                self.app.set_override(manual, who=page.peer)
            elif kind == "action":
                key = message.get("key")
                if key not in ACTIONS:
                    return page.bad(f"unknown action {str(key)[:40]!r}")
                self.app.handle_action(key, who=page.peer)
            else:
                x, y = message.get("x"), message.get("y")
                if not (is_number(x) and is_number(y)):
                    return page.bad("canvas-click without numeric x and y")
                self.app.canvas_click(float(x), float(y))
        except Exception:
            # Logged here, never sent to the page
            log.exception("Handling %s from %s failed", kind, page.peer)
