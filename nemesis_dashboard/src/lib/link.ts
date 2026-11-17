// The page's one connection to the backend: a WebSocket on the same address
// the page came from (/ws), reconnecting on its own. The message formats are
// described at the top of backend/server.py.
import type { DashboardState } from "@/types/dashboard";

export type Vec2 = { x: number; y: number };
export type Centroid = { cx: number; cy: number };
export type SegmentPolygon = [number, number][]; // [x, y] points in camera pixels

export type VisionMessage = {
  detW: number;
  detH: number;
  nemesis: Centroid | null;
  opponent: Centroid | null;
  nemesisHomog: Vec2 | null;
  opponentHomog: Vec2 | null;
  nemesisSegment: SegmentPolygon | null;
  opponentSegment: SegmentPolygon | null;
  invHomog: number[] | null; // flat 9-element row-major 3×3 inverse homography (arena→pixel)
  ts?: number;
};

export type ClientMessage =
  | { type: "ready" }
  | { type: "override"; manual: boolean }
  | { type: "action"; key: "swap_btn_pressed" | "fight_video_btn_pressed" }
  | { type: "canvas-click"; x: number; y: number };

type Events = {
  open: void;
  close: void;
  state: DashboardState;
  vision: VisionMessage;
  arc: Vec2[];
  frame: ArrayBuffer; // one JPEG; answer with send({ type: "ready" }) once it is shown
};

type Listener<K extends keyof Events> = (data: Events[K]) => void;

const RETRY_MS = [250, 500, 1000, 2000];

class Link {
  connected = false;
  private ws: WebSocket | null = null;
  private retries = 0;
  private listeners: { [K in keyof Events]: Set<Listener<K>> } = {
    open: new Set(),
    close: new Set(),
    state: new Set(),
    vision: new Set(),
    arc: new Set(),
    frame: new Set(),
  };

  constructor() {
    this.connect();
  }

  /** Subscribe; returns the unsubscribe function. */
  on<K extends keyof Events>(event: K, listener: Listener<K>): () => void {
    this.listeners[event].add(listener);
    return () => {
      this.listeners[event].delete(listener);
    };
  }

  /** Dropped when not connected: every message is a fresh request. */
  send(message: ClientMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private emit<K extends keyof Events>(event: K, data: Events[K]) {
    for (const listener of this.listeners[event]) listener(data);
  }

  private connect() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.retries = 0;
      this.connected = true;
      this.emit("open", undefined);
    };

    ws.onclose = () => {
      if (this.connected) {
        this.connected = false;
        this.emit("close", undefined);
      }
      const delay = RETRY_MS[Math.min(this.retries++, RETRY_MS.length - 1)];
      window.setTimeout(() => this.connect(), delay);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") {
        this.emit("frame", event.data as ArrayBuffer);
        return;
      }
      let message: { type?: string; data?: DashboardState; points?: Vec2[] };
      try {
        message = JSON.parse(event.data);
      } catch {
        return; // e.g. NaN from a degenerate homography: skip that one message
      }
      if (message.type === "state" && message.data) this.emit("state", message.data);
      else if (message.type === "vision") this.emit("vision", message as unknown as VisionMessage);
      else if (message.type === "arc") this.emit("arc", message.points ?? []);
    };
  }
}

export const link = new Link();
