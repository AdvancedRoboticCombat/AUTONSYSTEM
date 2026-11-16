import { DashboardPanel } from "./DashboardPanel";
import { useEffect, useRef } from "react";
import { io, Socket } from "socket.io-client";
import { Maximize, Minimize } from "lucide-react";

type Centroid = { cx: number; cy: number };
type HomogCoord = { x: number; y: number };
type SegmentPolygon = [number, number][]; // array of [x, y] points in detection-space
type VisionCentroidsPayload = {
  detW: number;
  detH: number;
  nemesis: Centroid | null;
  opponent: Centroid | null;
  nemesisHomog: HomogCoord | null;
  opponentHomog: HomogCoord | null;
  nemesisSegment: SegmentPolygon | null;
  opponentSegment: SegmentPolygon | null;
  invHomog: number[] | null; // flat 9-element row-major 3×3 inverse homography (world→pixel)
  ts?: number;
};

type Vec2 = { x: number; y: number };

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpVec(curr: Vec2, target: Vec2, t: number): Vec2 {
  return { x: lerp(curr.x, target.x, t), y: lerp(curr.y, target.y, t) };
}

function containRect(detW: number, detH: number, cw: number, ch: number) {
  const scale = Math.min(cw / detW, ch / detH);
  const dw = detW * scale;
  const dh = detH * scale;
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;
  return { dx, dy, dw, dh, scale };
}

function drawDot(ctx: CanvasRenderingContext2D, p: Vec2, label: string) {
  if (label == "nemesis"){
    ctx.fillStyle = "#5FCC5F";
  }
  else{
    ctx.fillStyle = "red";
  }
  
  ctx.beginPath();
  ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.font = "12px monospace";
  const pad = 4;
  const textW = ctx.measureText(label).width;
  const bx = p.x + 8;
  const by = p.y - 18;

  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(bx, by, textW + pad * 2, 16);

  ctx.fillStyle = "white";
  ctx.fillText(label, bx + pad, by + 12);
}

const HOMOG_POINT_COUNT = 12;

function drawHomogLabel(ctx: CanvasRenderingContext2D, p: Vec2, label: string, color: string) {
  // Crosshair
  const size = 8;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p.x - size, p.y);
  ctx.lineTo(p.x + size, p.y);
  ctx.moveTo(p.x, p.y - size);
  ctx.lineTo(p.x, p.y + size);
  ctx.stroke();

  // Circle
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
  ctx.stroke();

  // Label box
  ctx.font = "11px monospace";
  const pad = 3;
  const textW = ctx.measureText(label).width;
  const bx = p.x + 12;
  const by = p.y - 14;
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(bx, by, textW + pad * 2, 15);
  ctx.fillStyle = color;
  ctx.fillText(label, bx + pad, by + 11);
}

function drawHomogPoint(ctx: CanvasRenderingContext2D, p: Vec2, index: number, total: number) {
  const radius = 6;

  // outer ring
  ctx.strokeStyle = "#FFD600";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.stroke();

  // filled centre
  ctx.fillStyle = index === total - 1 ? "#FFD600" : "rgba(255,214,0,0.5)";
  ctx.beginPath();
  ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
  ctx.fill();

  // numbered label
  const label = `H${index + 1}`;
  ctx.font = "11px monospace";
  const pad = 3;
  const textW = ctx.measureText(label).width;
  const bx = p.x + 10;
  const by = p.y - 14;

  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(bx, by, textW + pad * 2, 15);

  ctx.fillStyle = "#FFD600";
  ctx.fillText(label, bx + pad, by + 11);
}

/**
 * Apply a 3×3 homography (given as flat 9-element row-major array) to a 2-D
 * point, returning the projected result.  This is the JS equivalent of
 * cv2.perspectiveTransform for a single point.
 */
function applyHomography(H: number[], x: number, y: number): Vec2 {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

/** Draw a YOLO segmentation mask polygon on the canvas. */
function drawSegmentMask(
  ctx: CanvasRenderingContext2D,
  polygon: SegmentPolygon,
  dx: number,
  dy: number,
  scale: number,
  fillColor: string,
  strokeColor: string,
) {
  if (polygon.length < 3) return;

  ctx.save();
  ctx.beginPath();
  const [firstX, firstY] = polygon[0];
  ctx.moveTo(dx + firstX * scale, dy + firstY * scale);
  for (let i = 1; i < polygon.length; i++) {
    ctx.lineTo(dx + polygon[i][0] * scale, dy + polygon[i][1] * scale);
  }
  ctx.closePath();

  // Semi-transparent fill
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Outlined border
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();
}

interface LiveFeedPanelProps {
  expanded?: boolean;
  onToggleExpand?: () => void;
}

export function LiveFeedPanel({ expanded = false, onToggleExpand }: LiveFeedPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const lastBmpRef = useRef<ImageBitmap | null>(null);

  // FIX: Track whether a decode is already in flight.
  // Without this, every incoming frame kicks off an async decode, and they
  // pile up. Older decodes finishing after newer ones cause frames to render
  // out of order and the video appears to play in slow motion.
  const decodeInFlightRef = useRef(false);

  const detSizeRef = useRef<{ w: number; h: number }>({ w: 480, h: 270 });

  const targetNemRef = useRef<Vec2 | null>(null);
  const targetOppRef = useRef<Vec2 | null>(null);

  const currNemRef = useRef<Vec2 | null>(null);
  const currOppRef = useRef<Vec2 | null>(null);

  // Homography calibration points stored in detection-space coords
  const homogPointsRef = useRef<Vec2[]>([]);

  // Homographied world-space coords from backend (metres)
  const nemHomogRef = useRef<{ x: number; y: number } | null>(null);
  const oppHomogRef = useRef<{ x: number; y: number } | null>(null);

  // Inverse homography matrix (world→pixel), flat 9-element row-major 3×3
  const invHomogRef = useRef<number[] | null>(null);

  // YOLO segmentation mask polygons (detection-space coords)
  const nemSegRef = useRef<SegmentPolygon | null>(null);
  const oppSegRef = useRef<SegmentPolygon | null>(null);

  // Predicted trajectory arc from policy engine (world-space metres)
  const predictedArcRef = useRef<Vec2[]>([]);

  const socketRef = useRef<Socket | null>(null);

  // Click handler stored in a ref — assigned inside useEffect, called from
  // the React onClick on the <canvas>. This way the handler always fires on
  // whichever canvas element is currently mounted (expanded or not).
  const handleClickRef = useRef<(e: React.MouseEvent<HTMLCanvasElement>) => void>(() => {});

  useEffect(() => {
    const socket: Socket = io("http://localhost:3001", {
      transports: ["websocket"],
    });
    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("✅ Socket connected:", socket.id);
    });

    socket.on("connect_error", (err) => {
      console.error("❌ Socket connection error:", err);
    });

    // FIX: Single consolidated handler for video-frame (was registered twice before,
    // which meant two parallel decode pipelines ran for every frame)
    socket.on("video-frame", async (data: ArrayBuffer) => {
      // FIX: Drop this frame if we're still decoding the previous one.
      // This keeps us at real-time by skipping frames rather than queuing them.
      if (decodeInFlightRef.current) return;

      decodeInFlightRef.current = true;
      try {
        const blob = new Blob([data], { type: "image/jpeg" });
        const bmp = await createImageBitmap(blob);

        const prev = lastBmpRef.current;
        lastBmpRef.current = bmp;
        if (prev) prev.close();
      } finally {
        decodeInFlightRef.current = false;
      }
    });

    socket.on("vision-centroids", (payload: VisionCentroidsPayload) => {
      console.log("vision-centroids received", payload);
      detSizeRef.current = { w: payload.detW, h: payload.detH };

      targetNemRef.current = payload.nemesis
        ? { x: payload.nemesis.cx, y: payload.nemesis.cy }
        : null;

      targetOppRef.current = payload.opponent
        ? { x: payload.opponent.cx, y: payload.opponent.cy }
        : null;

      nemHomogRef.current = payload.nemesisHomog ?? null;
      oppHomogRef.current = payload.opponentHomog ?? null;

      // Inverse homography matrix (world→pixel)
      invHomogRef.current = payload.invHomog ?? null;

      // YOLO segmentation mask polygons
      nemSegRef.current = payload.nemesisSegment ?? null;
      oppSegRef.current = payload.opponentSegment ?? null;

      if (targetNemRef.current && !currNemRef.current) {
        currNemRef.current = { ...targetNemRef.current };
      }
      if (targetOppRef.current && !currOppRef.current) {
        currOppRef.current = { ...targetOppRef.current };
      }
    });

    // Predicted trajectory arc from policy engine
    socket.on("predicted-arc", (arc: Vec2[]) => {
      predictedArcRef.current = arc ?? [];
    });

    let rafId = 0;

    const drawLoop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const bmp = lastBmpRef.current;

      if (canvas && ctx) {
        const cssW = canvas.clientWidth;
        const cssH = canvas.clientHeight;
        if (canvas.width !== cssW || canvas.height !== cssH) {
          canvas.width = cssW;
          canvas.height = cssH;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (bmp) {
          const { w: detW, h: detH } = detSizeRef.current;
          const { dx, dy, dw, dh, scale } = containRect(detW, detH, canvas.width, canvas.height);
          ctx.drawImage(bmp, dx, dy, dw, dh);

          // Draw YOLO segmentation masks (behind the centroid dots)
          const nemSeg = nemSegRef.current;
          const oppSeg = oppSegRef.current;
          if (nemSeg) {
            drawSegmentMask(ctx, nemSeg, dx, dy, scale, "rgba(95,204,95,0.25)", "rgba(95,204,95,0.7)");
          }
          if (oppSeg) {
            drawSegmentMask(ctx, oppSeg, dx, dy, scale, "rgba(255,60,60,0.25)", "rgba(255,60,60,0.7)");
          }

          const alpha = 0.2;
          const tNem = targetNemRef.current;
          const tOpp = targetOppRef.current;

          if (tNem && currNemRef.current) currNemRef.current = lerpVec(currNemRef.current, tNem, alpha);
          if (tOpp && currOppRef.current) currOppRef.current = lerpVec(currOppRef.current, tOpp, alpha);

          if (currNemRef.current) {
            const p = {
              x: dx + currNemRef.current.x * scale,
              y: dy + currNemRef.current.y * scale,
            };
            drawDot(ctx, p, "nemesis");
          }

          if (currOppRef.current) {
            const p = {
              x: dx + currOppRef.current.x * scale,
              y: dy + currOppRef.current.y * scale,
            };
            drawDot(ctx, p, "opponent");
          }

          // Draw homography calibration points
          const hPts = homogPointsRef.current;
          for (let i = 0; i < hPts.length; i++) {
            const hp = {
              x: dx + hPts[i].x * scale,
              y: dy + hPts[i].y * scale,
            };
            drawHomogPoint(ctx, hp, i, hPts.length);
          }

          // Draw homographied world-space positions back onto the main
          // video feed by applying the inverse homography (world→pixel).
          const nemH = nemHomogRef.current;
          const oppH = oppHomogRef.current;
          const invH = invHomogRef.current;

          if (invH && (nemH !== null || oppH !== null)) {
            // Convert world-space → detection-pixel-space → canvas-space
            const worldToCanvas = (wx: number, wy: number): Vec2 => {
              const det = applyHomography(invH, wx, wy);
              return { x: dx + det.x * scale, y: dy + det.y * scale };
            };

            if (nemH) {
              const p = worldToCanvas(nemH.x, nemH.y);
              drawHomogLabel(ctx, p, `N (${nemH.x.toFixed(2)},${nemH.y.toFixed(2)})`, "#5FCC5F");
            }
            if (oppH) {
              const p = worldToCanvas(oppH.x, oppH.y);
              drawHomogLabel(ctx, p, `O (${oppH.x.toFixed(2)},${oppH.y.toFixed(2)})`, "red");
            }

            // Draw predicted trajectory arc from the policy engine
            const arc = predictedArcRef.current;
            if (arc.length >= 2) {
              ctx.save();
              ctx.beginPath();
              const p0 = worldToCanvas(arc[0].x, arc[0].y);
              ctx.moveTo(p0.x, p0.y);
              for (let i = 1; i < arc.length; i++) {
                const pi = worldToCanvas(arc[i].x, arc[i].y);
                ctx.lineTo(pi.x, pi.y);
              }
              // Gradient fade: bright near robot → dim at end
              const pLast = worldToCanvas(arc[arc.length - 1].x, arc[arc.length - 1].y);
              const grad = ctx.createLinearGradient(p0.x, p0.y, pLast.x, pLast.y);
              grad.addColorStop(0, "rgba(0,200,255,0.9)");
              grad.addColorStop(1, "rgba(0,200,255,0.15)");
              ctx.strokeStyle = grad;
              ctx.lineWidth = 2;
              ctx.stroke();

              // Arrowhead at the tip
              if (arc.length >= 2) {
                const last = worldToCanvas(arc[arc.length - 1].x, arc[arc.length - 1].y);
                const prev = worldToCanvas(arc[arc.length - 2].x, arc[arc.length - 2].y);
                const angle = Math.atan2(last.y - prev.y, last.x - prev.x);
                const headLen = 8;
                ctx.beginPath();
                ctx.moveTo(last.x, last.y);
                ctx.lineTo(last.x - headLen * Math.cos(angle - 0.5), last.y - headLen * Math.sin(angle - 0.5));
                ctx.moveTo(last.x, last.y);
                ctx.lineTo(last.x - headLen * Math.cos(angle + 0.5), last.y - headLen * Math.sin(angle + 0.5));
                ctx.strokeStyle = "rgba(0,200,255,0.7)";
                ctx.lineWidth = 2;
                ctx.stroke();
              }
              ctx.restore();
            }
          }
        } else {
          ctx.fillStyle = "rgba(255,60,90,0.9)";
          ctx.font = "14px 'Share Tech Mono', monospace";
          ctx.fillText("NO SIGNAL  //  waiting for camera stream", 14, 26);
        }
      }

      rafId = requestAnimationFrame(drawLoop);
    };

    rafId = requestAnimationFrame(drawLoop);

    // ============================================================
    // CANVAS CLICK — handler stored in a ref so the React onClick
    // on the <canvas> always calls the latest version, even after
    // the component re-renders (e.g. expand/collapse).
    // ============================================================
    handleClickRef.current = (e: React.MouseEvent<HTMLCanvasElement>) => {
      console.log("Canvas clicked, raw:", e.clientX, e.clientY);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      // Map canvas-pixel → detection-pixel using the same transform
      // the draw loop uses (containRect).
      const { w: detW, h: detH } = detSizeRef.current;
      const { dx, dy, scale } = containRect(detW, detH, canvas.width, canvas.height);
      const detX = (cx - dx) / scale;
      const detY = (cy - dy) / scale;

      console.log("Emitting canvas-click:", { x: detX, y: detY });

      // Mirror backend logic: keep at most HOMOG_POINT_COUNT points
      const pts = homogPointsRef.current;
      if (pts.length >= HOMOG_POINT_COUNT) {
        pts.shift();
      }
      pts.push({ x: detX, y: detY });

      socketRef.current?.emit("canvas-click", { x: detX, y: detY });
    };
    // ============================================================

    return () => {
      cancelAnimationFrame(rafId);
      socket.disconnect();

      const prev = lastBmpRef.current;
      if (prev) prev.close();
      lastBmpRef.current = null;
    };
  }, []);

  const wrapper = expanded
    ? "fixed inset-0 z-[9999] bg-background flex flex-col"
    : "";

  const feedButton =
    "border border-primary/60 bg-background/80 backdrop-blur text-primary hover:bg-primary hover:text-black transition-colors";

  const inner = (
    <div className={`relative w-full h-full bg-black/60 border border-primary/25 overflow-hidden ${expanded ? "flex-1" : ""}`}>
      <canvas ref={canvasRef} className="w-full h-full" onClick={(e) => handleClickRef.current(e)} />
      <div className="absolute top-2 right-2 flex gap-2" style={{ zIndex: 10000 }}>
        <button
          onClick={() => socketRef.current?.emit("trigger-action", { key: "fight_video_btn_pressed" })}
          className={`${feedButton} px-2 text-xs uppercase tracking-widest`}
          title="Play the recorded fight through the detector instead of the camera (press again to restart)"
        >
          Run fight video
        </button>
        {onToggleExpand && (
          <button
            onClick={onToggleExpand}
            className={`${feedButton} p-1.5`}
            title={expanded ? "Collapse camera" : "Expand camera"}
          >
            {expanded ? <Minimize size={16} /> : <Maximize size={16} />}
          </button>
        )}
      </div>
    </div>
  );

  if (expanded) {
    return (
      <div className={wrapper}>
        <div className="px-3 pt-3 pb-1 flex-shrink-0">
          <h2 className="panel-header text-sm">LIVE FEED</h2>
        </div>
        {inner}
      </div>
    );
  }

  return (
    <DashboardPanel title="Live Feed" tag="01" aside="click the feed to set arena points">
      {inner}
    </DashboardPanel>
  );
}