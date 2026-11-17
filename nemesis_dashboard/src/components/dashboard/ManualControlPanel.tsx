import { DashboardPanel } from "./DashboardPanel";
import { ControlBars } from "@/types/dashboard";
import { link } from "@/lib/link";

interface ManualControlPanelProps {
  data: ControlBars;
  /** true while the driver has manual control */
  override?: boolean;
  onOverride?: () => void;
}

function ControlBarGroup({ label, values }: { label: string; values: number[] }) {
  return (
    <div className="flex-1 min-w-0">
      <div className="section-title mb-2">{label}</div>
      <div className="flex flex-col gap-2">
        {values.map((value, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="label !text-[10px] w-6">CH{i + 1}</span>
            <div className="meter flex-1"><div className="meter-fill" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ManualControlPanel({ data, override = false, onOverride }: ManualControlPanelProps) {
  const handleSwapDetect = () => {
    link.send({ type: "action", key: "swap_btn_pressed" });
  };

  return (
    <DashboardPanel title="Control" tag="04" aside={override ? "driver has the robot" : "policy has the robot"}>
      <div className="flex flex-col h-full gap-3">
        <div className="flex gap-8 flex-1 min-h-0">
          <ControlBarGroup label="Vision" values={data.vision} />
          <ControlBarGroup label="Path" values={data.path} />
        </div>

        <div className="flex items-center justify-center gap-6 pb-1 flex-shrink-0">
          <button className="cy-btn" onClick={handleSwapDetect} title="Swap which detected robot is Nemesis">
            <span>Swap detection</span>
          </button>
          <button
            className={`cy-btn cy-btn-lg ${override ? "cy-btn-hazard" : ""}`}
            onClick={onOverride}
            aria-pressed={override}
            title="Toggle between autonomous and manual (driver) control"
          >
            <span>{override ? "Manual — return to auto" : "Override — take manual"}</span>
          </button>
        </div>
      </div>
    </DashboardPanel>
  );
}
