import { DashboardPanel } from './DashboardPanel';
import { SystemHealth } from '@/types/dashboard';

interface SystemHealthPanelProps {
  data: SystemHealth;
}

function Stat({ label, value, unit }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="stat">
      <div className="label !text-[11px]">{label}</div>
      <div className="stat-value">{value}{unit && <span className="stat-unit">{unit}</span>}</div>
    </div>
  );
}

function Confidence({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div>
      <div className="kv">
        <span className="label">{label}</span>
        <span className="kv-lead" />
        <span className="kv-value">{pct.toFixed(0)}%</span>
      </div>
      <div className="meter"><div className="meter-fill" style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

export function SystemHealthPanel({ data }: SystemHealthPanelProps) {
  const v = data.vision;
  return (
    <DashboardPanel title="System Health" tag="02">
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Camera" value={v.captureFps ?? 0} unit="fps" />
          <Stat label="Vision" value={v.fps} unit="fps" />
          <Stat label="Latency" value={v.latencyMs ?? 0} unit="ms" />
        </div>
        <div className="section-title">Detection confidence</div>
        <div className="flex flex-col gap-2">
          <Confidence label="Nemesis" value={v.nemConfidence} />
          <Confidence label="Opponent" value={v.oppConfidence} />
        </div>
      </div>
    </DashboardPanel>
  );
}
