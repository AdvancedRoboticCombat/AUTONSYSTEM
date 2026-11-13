import { DashboardPanel } from './DashboardPanel';
import { Calculations } from '@/types/dashboard';

interface CalculationsPanelProps {
  data: Calculations;
}

function Row({ label, value, unit }: { label: string; value: number; unit: string }) {
  return (
    <div className="kv">
      <span className="label">{label}</span>
      <span className="kv-lead" />
      <span className="kv-value">{Number(value).toFixed(1)} {unit}</span>
    </div>
  );
}

export function CalculationsPanel({ data }: CalculationsPanelProps) {
  return (
    <DashboardPanel title="Calc" tag="07">
      <div className="flex flex-col gap-2">
        <div className="section-title">Wheel slip</div>
        <div>
          <Row label="Odometry" value={data.WheelSlip.odometrySpeed} unit="m/s" />
          <Row label="Vel diff" value={data.WheelSlip.absVelDiff} unit="m/s" />
        </div>
        <div className="section-title">Weapon</div>
        <div>
          <Row label="Energy" value={data.weapon.kineticEnergy} unit="J" />
          <Row label="Transfer" value={data.weapon.kineticETransfer} unit="J" />
        </div>
      </div>
    </DashboardPanel>
  );
}
