import { DashboardPanel } from './DashboardPanel';
import { TelemetryData } from '@/types/dashboard';

interface TelemetryPanelProps {
  data: TelemetryData;
}

type Channel = TelemetryData['driveL'];

function Cell({ label, value, unit }: { label: string; value: number; unit: string }) {
  return (
    <div className="stat">
      <div className="label !text-[10px]">{label}</div>
      <div className="stat-value !text-[18px]">{value}<span className="stat-unit">{unit}</span></div>
    </div>
  );
}

function ChannelBlock({ name, d }: { name: string; d: Channel }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="section-title">{name}</div>
      <div className="grid grid-cols-4 gap-1.5">
        <Cell label="Volt" value={d.escVolt} unit="V" />
        <Cell label="Curr" value={d.escCurr} unit="A" />
        <Cell label="Temp" value={d.escTemp} unit="°C" />
        <Cell label="Speed" value={d.motorRPM} unit="rpm" />
      </div>
    </div>
  );
}

export function TelemetryPanel({ data }: TelemetryPanelProps) {
  return (
    <DashboardPanel title="Telemetry" tag="03" aside="ESC + motor">
      <div className="flex flex-col gap-3 justify-between h-full">
        <ChannelBlock name="Drive left" d={data.driveL} />
        <ChannelBlock name="Drive right" d={data.driveR} />
        <ChannelBlock name="Arm" d={data.arm} />
        <ChannelBlock name="Weapon" d={data.weapon} />
      </div>
    </DashboardPanel>
  );
}
