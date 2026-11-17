import { LiveFeedPanel } from '@/components/dashboard/LiveFeedPanel';
import { SystemHealthPanel } from '@/components/dashboard/SystemHealthPanel';
import { IMUPanel } from '@/components/dashboard/IMUPanel';
import { ManualControlPanel } from '@/components/dashboard/ManualControlPanel';
import { MachineConditionPanel } from '@/components/dashboard/MachineConditionPanel';
import { TelemetryPanel } from '@/components/dashboard/TelemetryPanel';
import { CalculationsPanel } from '@/components/dashboard/CalculationsPanel';
import { useDashboardData } from '@/hooks/useDashboardData';
import { useCallback, useEffect, useState } from 'react';

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toLocaleTimeString('en-GB');
}

const Dashboard = () => {
  const { data, handleOverride, isConnected } = useDashboardData();
  // Open the page as /?expand to start with the live feed filling the window.
  const [feedExpanded, setFeedExpanded] = useState(() => new URLSearchParams(window.location.search).has('expand'));
  const clock = useClock();
  const manual = !!data.override;

  const toggleFeedExpanded = useCallback(() => setFeedExpanded(prev => !prev), []);

  // When the live feed is expanded it renders as a fixed full-page overlay
  // from inside LiveFeedPanel itself, so we still render it here as well
  // to keep its canvas and subscriptions alive.

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col p-2 gap-2">
      {/* Top bar: name, who is driving, link state */}
      <header className="flex items-center gap-4 px-2 h-11 flex-shrink-0">
        <span className="brand">NEMESIS</span>
        <span className="label hidden md:inline">// autonomy control</span>
        <span className="panel-rule" />
        <div className={`mode-pill ${manual ? 'mode-pill-manual' : ''}`} role="status">
          <span>{manual ? 'MANUAL OVERRIDE' : 'AUTONOMOUS'}</span>
        </div>
        <span className="panel-rule" style={{ transform: 'scaleX(-1)' }} />
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rotate-45 ${isConnected ? 'bg-status-ok' : 'bg-status-critical'} animate-pulse`} />
          <span className={`label ${isConnected ? '!text-[hsl(var(--status-ok))]' : ''}`}>{isConnected ? 'Link up' : 'No link'}</span>
        </div>
        <span className="kv-value text-sm w-[72px] text-right">{clock}</span>
      </header>

      {/* 3 columns: readouts | camera + control | robot */}
      <div className="grid flex-1 min-h-0 gap-2 grid-cols-[minmax(0,1fr)_minmax(0,2.1fr)_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.82fr)]">
        <div className="min-h-0 col-start-1 row-start-1">
          <SystemHealthPanel data={data.systemHealth} />
        </div>
        <div className="min-h-0 col-start-1 row-start-2 row-span-2">
          <TelemetryPanel data={data.telemetry} />
        </div>

        <div className="min-h-0 col-start-2 row-start-1 row-span-2">
          <LiveFeedPanel expanded={feedExpanded} onToggleExpand={toggleFeedExpanded} />
        </div>
        <div className="min-h-0 col-start-2 row-start-3">
          <ManualControlPanel data={data.controls} override={data.override} onOverride={handleOverride} />
        </div>

        <div className="min-h-0 col-start-3 row-start-1 row-span-2">
          <MachineConditionPanel data={data.machineCondition} alerts={data.motorAlerts} />
        </div>
        <div className="min-h-0 col-start-3 row-start-3 grid grid-cols-2 gap-2">
          <div className="min-h-0 min-w-0"><IMUPanel data={data.imu} /></div>
          <div className="min-h-0 min-w-0"><CalculationsPanel data={data.calculations} /></div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
