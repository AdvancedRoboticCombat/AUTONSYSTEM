import { DashboardPanel } from './DashboardPanel';
import { IMUData } from '@/types/dashboard';

interface IMUPanelProps {
  data: IMUData;
}

const TICKS = Array.from({ length: 24 }, (_, i) => i * 15);

export function IMUPanel({ data }: IMUPanelProps) {
  const rot = ((data.rotation % 360) + 360) % 360;
  return (
    <DashboardPanel title="IMU" tag="06">
      <div className="flex flex-col items-center justify-center gap-2 h-full">
        <svg viewBox="-50 -50 100 100" className="w-full max-w-[150px] min-h-0 flex-1" role="img" aria-label={`Heading ${rot.toFixed(0)} degrees`}>
          <circle r="46" fill="none" stroke="hsl(350 100% 50% / 0.5)" strokeWidth="1" />
          <circle r="30" fill="none" stroke="hsl(350 100% 50% / 0.22)" strokeWidth="1" />
          <circle r="14" fill="none" stroke="hsl(350 100% 50% / 0.14)" strokeWidth="1" />
          {TICKS.map((a) => (
            <line key={a} x1="0" y1={a % 90 === 0 ? -46 : -46} x2="0" y2={a % 90 === 0 ? -38 : -42}
              stroke={a % 90 === 0 ? 'hsl(350 20% 94%)' : 'hsl(350 100% 55% / 0.7)'} strokeWidth={a % 90 === 0 ? 1.6 : 0.8}
              transform={`rotate(${a})`} />
          ))}
          <g transform={`rotate(${rot})`} style={{ transition: 'transform 0.15s linear' }}>
            <polygon points="0,-36 5,4 0,0 -5,4" fill="hsl(350 100% 52%)" style={{ filter: 'drop-shadow(0 0 4px hsl(350 100% 55%))' }} />
          </g>
          <circle r="2" fill="hsl(350 20% 94%)" />
        </svg>
        <div className="stat-value !text-[20px] flex-shrink-0">{rot.toFixed(1)}<span className="stat-unit">deg</span></div>
      </div>
    </DashboardPanel>
  );
}
