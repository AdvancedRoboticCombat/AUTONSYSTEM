import { DashboardPanel } from './DashboardPanel';
import { MachineCondition, MotorAlerts } from '@/types/dashboard';

import defaultDiagram from './assets/base.png';
import weaponMotorAlert from './assets/motor-weapon.png';
import weaponESCAlert from './assets/vesc-1.png';
import armMotorAlert from './assets/motor-arm.png';
import armESCAlert from './assets/vesc-4.png';
import leftWheelMotorAlert from './assets/motor-left.png';
import leftWheelESCAlert from './assets/vesc-2.png';
import rightWheelMotorAlert from './assets/motor-right.png';
import rightWheelESCAlert from './assets/vesc-3.png';

interface MachineConditionPanelProps {
  data: MachineCondition;
  alerts: MotorAlerts;
}

// Each alert is a full-size overlay of the same drawing with one part lit
const OVERLAYS: { key: keyof MotorAlerts; src: string; name: string }[] = [
  { key: 'weapon_motor', src: weaponMotorAlert, name: 'Weapon motor' },
  { key: 'weapon_esc', src: weaponESCAlert, name: 'Weapon ESC' },
  { key: 'arm_motor', src: armMotorAlert, name: 'Arm motor' },
  { key: 'arm_esc', src: armESCAlert, name: 'Arm ESC' },
  { key: 'leftWheel_motor', src: leftWheelMotorAlert, name: 'Left motor' },
  { key: 'leftWheel_esc', src: leftWheelESCAlert, name: 'Left ESC' },
  { key: 'rightWheel_motor', src: rightWheelMotorAlert, name: 'Right motor' },
  { key: 'rightWheel_esc', src: rightWheelESCAlert, name: 'Right ESC' },
];

export function MachineConditionPanel({ alerts }: MachineConditionPanelProps) {
  const active = OVERLAYS.filter((o) => alerts[o.key]);
  return (
    <DashboardPanel title="Nemesis" tag="05" aside={active.length ? `${active.length} silent` : 'all reporting'}>
      <div className="flex flex-col h-full gap-2">
        <div className="relative flex-1 min-h-0">
          <img src={defaultDiagram} alt="Robot diagram" className="robot-base absolute inset-0 w-full h-full object-contain" />
          {active.map((o) => (
            <img key={o.key} src={o.src} alt={`${o.name} alert`} className="robot-alert absolute inset-0 w-full h-full object-contain pointer-events-none" />
          ))}
        </div>
        <div className="grid grid-cols-4 gap-x-2 gap-y-1 flex-shrink-0">
          {OVERLAYS.map((o) => (
            <div key={o.key} className="flex items-center gap-1.5 min-w-0">
              <span className={`w-1.5 h-1.5 rotate-45 flex-shrink-0 ${alerts[o.key] ? 'bg-status-critical animate-pulse' : 'bg-status-ok'}`} />
              <span className={`label !text-[10px] truncate ${alerts[o.key] ? '!text-[hsl(var(--foreground))]' : ''}`}>{o.name}</span>
            </div>
          ))}
        </div>
      </div>
    </DashboardPanel>
  );
}
