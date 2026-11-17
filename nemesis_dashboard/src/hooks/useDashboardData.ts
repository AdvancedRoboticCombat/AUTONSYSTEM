import { useState, useEffect, useCallback } from 'react';
import { link } from '@/lib/link';
import { DashboardState } from '@/types/dashboard';


const INITIAL_DATA: DashboardState = {
  systemHealth: {
    vision: { fps: 0, nemConfidence: 0, oppConfidence: 0, track: "ok" },
    path: { validity: true },
  },
  imu: { rotation: 0, track: 0, linSpeed: 0 },
  telemetry: {
    driveL: { escVolt: 0, escCurr: 0, escTemp: 0, motorRPM: 0 },
    driveR: { escVolt: 0, escCurr: 0, escTemp: 0, motorRPM: 0 },
    arm:    { escVolt: 0, escCurr: 0, escTemp: 0, motorRPM: 0 },
    weapon: { escVolt: 0, escCurr: 0, escTemp: 0, motorRPM: 0 },
  },
  controls: { vision: [0, 0, 0], path: [0, 0] },
  machineCondition: { motors: [], wheels: [] },
  calculations: {
    WheelSlip: { odometrySpeed: 0, absVelDiff: 0 },
    weapon: { kineticEnergy: 0, kineticETransfer: 0 },
  },
  motorAlerts: { weapon_motor: false, weapon_esc: false, 
    arm_motor: false, arm_esc: false, 
    leftWheel_motor: false, leftWheel_esc: false, 
    rightWheel_motor: false, rightWheel_esc: false },
  override: false,
};

export const useDashboardData = () => {
  // Explicitly tell useState to use the Interface
  const [data, setData] = useState<DashboardState>(INITIAL_DATA);
  const [isConnected, setIsConnected] = useState(false);

  // Dashboard numbers from the backend (at most 10 a second, only when changed)
  useEffect(() => {
    setIsConnected(link.connected);
    const offs = [
      link.on('open', () => setIsConnected(true)),
      link.on('close', () => setIsConnected(false)),
      link.on('state', (newData) => setData((prev) => ({ ...prev, ...newData }))),
    ];
    return () => offs.forEach((off) => off());
  }, []);

  // Manual override: asks for the state the button shows (not a toggle), so a
  // repeated message cannot flip it back. The backend publishes it on
  // /nemesis/override and echoes it in the next state message.
  const manual = !!data.override;
  const handleOverride = useCallback(() => {
    link.send({ type: 'override', manual: !manual });
  }, [manual]);

  return { data, handleOverride, isConnected };

};