import { useState, useEffect, useCallback } from 'react';
import { socket } from '@/socket';
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

  // Simulate real-time updates - Replace with ROS2 subscription
  useEffect(() => {
    function onConnect() {
      setIsConnected(true);
      console.log('Connected to Robot');
    }

    function onDisconnect() {
      setIsConnected(false);
      console.log('Disconnected from Robot');
    }

    function onDataUpdate(newData: any) {
      setData((prev) => ({ ...prev, ...newData }));
    }

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('dashboard-update', onDataUpdate);


    if (socket.connected) {
      onConnect();
    }

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('dashboard-update', onDataUpdate);
    };
  }, []);

  // Manual override toggle: the backend flips its state, publishes it on
  // /nemesis/override and echoes it back in the next dashboard-update.
  const handleOverride = useCallback(() => {
    socket.emit('trigger-action', { key: 'override_btn_pressed' });
  }, []);

  return { data, handleOverride, isConnected };

};