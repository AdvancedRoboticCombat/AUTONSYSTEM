// ROS2-compatible type definitions for dashboard data

export type HealthStatus = 'ok' | 'warning' | 'critical' | 'degraded';

export interface SystemHealth {
  vision: {
    fps: number;
    /** camera frames per second, and ms from grabbing a frame to publishing positions */
    captureFps?: number;
    latencyMs?: number;
    nemConfidence: number;
    oppConfidence: number;
    track: HealthStatus;
  };
  path: {
    validity: boolean;
  };
}

export interface IMUData {
  rotation: number;
  track: number;
  linSpeed: number;
}

export interface TelemetryData {
  driveL: {
    escVolt: number;
    escCurr: number; 
    escTemp: number; 
    motorRPM: number; 
  };
  driveR: {
    escVolt: number;
    escCurr: number; 
    escTemp: number; 
    motorRPM: number; 
  }
  arm: {
    escVolt: number; 
    escCurr: number;
    escTemp: number; 
    motorRPM: number;
  };
  weapon: {
    escVolt: number; 
    escCurr: number;
    escTemp: number; 
    motorRPM: number;
  };
}

export interface ControlBars {
  vision: number[];
  path: number[];
}

export interface MotorStatus {
  id: number;
  status: HealthStatus;
}

export interface WheelStatus {
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  status: HealthStatus;
}

export interface MachineCondition {
  motors: MotorStatus[];
  wheels: WheelStatus[];
}

export interface Calculations {
  WheelSlip: {
    odometrySpeed: number; 
    absVelDiff: number; 
  };
  weapon: {
    kineticEnergy: number;
    kineticETransfer: number; 
  };
}

export interface DashboardState {
  systemHealth: SystemHealth;
  imu: IMUData;
  telemetry: TelemetryData;
  controls: ControlBars;
  machineCondition: MachineCondition;
  calculations: Calculations;
  motorAlerts: MotorAlerts;
  /** true = manual (driver) control, false = autonomous. Mirrors /nemesis/override. */
  override?: boolean;
  liveStreamUrl?: string;
}

export interface MotorAlerts {
  weapon_motor: boolean;
  weapon_esc: boolean;
  arm_motor: boolean;
  arm_esc: boolean;
  leftWheel_motor: boolean;
  leftWheel_esc: boolean; 
  rightWheel_motor: boolean;
  rightWheel_esc: boolean;
}
