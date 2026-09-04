/**
 * Shared application-level TypeScript types for the AI Driver Safety Platform.
 * These are the domain types used throughout the UI — backed by the DB types in database.ts.
 */
import type {
  Database,
  TripStatus,
  RiskClass,
  Severity,
} from './database';

// ─── Re-export DB row types for convenience ────────────────────
export type Profile       = Database['public']['Tables']['profiles']['Row'];
export type Trip          = Database['public']['Tables']['trips']['Row'];
export type Frame         = Database['public']['Tables']['frames']['Row'];
export type Detection     = Database['public']['Tables']['detections']['Row'];
export type DrivingMetric = Database['public']['Tables']['driving_metrics']['Row'];
export type DrivingEvent  = Database['public']['Tables']['driving_events']['Row'];
export type RiskPrediction = Database['public']['Tables']['risk_predictions']['Row'];
export type ModelVersion  = Database['public']['Tables']['model_versions']['Row'];

// ─── Re-export enum types ──────────────────────────────────────
export type { TripStatus, RiskClass, Severity };

// ─── Trip with enriched data ───────────────────────────────────
export interface TripDetail extends Trip {
  frames?:          Frame[];
  detections?:      Detection[];
  driving_metrics?: DrivingMetric[];
  driving_events?:  DrivingEvent[];
  risk_predictions?: RiskPrediction[];
}

// ─── API ──────────────────────────────────────────────────────
export interface HealthResponse {
  status: 'ok' | 'error';
}

export interface ApiError {
  message: string;
  code?:   string;
  detail?: string;
}

// ─── Trip list filters ─────────────────────────────────────────
export interface TripFilters {
  status?:   TripStatus;
  search?:   string;
  page?:     number;
  pageSize?: number;
  sortBy?:   'created_at' | 'title' | 'status';
  sortDir?:  'asc' | 'desc';
}

// ─── Dashboard summary stats ───────────────────────────────────
export interface DashboardStats {
  totalTrips:        number;
  completedTrips:    number;
  avgRiskScore:      number | null;
  highRiskEvents:    number;
  totalDrivingEvents: number;
}

// ─── Navigation ───────────────────────────────────────────────
export interface NavLink {
  label:     string;
  href:      string;
  icon?:     string;
  external?: boolean;
}

// ─── Feature Card (Landing page) ──────────────────────────────
export interface Feature {
  title:       string;
  description: string;
  icon:        string;
  gradient:    string;
}

// ─── Upload form ──────────────────────────────────────────────
export interface TripUploadForm {
  title:     string;
  videoFile: File | null;
}

export type UploadStatus =
  | 'idle'
  | 'uploading'
  | 'creating'
  | 'error'
  | 'success';
