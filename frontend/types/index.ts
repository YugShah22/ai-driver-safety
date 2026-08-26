/**
 * Shared TypeScript type definitions for the AI Driver Safety Platform.
 * Phase 2+ will expand these as backend schemas are implemented.
 */

// ─── Risk Assessment ──────────────────────────────────────────
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface RiskAssessment {
  id: string;
  sessionId: string;
  riskLevel: RiskLevel;
  riskScore: number; // 0–100
  createdAt: string;
  summary: string;
  recommendations: string[];
}

// ─── Analysis Session ─────────────────────────────────────────
export type SessionStatus =
  | 'PENDING'
  | 'UPLOADING'
  | 'PROCESSING'
  | 'COMPLETE'
  | 'ERROR';

export interface AnalysisSession {
  id: string;
  userId: string;
  videoUrl: string;
  status: SessionStatus;
  createdAt: string;
  updatedAt: string;
  assessment?: RiskAssessment;
}

// ─── Detected Object ──────────────────────────────────────────
export interface DetectedObject {
  label: string;
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  frameIndex: number;
}

// ─── API ──────────────────────────────────────────────────────
export interface HealthResponse {
  status: 'ok' | 'error';
}

export interface ApiError {
  message: string;
  code?: string;
  detail?: string;
}

// ─── Navigation ───────────────────────────────────────────────
export interface NavLink {
  label: string;
  href: string;
  external?: boolean;
}

// ─── Feature Card ─────────────────────────────────────────────
export interface Feature {
  title: string;
  description: string;
  icon: string;
  gradient: string;
}
