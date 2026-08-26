/**
 * FastAPI client — all calls to the backend go through this module.
 * The base URL is read from the NEXT_PUBLIC_FASTAPI_URL env var.
 */
import type { HealthResponse, ApiError } from '@/types';

const BASE_URL =
  process.env.NEXT_PUBLIC_FASTAPI_URL ?? 'http://localhost:8000';

async function apiFetch<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${path}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    let errorBody: ApiError = { message: 'Unknown error' };
    try {
      errorBody = await response.json();
    } catch {
      errorBody = { message: response.statusText };
    }
    throw new Error(errorBody.message ?? errorBody.detail ?? 'Request failed');
  }

  return response.json() as Promise<T>;
}

// ─── Health ──────────────────────────────────────────────────
export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

// Future endpoints (Phase 2+):
// export async function createSession(file: File): Promise<AnalysisSession> { ... }
// export async function getSession(id: string): Promise<AnalysisSession> { ... }
// export async function listSessions(): Promise<AnalysisSession[]> { ... }
