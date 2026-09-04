import type { Metadata } from 'next';
import Link from 'next/link';
import { createClient } from '@/lib/supabase/server';
import type { Trip } from '@/types';

export const metadata: Metadata = {
  title: 'Trips — AI Driver Safety Platform',
};

const statusColors: Record<string, { bg: string; text: string }> = {
  UPLOADED:   { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' },
  PROCESSING: { bg: 'rgba(245,158,11,0.12)',  text: '#fbbf24' },
  ANALYZING:  { bg: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
  COMPLETED:  { bg: 'rgba(16,185,129,0.12)',  text: '#34d399' },
  FAILED:     { bg: 'rgba(239,68,68,0.12)',   text: '#f87171' },
};

export default async function TripsPage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  const { data, error } = await supabase
    .from('trips')
    .select('*')
    .eq('user_id', user!.id)
    .order('created_at', { ascending: false });

  const trips: Trip[] = (data as Trip[]) ?? [];

  return (
    <div style={{ maxWidth: '1000px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
        <div>
          <h1
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: '26px',
              fontWeight: 800,
              color: '#f0f9ff',
              letterSpacing: '-0.02em',
              marginBottom: '4px',
            }}
          >
            My Trips
          </h1>
          <p style={{ fontSize: '14px', color: '#64748b' }}>
            {trips.length} trip{trips.length !== 1 ? 's' : ''} total
          </p>
        </div>
        <Link
          href="/trips/upload"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
            color: '#fff',
            textDecoration: 'none',
            padding: '10px 20px',
            borderRadius: '10px',
            fontSize: '14px',
            fontWeight: 600,
            boxShadow: '0 0 20px rgba(0,212,255,0.20)',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17,8 12,3 7,8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          Upload trip
        </Link>
      </div>

      {error && (
        <div style={{ background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.20)', borderRadius: '12px', padding: '16px', marginBottom: '24px', fontSize: '14px', color: '#fca5a5' }}>
          Error loading trips: {error.message}
        </div>
      )}

      {trips.length === 0 ? (
        <div
          style={{
            background: 'rgba(8,18,40,0.60)',
            border: '1px solid rgba(0,212,255,0.08)',
            borderRadius: '20px',
            padding: '64px',
            textAlign: 'center',
          }}
        >
          <p style={{ fontSize: '48px', marginBottom: '16px' }}>📹</p>
          <p style={{ fontSize: '18px', fontWeight: 700, color: '#94a3b8', marginBottom: '8px' }}>
            No trips yet
          </p>
          <p style={{ fontSize: '14px', color: '#475569', marginBottom: '24px' }}>
            Upload your first dashcam video to start analyzing your driving.
          </p>
          <Link
            href="/trips/upload"
            style={{
              display: 'inline-block',
              background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
              color: '#fff',
              textDecoration: 'none',
              padding: '12px 28px',
              borderRadius: '10px',
              fontSize: '14px',
              fontWeight: 600,
            }}
          >
            Upload first trip
          </Link>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {trips.map((trip) => {
            const { bg, text } = statusColors[trip.status] ?? { bg: 'transparent', text: '#94a3b8' };
            return (
              <Link
                key={trip.id}
                href={`/trips/${trip.id}`}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr auto auto',
                  alignItems: 'center',
                  gap: '16px',
                  background: 'rgba(8,18,40,0.65)',
                  border: '1px solid rgba(0,212,255,0.08)',
                  borderRadius: '14px',
                  padding: '18px 22px',
                  textDecoration: 'none',
                  transition: 'all 0.18s',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.22)';
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.08)';
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
                }}
              >
                <div>
                  <p style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0', marginBottom: '4px' }}>
                    {trip.title}
                  </p>
                  <p style={{ fontSize: '12px', color: '#475569' }}>
                    {new Date(trip.created_at).toLocaleString()}
                    {trip.duration ? ` · ${Math.round(trip.duration)}s` : ''}
                  </p>
                </div>
                <span
                  style={{
                    background: bg,
                    color: text,
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '4px 10px',
                    borderRadius: '6px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {trip.status}
                </span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
