import type { Metadata } from 'next';
import { createClient } from '@/lib/supabase/server';
import { notFound } from 'next/navigation';
import type { Trip } from '@/types';

export const metadata: Metadata = {
  title: 'Trip Details — AI Driver Safety Platform',
};

export default async function TripDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  const { data, error } = await supabase
    .from('trips')
    .select('*')
    .eq('id', id)
    .eq('user_id', user!.id)
    .single();

  if (error || !data) notFound();

  const trip = data as Trip;

  const statusColors: Record<string, { bg: string; text: string }> = {
    UPLOADED:   { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' },
    PROCESSING: { bg: 'rgba(245,158,11,0.12)',  text: '#fbbf24' },
    ANALYZING:  { bg: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
    COMPLETED:  { bg: 'rgba(16,185,129,0.12)',  text: '#34d399' },
    FAILED:     { bg: 'rgba(239,68,68,0.12)',   text: '#f87171' },
  };
  const { bg, text } = statusColors[trip.status] ?? { bg: 'transparent', text: '#94a3b8' };

  return (
    <div style={{ maxWidth: '900px' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <h1
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: '26px',
              fontWeight: 800,
              color: '#f0f9ff',
              letterSpacing: '-0.02em',
            }}
          >
            {trip.title}
          </h1>
          <span style={{ background: bg, color: text, fontSize: '11px', fontWeight: 600, padding: '4px 10px', borderRadius: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {trip.status}
          </span>
        </div>
        <p style={{ fontSize: '13px', color: '#475569' }}>
          Created {new Date(trip.created_at).toLocaleString()}
          {trip.completed_at ? ` · Completed ${new Date(trip.completed_at).toLocaleString()}` : ''}
          {trip.duration ? ` · Duration: ${Math.round(trip.duration)}s` : ''}
        </p>
      </div>

      {/* Placeholder panels — filled in Phase 3+ */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        {[
          { title: 'Video',          icon: '🎬', note: 'Video player — Phase 3' },
          { title: 'Object Detection', icon: '🎯', note: 'YOLO detections — Phase 7' },
          { title: 'Risk Score',     icon: '📊', note: 'ANN/ML risk output — Phase 12' },
          { title: 'Driving Events', icon: '⚠️', note: 'Event timeline — Phase 13' },
          { title: 'Metrics',        icon: '📈', note: 'Driving metrics — Phase 10' },
          { title: 'Explainability', icon: '💡', note: 'Grad-CAM & SHAP — Phase 15' },
        ].map((panel) => (
          <div
            key={panel.title}
            style={{
              background: 'rgba(8,18,40,0.60)',
              border: '1px solid rgba(0,212,255,0.08)',
              borderRadius: '14px',
              padding: '24px',
            }}
          >
            <p style={{ fontSize: '20px', marginBottom: '10px' }}>{panel.icon}</p>
            <p style={{ fontSize: '14px', fontWeight: 700, color: '#e2e8f0', marginBottom: '4px' }}>{panel.title}</p>
            <p style={{ fontSize: '12px', color: '#475569' }}>{panel.note}</p>
          </div>
        ))}
      </div>

      {trip.status === 'UPLOADED' && (
        <div
          style={{
            marginTop: '24px',
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.18)',
            borderRadius: '12px',
            padding: '16px 20px',
            fontSize: '14px',
            color: '#fbbf24',
          }}
        >
          📋 Video uploaded. AI analysis pipeline will be triggered in Phase 4.
        </div>
      )}
    </div>
  );
}
