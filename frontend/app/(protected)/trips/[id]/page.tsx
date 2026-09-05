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

  // ─── Fetch AI Analysis Data ────────────────────────────
  const { data: riskData } = await supabase.from('risk_predictions').select('*').eq('trip_id', id).order('timestamp');
  const { data: eventsData } = await supabase.from('driving_events').select('*').eq('trip_id', id).order('timestamp');
  const { data: detectionsData } = await supabase.from('detections').select('object_type, confidence').eq('trip_id', id);
  const { data: metricsData } = await supabase.from('driving_metrics').select('*').eq('trip_id', id).order('timestamp');

  const risks      = (riskData ?? []) as { risk_score: number; risk_class: string; timestamp: number }[];
  const events     = (eventsData ?? []) as { event_type: string; severity: string; timestamp: number; description: string | null }[];
  const detections = (detectionsData ?? []) as { object_type: string; confidence: number }[];
  const metrics    = (metricsData ?? []) as { timestamp: number; speed: number | null; acceleration: number | null; lane_deviation: number | null }[];

  const avgRisk = risks.length > 0 ? (risks.reduce((a, r) => a + r.risk_score, 0) / risks.length * 100).toFixed(0) : null;
  const riskClass = risks.length > 0 ? risks[risks.length - 1].risk_class : null;

  // Group detections by type
  const detectionCounts: Record<string, number> = {};
  detections.forEach((d) => { detectionCounts[d.object_type] = (detectionCounts[d.object_type] ?? 0) + 1; });

  const hasAnalysis = risks.length > 0 || events.length > 0 || metrics.length > 0 || detections.length > 0;

  const statusColors: Record<string, { bg: string; text: string }> = {
    UPLOADED:   { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' },
    PROCESSING: { bg: 'rgba(245,158,11,0.12)',  text: '#fbbf24' },
    ANALYZING:  { bg: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
    COMPLETED:  { bg: 'rgba(16,185,129,0.12)',  text: '#34d399' },
    FAILED:     { bg: 'rgba(239,68,68,0.12)',   text: '#f87171' },
  };
  const { bg, text } = statusColors[trip.status] ?? { bg: 'transparent', text: '#94a3b8' };

  return (
    <div style={{ maxWidth: '1000px', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <h1
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: '28px',
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
        <p style={{ fontSize: '14px', color: '#64748b' }}>
          Created {new Date(trip.created_at).toLocaleString()}
          {trip.completed_at ? ` · Completed ${new Date(trip.completed_at).toLocaleString()}` : ''}
          {trip.duration ? ` · Duration: ${Math.round(trip.duration)}s` : ''}
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
        
        {/* AI Processing Status */}
        <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px' }}>
          <h2 style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            ⚙️ AI Processing Status
          </h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            {['UPLOADED', 'PROCESSING', 'ANALYZING', 'COMPLETED'].map((step, i) => {
                const order = ['UPLOADED', 'PROCESSING', 'ANALYZING', 'COMPLETED'];
                const currentIdx = order.indexOf(trip.status);
                const done = i <= currentIdx && trip.status !== 'FAILED';
                const active = i === currentIdx && trip.status !== 'FAILED';
                return (
                    <div key={step} style={{
                        flex: 1,
                        padding: '12px',
                        borderRadius: '10px',
                        textAlign: 'center',
                        fontSize: '13px',
                        fontWeight: 600,
                        background: done ? 'rgba(0,212,255,0.08)' : 'rgba(255,255,255,0.02)',
                        color: done ? '#00d4ff' : '#64748b',
                        border: active ? '1px solid rgba(0,212,255,0.3)' : '1px solid transparent',
                        transition: 'all 0.2s'
                    }}>
                        {done ? '✓' : (i + 1)} {step}
                    </div>
                );
            })}
          </div>
          {trip.status === 'FAILED' && <p style={{ color: '#ef4444', fontSize: '14px', marginTop: '16px' }}>Analysis failed. Please check backend logs.</p>}
        </div>

        {/* Video Player */}
        <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', overflow: 'hidden' }}>
          <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(0,212,255,0.08)' }}>
            <h2 style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '8px' }}>
              🎬 Dashcam Video
            </h2>
          </div>
          <div style={{ background: '#020610', aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {trip.video_path ? (
               <div style={{ textAlign: 'center' }}>
                 <p style={{ fontSize: '32px', marginBottom: '12px' }}>▶️</p>
                 <p style={{ color: '#475569', fontSize: '14px' }}>Video player will stream from Supabase Storage</p>
                 <p style={{ color: '#334155', fontSize: '12px', marginTop: '4px' }}>Path: {trip.video_path}</p>
               </div>
            ) : (
               <p style={{ color: '#475569', fontSize: '14px' }}>No video uploaded</p>
            )}
          </div>
        </div>

        {/* AI Results Section */}
        {hasAnalysis ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
            
            {/* Risk Assessment */}
            <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px' }}>
              <h2 style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '16px' }}>Risk Assessment</h2>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
                <p style={{ fontSize: '48px', fontWeight: 800, fontFamily: 'Space Grotesk, Inter, sans-serif', color: Number(avgRisk) < 30 ? '#10b981' : Number(avgRisk) < 60 ? '#f59e0b' : '#ef4444', lineHeight: 1 }}>{avgRisk ?? '—'}</p>
                {riskClass && <span style={{ fontSize: '16px', fontWeight: 600, color: '#e2e8f0' }}>{riskClass}</span>}
              </div>
              <p style={{ fontSize: '13px', color: '#64748b', marginTop: '12px' }}>Average AI Risk Score</p>
            </div>

            {/* Event Timeline */}
            <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px' }}>
              <h2 style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '16px' }}>Event Timeline</h2>
              {events.length === 0 ? <p style={{ fontSize: '14px', color: '#64748b' }}>No events detected</p> : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', maxHeight: '200px', overflowY: 'auto', paddingRight: '8px' }}>
                  {events.map((evt, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '14px', color: '#e2e8f0', fontWeight: 500 }}>{evt.event_type.replace(/_/g, ' ')}</span>
                      <span style={{ fontSize: '11px', fontWeight: 700, padding: '3px 8px', borderRadius: '6px', background: evt.severity === 'CRITICAL' ? 'rgba(239,68,68,0.15)' : evt.severity === 'HIGH' ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)', color: evt.severity === 'CRITICAL' ? '#f87171' : evt.severity === 'HIGH' ? '#fbbf24' : '#34d399' }}>{evt.severity}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Driving Metrics */}
            <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px' }}>
              <h2 style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '20px' }}>Driving Metrics</h2>
              {metrics.length === 0 ? <p style={{ fontSize: '14px', color: '#64748b' }}>No metrics available</p> : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div>
                    <p style={{ fontSize: '28px', fontWeight: 700, color: '#f0f9ff', fontFamily: 'Space Grotesk, Inter, sans-serif' }}>{metrics.filter(m => m.speed != null).length > 0 ? (metrics.reduce((a, m) => a + (m.speed ?? 0), 0) / metrics.filter(m => m.speed != null).length).toFixed(1) : '—'}</p>
                    <p style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Avg Speed (km/h)</p>
                  </div>
                  <div>
                    <p style={{ fontSize: '28px', fontWeight: 700, color: '#f0f9ff', fontFamily: 'Space Grotesk, Inter, sans-serif' }}>{metrics.length}</p>
                    <p style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Data Points</p>
                  </div>
                </div>
              )}
            </div>

            {/* Detected Objects */}
            <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px' }}>
              <h2 style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '16px' }}>Detected Objects</h2>
              {Object.keys(detectionCounts).length === 0 ? <p style={{ fontSize: '14px', color: '#64748b' }}>No objects detected</p> : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {Object.entries(detectionCounts).map(([type, count]) => (
                    <div key={type} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                      <span style={{ color: '#e2e8f0', textTransform: 'capitalize', fontWeight: 500 }}>{type}</span>
                      <span style={{ color: '#00d4ff', fontWeight: 700 }}>{count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Analysis Summary */}
            <div style={{ background: 'rgba(8,18,40,0.60)', border: '1px solid rgba(0,212,255,0.08)', borderRadius: '16px', padding: '24px', gridColumn: '1 / -1' }}>
              <h2 style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '16px' }}>Analysis Summary</h2>
              <p style={{ fontSize: '15px', color: '#cbd5e1', lineHeight: 1.6 }}>
                This trip resulted in a risk score of <span style={{ color: '#fff', fontWeight: 600 }}>{avgRisk ?? '—'}</span> {riskClass ? `(${riskClass})` : ''}. 
                There were <span style={{ color: '#fff', fontWeight: 600 }}>{events.length}</span> safety events detected during the recording.
                {events.length === 0 ? " Overall, the driving was safe." : " Please review the event timeline to see specific incidents."}
              </p>
            </div>
            
          </div>
        ) : (
          <div style={{ background: 'rgba(8,18,40,0.40)', border: '1px dashed rgba(0,212,255,0.15)', borderRadius: '16px', padding: '64px', textAlign: 'center' }}>
            <p style={{ fontSize: '32px', marginBottom: '16px' }}>⏳</p>
            <p style={{ fontSize: '18px', fontWeight: 700, color: '#e2e8f0', marginBottom: '8px' }}>Waiting for AI Analysis</p>
            <p style={{ fontSize: '15px', color: '#64748b', maxWidth: '400px', margin: '0 auto' }}>Risk score, events, detections, and metrics will appear here once the backend finishes processing the video.</p>
          </div>
        )}
      </div>
    </div>
  );
}
