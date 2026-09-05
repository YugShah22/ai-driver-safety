import type { Metadata } from 'next';
import { createClient } from '@/lib/supabase/server';
import type { Trip } from '@/types';
import type { Profile } from '@/types';

export const metadata: Metadata = {
  title: 'Dashboard — AI Driver Safety Platform',
};

// ─── Stat card ────────────────────────────────────────────────
function StatCard({
  title,
  value,
  subtitle,
  accent,
}: {
  title: string;
  value: string;
  subtitle?: string;
  accent: string;
}) {
  return (
    <div
      style={{
        background: 'rgba(8,18,40,0.70)',
        border: `1px solid ${accent}22`,
        borderRadius: '16px',
        padding: '24px',
        backdropFilter: 'blur(16px)',
      }}
    >
      <p style={{ fontSize: '12px', fontWeight: 600, color: accent, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '10px' }}>
        {title}
      </p>
      <p style={{ fontSize: '32px', fontWeight: 800, fontFamily: 'Space Grotesk, Inter, sans-serif', color: '#f0f9ff', letterSpacing: '-0.02em', lineHeight: 1.1 }}>
        {value}
      </p>
      {subtitle && (
        <p style={{ fontSize: '13px', color: '#475569', marginTop: '6px' }}>{subtitle}</p>
      )}
    </div>
  );
}

export default async function DashboardPage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  // Fetch trip stats
  const { data: trips } = await supabase
    .from('trips')
    .select('id, status, created_at')
    .eq('user_id', user!.id)
    .order('created_at', { ascending: false });

  const tripList: Trip[] = (trips as Trip[]) ?? [];
  const total     = tripList.length;
  const completed = tripList.filter((t) => t.status === 'COMPLETED').length;
  const failed    = tripList.filter((t) => t.status === 'FAILED').length;
  const recent    = tripList.slice(0, 3);

  // Fetch AI analysis data for stats
  const tripIds = tripList.map((t) => t.id);

  let avgRiskScore = '—';
  let highRiskEventsCount = '—';
  let laneDeviationsCount = '—';
  let hardBrakingCount = '—';
  let pedestrianWarningsCount = '—';

  if (tripIds.length > 0) {
    // Risk score
    const { data: riskData } = await supabase.from('risk_predictions').select('risk_score').in('trip_id', tripIds);
    if (riskData && riskData.length > 0) {
      const totalScore = riskData.reduce((acc, curr) => acc + curr.risk_score, 0);
      avgRiskScore = (totalScore / riskData.length * 100).toFixed(1) + '%';
    }

    // Events
    const { data: eventsData } = await supabase.from('driving_events').select('event_type, severity').in('trip_id', tripIds);
    if (eventsData) {
      const highRisk = eventsData.filter(e => e.severity === 'HIGH' || e.severity === 'CRITICAL');
      highRiskEventsCount = String(highRisk.length);
      
      const laneDevs = eventsData.filter(e => e.event_type === 'LANE_DEPARTURE');
      laneDeviationsCount = String(laneDevs.length);

      const hardBraking = eventsData.filter(e => e.event_type === 'HARD_BRAKING');
      hardBrakingCount = String(hardBraking.length);

      const pedWarnings = eventsData.filter(e => e.event_type === 'PEDESTRIAN_PROXIMITY');
      pedestrianWarningsCount = String(pedWarnings.length);
    }
  }

  // Fetch profile for greeting
  const { data: profile } = await supabase
    .from('profiles')
    .select('full_name')
    .eq('id', user!.id)
    .single();

  const firstName = (profile as Profile | null)?.full_name?.split(' ')[0] ?? 'there';

  return (
    <div style={{ maxWidth: '1200px' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h1
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '28px',
            fontWeight: 800,
            color: '#f0f9ff',
            letterSpacing: '-0.02em',
            marginBottom: '6px',
          }}
        >
          Welcome back, {firstName} 👋
        </h1>
        <p style={{ fontSize: '14px', color: '#64748b' }}>
          Here&apos;s an overview of your driving analysis activity.
        </p>
      </div>

      {/* Stat cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: '16px',
          marginBottom: '40px',
        }}
      >
        <StatCard title="Total Trips"         value={String(total)}           subtitle="All time"                   accent="#00d4ff" />
        <StatCard title="Avg Risk Score"      value={avgRiskScore}            subtitle="Based on AI analysis"       accent="#8b5cf6" />
        <StatCard title="High-Risk Events"    value={highRiskEventsCount}     subtitle="Severity High/Critical"     accent="#ef4444" />
        <StatCard title="Lane Deviations"     value={laneDeviationsCount}     subtitle="Detected departures"        accent="#f59e0b" />
        <StatCard title="Hard Braking"        value={hardBrakingCount}        subtitle="Abrupt stops"               accent="#ef4444" />
        <StatCard title="Pedestrian Warnings" value={pedestrianWarningsCount} subtitle="Proximity alerts"           accent="#10b981" />
      </div>

      {/* Recent trips */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <h2 style={{ fontFamily: 'Space Grotesk, Inter, sans-serif', fontSize: '17px', fontWeight: 700, color: '#e2e8f0' }}>
            Recent Trips
          </h2>
          <a href="/trips" style={{ fontSize: '13px', color: '#00d4ff', textDecoration: 'none' }}>
            View all →
          </a>
        </div>

        {recent.length === 0 ? (
          <div
            style={{
              background: 'rgba(8,18,40,0.60)',
              border: '1px solid rgba(0,212,255,0.08)',
              borderRadius: '16px',
              padding: '48px',
              textAlign: 'center',
            }}
          >
            <p style={{ fontSize: '36px', marginBottom: '12px' }}>🚗</p>
            <p style={{ fontSize: '16px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
              No trips yet
            </p>
            <p style={{ fontSize: '14px', color: '#475569', marginBottom: '20px' }}>
              Upload your first dashcam video to get started.
            </p>
            <a
              href="/trips/upload"
              style={{
                display: 'inline-block',
                background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
                color: '#fff',
                textDecoration: 'none',
                padding: '10px 24px',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: 600,
              }}
            >
              Upload first trip
            </a>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {recent.map((trip) => (
              <a
                key={trip.id}
                href={`/trips/${trip.id}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: 'rgba(8,18,40,0.60)',
                  border: '1px solid rgba(0,212,255,0.08)',
                  borderRadius: '12px',
                  padding: '16px 20px',
                  textDecoration: 'none',
                  transition: 'border-color 0.18s',
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.22)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.08)'; }}
              >
                <div>
                  <p style={{ fontSize: '14px', fontWeight: 600, color: '#e2e8f0' }}>{trip.title}</p>
                  <p style={{ fontSize: '12px', color: '#475569', marginTop: '2px' }}>
                    {new Date(trip.created_at).toLocaleDateString()}
                  </p>
                </div>
                <StatusBadge status={trip.status} />
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    UPLOADED:   { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' },
    PROCESSING: { bg: 'rgba(245,158,11,0.12)',  text: '#fbbf24' },
    ANALYZING:  { bg: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
    COMPLETED:  { bg: 'rgba(16,185,129,0.12)',  text: '#34d399' },
    FAILED:     { bg: 'rgba(239,68,68,0.12)',   text: '#f87171' },
  };
  const { bg, text } = colors[status] ?? { bg: 'transparent', text: '#94a3b8' };
  return (
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
      }}
    >
      {status}
    </span>
  );
}
