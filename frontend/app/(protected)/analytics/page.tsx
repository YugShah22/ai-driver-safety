import type { Metadata } from 'next';
import { createClient } from '@/lib/supabase/server';
import { RiskOverTimeChart } from '@/app/components/charts/RiskOverTimeChart';
import { EventsByTypeChart } from '@/app/components/charts/EventsByTypeChart';
import { EventsBySeverityChart } from '@/app/components/charts/EventsBySeverityChart';
import { TripComparisonChart } from '@/app/components/charts/TripComparisonChart';
import { DrivingMetricsChart } from '@/app/components/charts/DrivingMetricsChart';
import type { Trip } from '@/types';

export const metadata: Metadata = {
  title: 'Analytics — AI Driver Safety Platform',
};

export default async function AnalyticsPage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  // Fetch trips
  const { data: tripsRaw } = await supabase
    .from('trips')
    .select('id, title, status, created_at')
    .eq('user_id', user!.id)
    .eq('status', 'COMPLETED')
    .order('created_at');

  const trips = (tripsRaw ?? []) as Trip[];
  const tripIds = trips.map((t) => t.id);

  // 1. Risk over time: avg risk per trip
  const { data: riskRaw } = tripIds.length > 0
    ? await supabase.from('risk_predictions').select('risk_score, trip_id').in('trip_id', tripIds)
    : { data: [] };

  const riskByTrip: Record<string, number[]> = {};
  ((riskRaw ?? []) as { risk_score: number; trip_id: string }[]).forEach((r) => {
    (riskByTrip[r.trip_id] ??= []).push(r.risk_score);
  });

  const riskOverTime = trips
    .filter((t) => riskByTrip[t.id]?.length > 0)
    .map((t) => ({
      label: new Date(t.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
      risk_score: Math.round((riskByTrip[t.id].reduce((a, b) => a + b, 0) / riskByTrip[t.id].length) * 100),
    }));

  // 2 & 3. Events by type & severity
  const { data: eventsRaw } = tripIds.length > 0
    ? await supabase.from('driving_events').select('event_type, severity').in('trip_id', tripIds)
    : { data: [] };

  const events = (eventsRaw ?? []) as { event_type: string; severity: string }[];

  const typeCounts: Record<string, number> = {};
  const sevCounts: Record<string, number> = {};
  events.forEach((e) => {
    typeCounts[e.event_type] = (typeCounts[e.event_type] ?? 0) + 1;
    sevCounts[e.severity]    = (sevCounts[e.severity] ?? 0) + 1;
  });

  const eventsByType     = Object.entries(typeCounts).map(([event_type, count]) => ({ event_type: event_type.replace(/_/g, ' '), count }));
  const eventsBySeverity = Object.entries(sevCounts).map(([severity, count]) => ({ severity, count }));

  // 4. Trip comparison (last 10 trips)
  const tripComparison = trips
    .filter((t) => riskByTrip[t.id]?.length > 0)
    .slice(-10)
    .map((t) => ({
      label: t.title.length > 12 ? t.title.slice(0, 12) + '…' : t.title,
      risk_score: Math.round((riskByTrip[t.id].reduce((a, b) => a + b, 0) / riskByTrip[t.id].length) * 100),
    }));

  // 5. Driving metrics (aggregate average over time)
  // To keep it simple, we'll pull metrics from the most recent completed trip, or aggregate a few.
  // For demonstration, let's take the latest completed trip's metrics.
  let drivingMetrics: { time: string; speed: number; acceleration: number }[] = [];
  if (tripIds.length > 0) {
    const latestTripId = tripIds[tripIds.length - 1];
    const { data: metricsRaw } = await supabase
      .from('driving_metrics')
      .select('timestamp, speed, acceleration')
      .eq('trip_id', latestTripId)
      .order('timestamp')
      .limit(50); // Downsample for the chart
      
    if (metricsRaw) {
      drivingMetrics = metricsRaw
        .filter(m => m.speed != null && m.acceleration != null)
        .map(m => ({
          time: new Date(m.timestamp * 1000).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }),
          speed: Math.round(m.speed!),
          acceleration: Number(m.acceleration!.toFixed(2))
        }));
    }
  }

  // Summary stats
  const totalEvents     = events.length;
  const totalCompleted  = trips.length;
  const overallAvgRisk  = riskOverTime.length > 0 ? Math.round(riskOverTime.reduce((a, r) => a + r.risk_score, 0) / riskOverTime.length) : null;

  const hasData = trips.length > 0;

  const panelStyle = {
    background: 'rgba(8,18,40,0.60)',
    border: '1px solid rgba(0,212,255,0.08)',
    borderRadius: '16px',
    padding: '24px',
  };

  const titleStyle = {
    fontSize: '14px',
    fontWeight: 600,
    color: '#e2e8f0',
    marginBottom: '20px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  };

  return (
    <div style={{ maxWidth: '1200px', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <div>
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
          Analytics
        </h1>
        <p style={{ fontSize: '14px', color: '#64748b' }}>
          Risk trends, event frequency, and driving behavior over time.
        </p>
      </div>

      {/* Summary row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
        <div style={{ ...panelStyle, textAlign: 'center' as const, padding: '32px 20px' }}>
          <p style={{ fontSize: '36px', fontWeight: 800, color: '#f0f9ff', fontFamily: 'Space Grotesk, Inter, sans-serif', lineHeight: 1 }}>{totalCompleted}</p>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '8px' }}>Completed Trips</p>
        </div>
        <div style={{ ...panelStyle, textAlign: 'center' as const, padding: '32px 20px' }}>
          <p style={{ fontSize: '36px', fontWeight: 800, fontFamily: 'Space Grotesk, Inter, sans-serif', lineHeight: 1, color: overallAvgRisk !== null ? (overallAvgRisk < 30 ? '#10b981' : overallAvgRisk < 60 ? '#f59e0b' : '#ef4444') : '#475569' }}>
            {overallAvgRisk !== null ? `${overallAvgRisk}%` : '—'}
          </p>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '8px' }}>Avg Risk Score</p>
        </div>
        <div style={{ ...panelStyle, textAlign: 'center' as const, padding: '32px 20px' }}>
          <p style={{ fontSize: '36px', fontWeight: 800, color: '#f0f9ff', fontFamily: 'Space Grotesk, Inter, sans-serif', lineHeight: 1 }}>{totalEvents}</p>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '8px' }}>Safety Events</p>
        </div>
      </div>

      {/* Charts Grid - Responsive */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
        
        {/* Risk Over Time */}
        <div style={{ ...panelStyle, gridColumn: '1 / -1' }}>
          <h2 style={titleStyle}>📈 Risk Score Over Time</h2>
          {hasData ? <RiskOverTimeChart data={riskOverTime} /> : <ChartEmpty />}
        </div>

        {/* Events By Type */}
        <div style={panelStyle}>
          <h2 style={titleStyle}>⚠️ Events by Type</h2>
          {hasData ? <EventsByTypeChart data={eventsByType} /> : <ChartEmpty />}
        </div>

        {/* Events By Severity */}
        <div style={panelStyle}>
          <h2 style={titleStyle}>🔴 Events by Severity</h2>
          {hasData ? <EventsBySeverityChart data={eventsBySeverity} /> : <ChartEmpty />}
        </div>

        {/* Driving Metrics (Latest Trip) */}
        <div style={{ ...panelStyle, gridColumn: '1 / -1' }}>
          <h2 style={titleStyle}>⏱️ Driving Metrics (Latest Trip Sample)</h2>
          {hasData ? <DrivingMetricsChart data={drivingMetrics} /> : <ChartEmpty />}
        </div>

        {/* Trip Comparison */}
        <div style={{ ...panelStyle, gridColumn: '1 / -1' }}>
          <h2 style={titleStyle}>📊 Trip Comparison</h2>
          {hasData ? <TripComparisonChart data={tripComparison} /> : <ChartEmpty />}
        </div>

      </div>

      {!hasData && (
        <div style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: '14px', padding: '16px', fontSize: '14px', color: '#a5b4fc', textAlign: 'center' }}>
          Analytics charts will populate once you have trips with completed AI analysis.
        </div>
      )}
    </div>
  );
}

function ChartEmpty() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 0', textAlign: 'center' }}>
      <p style={{ fontSize: '32px', marginBottom: '12px' }}>📊</p>
      <p style={{ fontSize: '13px', color: '#64748b' }}>Data appears after analysis</p>
    </div>
  );
}
