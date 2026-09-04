import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Analytics — AI Driver Safety Platform',
};

export default function AnalyticsPage() {
  return (
    <div style={{ maxWidth: '1000px' }}>
      <div style={{ marginBottom: '32px' }}>
        <h1
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '26px',
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

      {/* Placeholder charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
        {[
          { title: 'Risk Score Over Time',    icon: '📈', note: 'Line chart — Phase 3 (Recharts)' },
          { title: 'Events by Type',          icon: '🗂️', note: 'Bar chart — Phase 3' },
          { title: 'Events by Severity',     icon: '⚠️', note: 'Pie chart — Phase 3' },
          { title: 'Trip Comparison',         icon: '🔀', note: 'Multi-trip comparison — Phase 3' },
        ].map((chart) => (
          <div
            key={chart.title}
            style={{
              background: 'rgba(8,18,40,0.60)',
              border: '1px solid rgba(0,212,255,0.08)',
              borderRadius: '16px',
              padding: '32px',
              minHeight: '200px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              textAlign: 'center',
              gap: '10px',
            }}
          >
            <p style={{ fontSize: '32px' }}>{chart.icon}</p>
            <p style={{ fontSize: '15px', fontWeight: 700, color: '#94a3b8' }}>{chart.title}</p>
            <p style={{ fontSize: '12px', color: '#475569' }}>{chart.note}</p>
          </div>
        ))}
      </div>

      <div
        style={{
          background: 'rgba(99,102,241,0.06)',
          border: '1px solid rgba(99,102,241,0.14)',
          borderRadius: '12px',
          padding: '16px 20px',
          fontSize: '13px',
          color: '#a5b4fc',
        }}
      >
        Analytics charts will be built in Phase 3 (Next.js UI) once trips with completed analysis exist.
      </div>
    </div>
  );
}
