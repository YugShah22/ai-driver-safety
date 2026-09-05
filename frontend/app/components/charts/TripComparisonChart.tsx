'use client';

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export function TripComparisonChart({ data }: { data: { label: string; risk_score: number }[] }) {
  if (data.length === 0) return <p style={{ fontSize: '13px', color: '#64748b', textAlign: 'center', padding: '32px 0' }}>No trip comparison data yet</p>;
  return (
    <div style={{ width: '100%', height: '260px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} dy={10} />
          <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} dx={-10} />
          <Tooltip 
            contentStyle={{ background: '#0a1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }}
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          />
          <Bar dataKey="risk_score" fill="#6366f1" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
