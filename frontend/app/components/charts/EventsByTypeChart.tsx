'use client';

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const COLORS = ['#00d4ff', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#f97316', '#06b6d4'];

export function EventsByTypeChart({ data }: { data: { event_type: string; count: number }[] }) {
  if (data.length === 0) return <p style={{ fontSize: '13px', color: '#64748b', textAlign: 'center', padding: '32px 0' }}>No events data yet</p>;
  const colored = data.map((d, i) => ({ ...d, fill: COLORS[i % COLORS.length] }));
  return (
    <div style={{ width: '100%', height: '260px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={colored} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="event_type" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} dy={10} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} dx={-10} />
          <Tooltip 
            contentStyle={{ background: '#0a1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }}
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          />
          <Bar dataKey="count" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
