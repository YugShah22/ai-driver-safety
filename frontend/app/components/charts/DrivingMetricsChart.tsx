'use client';

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export function DrivingMetricsChart({ data }: { data: { time: string; speed: number; acceleration: number }[] }) {
  if (data.length === 0) return <p style={{ fontSize: '13px', color: '#64748b', textAlign: 'center', padding: '32px 0' }}>No metrics data yet</p>;
  return (
    <div style={{ width: '100%', height: '260px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} dy={10} />
          <YAxis yAxisId="left" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} dx={-10} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} dx={10} />
          <Tooltip 
            contentStyle={{ background: '#0a1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }}
          />
          <Legend formatter={(v: string) => <span style={{ color: '#94a3b8', fontSize: '11px', fontWeight: 500 }}>{v}</span>} />
          <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed (km/h)" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
          <Line yAxisId="right" type="monotone" dataKey="acceleration" name="Acceleration (m/s²)" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
