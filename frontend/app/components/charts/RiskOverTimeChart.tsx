'use client';

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export function RiskOverTimeChart({ data }: { data: { label: string; risk_score: number }[] }) {
  if (data.length === 0) return <p style={{ fontSize: '13px', color: '#64748b', textAlign: 'center', padding: '32px 0' }}>No data yet</p>;
  return (
    <div style={{ width: '100%', height: '260px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#00d4ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} dy={10} />
          <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} dx={-10} />
          <Tooltip 
            contentStyle={{ background: '#0a1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }} 
            itemStyle={{ color: '#00d4ff', fontWeight: 600 }}
          />
          <Area type="monotone" dataKey="risk_score" stroke="#00d4ff" strokeWidth={3} fill="url(#riskGrad)" activeDot={{ r: 6, fill: '#00d4ff', stroke: '#fff' }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
