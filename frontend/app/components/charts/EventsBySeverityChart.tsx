'use client';

import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

const SEV_COLORS: Record<string, string> = { LOW: '#10b981', MEDIUM: '#f59e0b', HIGH: '#f97316', CRITICAL: '#ef4444' };

export function EventsBySeverityChart({ data }: { data: { severity: string; count: number }[] }) {
  if (data.length === 0) return <p style={{ fontSize: '13px', color: '#64748b', textAlign: 'center', padding: '32px 0' }}>No severity data yet</p>;
  return (
    <div style={{ width: '100%', height: '260px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie 
            data={data} 
            dataKey="count" 
            nameKey="severity" 
            cx="50%" 
            cy="45%" 
            innerRadius={60} 
            outerRadius={90} 
            strokeWidth={0}
            paddingAngle={2}
          >
            {data.map((d) => (
              <Cell key={d.severity} fill={SEV_COLORS[d.severity] ?? '#64748b'} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ background: '#0a1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '10px', color: '#e2e8f0', fontSize: '12px' }} 
          />
          <Legend formatter={(v: string) => <span style={{ color: '#94a3b8', fontSize: '11px', fontWeight: 500 }}>{v}</span>} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
