import type { Metadata } from 'next';
import { createClient } from '@/lib/supabase/server';
import LogoutButton from '@/app/components/LogoutButton';
import type { Profile } from '@/types';

export const metadata: Metadata = {
  title: 'Profile — AI Driver Safety Platform',
};

export default async function ProfilePage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  const { data: profile } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', user!.id)
    .single();

  const { count: tripCount } = await supabase
    .from('trips')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', user!.id);

  const typedProfile = profile as Profile | null;

  const initials = (typedProfile?.full_name || user?.email || '?')
    .split(' ')
    .map((n: string) => n[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();

  return (
    <div style={{ maxWidth: '600px' }}>
      <div style={{ marginBottom: '32px' }}>
        <h1
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '26px',
            fontWeight: 800,
            color: '#f0f9ff',
            letterSpacing: '-0.02em',
          }}
        >
          Profile
        </h1>
      </div>

      {/* Profile card */}
      <div
        style={{
          background: 'rgba(8,18,40,0.70)',
          border: '1px solid rgba(0,212,255,0.10)',
          borderRadius: '20px',
          padding: '32px',
          marginBottom: '16px',
        }}
      >
        {/* Avatar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '18px', marginBottom: '28px' }}>
          <div
            style={{
              width: '64px',
              height: '64px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '22px',
              fontWeight: 800,
              color: '#fff',
              flexShrink: 0,
            }}
          >
            {initials}
          </div>
          <div>
            <p style={{ fontSize: '18px', fontWeight: 700, color: '#f0f9ff' }}>
              {typedProfile?.full_name || '—'}
            </p>
            <p style={{ fontSize: '13px', color: '#64748b' }}>{user?.email}</p>
          </div>
        </div>

        {/* Details */}
        {[
          { label: 'Full name',    value: typedProfile?.full_name   || '—' },
          { label: 'Email',        value: user?.email           || '—' },
          { label: 'Member since', value: typedProfile?.created_at ? new Date(typedProfile.created_at).toLocaleDateString() : '—' },
          { label: 'Total trips',  value: String(tripCount ?? 0) },
        ].map((row) => (
          <div
            key={row.label}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '12px 0',
              borderBottom: '1px solid rgba(0,212,255,0.06)',
              fontSize: '14px',
            }}
          >
            <span style={{ color: '#64748b' }}>{row.label}</span>
            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{row.value}</span>
          </div>
        ))}
      </div>

      {/* Danger zone */}
      <div
        style={{
          background: 'rgba(8,18,40,0.50)',
          border: '1px solid rgba(239,68,68,0.12)',
          borderRadius: '16px',
          padding: '24px',
        }}
      >
        <p style={{ fontSize: '13px', fontWeight: 600, color: '#f87171', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '16px' }}>
          Session
        </p>
        <LogoutButton />
      </div>
    </div>
  );
}
