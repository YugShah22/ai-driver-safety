'use client';

import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';

interface AppTopbarProps {
  displayName: string;
  userEmail:   string;
}

export default function AppTopbar({ displayName, userEmail }: AppTopbarProps) {
  const router = useRouter();

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push('/login');
    router.refresh();
  }

  const initials = displayName
    .split(' ')
    .map((n) => n[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();

  return (
    <header
      style={{
        height: '64px',
        borderBottom: '1px solid rgba(0,212,255,0.08)',
        background: 'rgba(4,12,28,0.80)',
        backdropFilter: 'blur(16px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 32px',
        gap: '16px',
        flexShrink: 0,
        position: 'sticky',
        top: 0,
        zIndex: 10,
      }}
    >
      {/* User info */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        {/* Avatar */}
        <div
          style={{
            width: '34px',
            height: '34px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '13px',
            fontWeight: 700,
            color: '#fff',
            flexShrink: 0,
          }}
        >
          {initials || '?'}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0', lineHeight: 1.3 }}>
            {displayName}
          </span>
          <span style={{ fontSize: '11px', color: '#475569', lineHeight: 1.3 }}>
            {userEmail}
          </span>
        </div>
      </div>

      {/* Separator */}
      <div style={{ width: '1px', height: '28px', background: 'rgba(0,212,255,0.10)' }} />

      {/* Logout */}
      <button
        id="topbar-logout"
        onClick={handleLogout}
        style={{
          background: 'transparent',
          border: '1px solid rgba(239,68,68,0.20)',
          borderRadius: '8px',
          padding: '7px 14px',
          fontSize: '13px',
          fontWeight: 500,
          color: '#f87171',
          cursor: 'pointer',
          transition: 'all 0.18s',
          fontFamily: 'inherit',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.08)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(239,68,68,0.40)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = 'transparent';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(239,68,68,0.20)';
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
          <polyline points="16,17 21,12 16,7" />
          <line x1="21" y1="12" x2="9" y2="12" />
        </svg>
        Sign out
      </button>
    </header>
  );
}
