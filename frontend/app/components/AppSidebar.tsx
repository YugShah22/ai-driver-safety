'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  {
    label: 'Dashboard',
    href: '/dashboard',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    label: 'Trips',
    href: '/trips',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M15 10l-4 4l6 6l4-16l-18 7l4 2l2 6l3-4" />
      </svg>
    ),
  },
  {
    label: 'Upload',
    href: '/trips/upload',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17,8 12,3 7,8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
  {
    label: 'Analytics',
    href: '/analytics',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6"  y1="20" x2="6"  y2="14" />
      </svg>
    ),
  },
  {
    label: 'Profile',
    href: '/profile',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
  },
];

export default function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside
      style={{
        width: '220px',
        flexShrink: 0,
        background: 'rgba(4,12,28,0.95)',
        borderRight: '1px solid rgba(0,212,255,0.08)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 0',
        position: 'sticky',
        top: 0,
        height: '100vh',
        overflowY: 'auto',
      }}
    >
      {/* Brand */}
      <Link
        href="/dashboard"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '0 20px',
          marginBottom: '32px',
          textDecoration: 'none',
        }}
      >
        <div
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '9px',
            background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L3 6.5V12c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V6.5L12 2z" fill="white" opacity="0.95" />
            <path d="M9 12l2 2 4-4" stroke="rgba(0,212,255,0.9)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <span
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontWeight: 700,
            fontSize: '15px',
            background: 'linear-gradient(90deg, #f0f9ff, #00d4ff)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          DriveAI
        </span>
      </Link>

      {/* Navigation */}
      <nav style={{ flex: 1, padding: '0 12px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {navItems.map((item) => {
          const active = pathname === item.href || (item.href !== '/dashboard' && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              id={`sidebar-${item.label.toLowerCase()}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '10px 12px',
                borderRadius: '10px',
                textDecoration: 'none',
                fontSize: '14px',
                fontWeight: active ? 600 : 400,
                color: active ? '#f0f9ff' : '#64748b',
                background: active ? 'rgba(0,212,255,0.10)' : 'transparent',
                border: active ? '1px solid rgba(0,212,255,0.18)' : '1px solid transparent',
                transition: 'all 0.18s',
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(0,212,255,0.05)';
                  (e.currentTarget as HTMLElement).style.color = '#94a3b8';
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                  (e.currentTarget as HTMLElement).style.color = '#64748b';
                }
              }}
            >
              <span style={{ color: active ? '#00d4ff' : 'currentColor', display: 'flex' }}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom badge */}
      <div style={{ padding: '16px 20px 0' }}>
        <div
          style={{
            background: 'rgba(0,212,255,0.06)',
            border: '1px solid rgba(0,212,255,0.10)',
            borderRadius: '10px',
            padding: '10px 12px',
          }}
        >
          <p style={{ fontSize: '11px', color: '#00d4ff', fontWeight: 600, marginBottom: '2px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Phase 2
          </p>
          <p style={{ fontSize: '11px', color: '#475569', lineHeight: 1.4 }}>
            Supabase Auth & DB active
          </p>
        </div>
      </div>
    </aside>
  );
}
