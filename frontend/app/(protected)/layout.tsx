/**
 * Protected route group layout.
 * Server-side auth check — unauthenticated users are redirected to /login.
 * Renders the full app shell: sidebar + topbar + content area.
 */
import { redirect } from 'next/navigation';
import { createClient } from '@/lib/supabase/server';
import AppSidebar from '@/app/components/AppSidebar';
import AppTopbar from '@/app/components/AppTopbar';

export default async function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // ── Guard: show setup notice if Supabase isn't configured yet ────────
  let supabase;
  try {
    supabase = await createClient();
  } catch {
    return (
      <div
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #020810, #050d1f)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'Inter, sans-serif',
          padding: '24px',
        }}
      >
        <div
          style={{
            background: 'rgba(8,18,40,0.85)',
            border: '1px solid rgba(245,158,11,0.25)',
            borderRadius: '20px',
            padding: '40px',
            maxWidth: '520px',
            width: '100%',
          }}
        >
          <p style={{ fontSize: '28px', marginBottom: '16px' }}>⚙️</p>
          <h1
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: '20px',
              fontWeight: 800,
              color: '#fbbf24',
              marginBottom: '12px',
            }}
          >
            Supabase not configured
          </h1>
          <p style={{ fontSize: '14px', color: '#94a3b8', lineHeight: 1.7, marginBottom: '20px' }}>
            To use the protected app, create{' '}
            <code
              style={{
                background: 'rgba(0,212,255,0.08)',
                padding: '2px 6px',
                borderRadius: '4px',
                color: '#00d4ff',
                fontSize: '13px',
              }}
            >
              frontend/.env.local
            </code>{' '}
            with your Supabase credentials:
          </p>
          <pre
            style={{
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid rgba(0,212,255,0.12)',
              borderRadius: '10px',
              padding: '16px',
              fontSize: '13px',
              color: '#e2e8f0',
              lineHeight: 1.7,
              overflowX: 'auto',
            }}
          >
{`NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_FASTAPI_URL=http://localhost:8000`}
          </pre>
          <p style={{ fontSize: '13px', color: '#475569', marginTop: '16px' }}>
            Get these from your{' '}
            <a
              href="https://supabase.com/dashboard"
              target="_blank"
              rel="noreferrer"
              style={{ color: '#00d4ff' }}
            >
              Supabase dashboard
            </a>{' '}
            → Settings → API
          </p>
        </div>
      </div>
    );
  }

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/login');
  }

  // Fetch profile for display name
  const { data: profile } = await supabase
    .from('profiles')
    .select('full_name, email')
    .eq('id', user.id)
    .single();

  const profileData = profile as { full_name: string | null; email: string | null } | null;
  const displayName = profileData?.full_name || profileData?.email || user.email || 'User';

  return (
    <div
      style={{
        display: 'flex',
        minHeight: '100vh',
        background: '#020810',
        color: '#f0f9ff',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      <AppSidebar />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <AppTopbar displayName={displayName} userEmail={user.email ?? ''} />
        <main
          style={{
            flex: 1,
            padding: '32px',
            overflowY: 'auto',
          }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
