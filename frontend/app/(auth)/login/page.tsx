'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const supabase = createClient();
    const { error: authError } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    router.push('/dashboard');
    router.refresh();
  }

  return (
    <div
      style={{
        background: 'rgba(8,18,40,0.80)',
        border: '1px solid rgba(0,212,255,0.15)',
        borderRadius: '20px',
        padding: '40px',
        backdropFilter: 'blur(24px)',
        boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
      }}
    >
      {/* Logo */}
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '14px',
            background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            boxShadow: '0 0 24px rgba(0,212,255,0.3)',
          }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2L3 6.5V12c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V6.5L12 2z"
              fill="white"
              opacity="0.95"
            />
            <path d="M9 12l2 2 4-4" stroke="rgba(0,212,255,0.9)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h1
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '22px',
            fontWeight: 700,
            color: '#f0f9ff',
            marginBottom: '6px',
            letterSpacing: '-0.02em',
          }}
        >
          Sign in
        </h1>
        <p style={{ fontSize: '14px', color: '#64748b' }}>
          AI Driver Safety & Intelligence Platform
        </p>
      </div>

      {/* Error */}
      {error && (
        <div
          id="login-error"
          style={{
            background: 'rgba(239,68,68,0.10)',
            border: '1px solid rgba(239,68,68,0.25)',
            borderRadius: '10px',
            padding: '12px 16px',
            marginBottom: '20px',
            fontSize: '14px',
            color: '#fca5a5',
          }}
        >
          {error}
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div>
          <label
            htmlFor="login-email"
            style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: '#94a3b8', marginBottom: '6px' }}
          >
            Email address
          </label>
          <input
            id="login-email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            style={{
              width: '100%',
              background: 'rgba(14,30,60,0.6)',
              border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: '10px',
              padding: '11px 14px',
              fontSize: '14px',
              color: '#f0f9ff',
              outline: 'none',
              transition: 'border-color 0.2s',
              boxSizing: 'border-box',
            }}
            onFocus={(e) => { e.target.style.borderColor = 'rgba(0,212,255,0.45)'; }}
            onBlur={(e)  => { e.target.style.borderColor = 'rgba(0,212,255,0.15)'; }}
          />
        </div>

        <div>
          <label
            htmlFor="login-password"
            style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: '#94a3b8', marginBottom: '6px' }}
          >
            Password
          </label>
          <input
            id="login-password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            style={{
              width: '100%',
              background: 'rgba(14,30,60,0.6)',
              border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: '10px',
              padding: '11px 14px',
              fontSize: '14px',
              color: '#f0f9ff',
              outline: 'none',
              transition: 'border-color 0.2s',
              boxSizing: 'border-box',
            }}
            onFocus={(e) => { e.target.style.borderColor = 'rgba(0,212,255,0.45)'; }}
            onBlur={(e)  => { e.target.style.borderColor = 'rgba(0,212,255,0.15)'; }}
          />
        </div>

        <button
          id="login-submit"
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            background: loading
              ? 'rgba(0,212,255,0.3)'
              : 'linear-gradient(135deg, #00d4ff, #6366f1)',
            border: 'none',
            borderRadius: '10px',
            padding: '13px',
            fontSize: '15px',
            fontWeight: 600,
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
            fontFamily: 'inherit',
            marginTop: '4px',
            boxShadow: loading ? 'none' : '0 0 24px rgba(0,212,255,0.25)',
          }}
        >
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
      </form>

      <p style={{ textAlign: 'center', fontSize: '14px', color: '#475569', marginTop: '24px' }}>
        Don&apos;t have an account?{' '}
        <Link
          href="/signup"
          style={{ color: '#00d4ff', textDecoration: 'none', fontWeight: 500 }}
        >
          Sign up
        </Link>
      </p>
    </div>
  );
}
