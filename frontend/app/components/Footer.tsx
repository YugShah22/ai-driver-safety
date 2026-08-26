'use client';

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer
      id="footer"
      role="contentinfo"
      style={{
        borderTop: '1px solid rgba(0,212,255,0.08)',
        background: 'rgba(2,8,16,0.8)',
        backdropFilter: 'blur(20px)',
        padding: '48px 24px 32px',
      }}
    >
      <div
        style={{
          maxWidth: '1280px',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '32px',
        }}
      >
        {/* Top row */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
            gap: '32px',
          }}
        >
          {/* Brand */}
          <div style={{ maxWidth: '300px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L3 6.5V12c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V6.5L12 2z" fill="white" opacity="0.9" />
                  <path d="M9 12l2 2 4-4" stroke="rgba(0,212,255,0.8)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <span
                style={{
                  fontFamily: 'Space Grotesk, Inter, sans-serif',
                  fontWeight: 700,
                  fontSize: '16px',
                  background: 'linear-gradient(90deg, #f0f9ff, #00d4ff)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                DriveAI Platform
              </span>
            </div>
            <p style={{ fontSize: '13px', color: '#475569', lineHeight: 1.65 }}>
              Production-quality AI platform for dashcam-based driver safety analysis
              using computer vision, deep learning, and machine learning.
            </p>
          </div>

          {/* Links */}
          <div style={{ display: 'flex', gap: '64px', flexWrap: 'wrap' }}>
            <div>
              <h4 style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '14px' }}>
                Platform
              </h4>
              {['Features', 'How It Works', 'Tech Stack', 'API Docs'].map((item) => (
                <a
                  key={item}
                  href="#"
                  style={{ display: 'block', fontSize: '14px', color: '#475569', textDecoration: 'none', marginBottom: '10px', transition: 'color 0.2s' }}
                  onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#00d4ff'; }}
                  onMouseLeave={(e) => { (e.target as HTMLElement).style.color = '#475569'; }}
                >
                  {item}
                </a>
              ))}
            </div>
            <div>
              <h4 style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '14px' }}>
                Tech Stack
              </h4>
              {['Next.js · TypeScript', 'FastAPI · Python', 'PyTorch · YOLO', 'Supabase'].map((item) => (
                <p key={item} style={{ fontSize: '14px', color: '#475569', marginBottom: '10px' }}>
                  {item}
                </p>
              ))}
            </div>
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: '1px', background: 'rgba(0,212,255,0.06)' }} />

        {/* Bottom row */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <p style={{ fontSize: '13px', color: '#334155' }}>
            © {year} AI Driver Safety & Intelligence Platform. All rights reserved.
          </p>
          <div style={{ display: 'flex', gap: '20px' }}>
            {['Privacy Policy', 'Terms of Service', 'MIT License'].map((item) => (
              <a
                key={item}
                href="#"
                style={{ fontSize: '13px', color: '#334155', textDecoration: 'none', transition: 'color 0.2s' }}
                onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#94a3b8'; }}
                onMouseLeave={(e) => { (e.target as HTMLElement).style.color = '#334155'; }}
              >
                {item}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
