'use client';

import Image from 'next/image';

const stats = [
  { value: 'YOLO', label: 'Object Detection' },
  { value: 'CNN', label: 'Feature Learning' },
  { value: 'XGBoost', label: 'Risk Scoring' },
  { value: 'Real-time', label: 'Analysis' },
];

export default function Hero() {
  return (
    <section
      id="hero"
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
        paddingTop: '80px',
        paddingBottom: '80px',
        background: 'linear-gradient(135deg, #020810 0%, #050d1f 50%, #060f28 100%)',
      }}
    >
      {/* Background effects */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
          pointerEvents: 'none',
        }}
      />

      {/* Radial glow top-center */}
      <div
        style={{
          position: 'absolute',
          top: '-10%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '800px',
          height: '500px',
          background: 'radial-gradient(ellipse, rgba(0,212,255,0.10) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Radial glow bottom-right */}
      <div
        style={{
          position: 'absolute',
          bottom: '0',
          right: '-10%',
          width: '600px',
          height: '500px',
          background: 'radial-gradient(ellipse, rgba(99,102,241,0.10) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Content */}
      <div
        style={{
          maxWidth: '1280px',
          width: '100%',
          margin: '0 auto',
          padding: '0 24px',
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '48px',
        }}
      >
        {/* Badge */}
        <div
          className="animate-fade-in-up"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(0,212,255,0.08)',
            border: '1px solid rgba(0,212,255,0.20)',
            borderRadius: '100px',
            padding: '6px 16px 6px 10px',
            fontSize: '13px',
            fontWeight: 500,
            color: '#00d4ff',
          }}
        >
          <span
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: '#00d4ff',
              boxShadow: '0 0 8px #00d4ff',
              animation: 'pulse-glow 2s ease-in-out infinite',
              display: 'inline-block',
            }}
          />
          Production-Quality AI Platform · Phase 1
        </div>

        {/* Headline */}
        <div
          className="animate-fade-in-up delay-100"
          style={{ textAlign: 'center', maxWidth: '860px' }}
        >
          <h1
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: 'clamp(40px, 6vw, 80px)',
              fontWeight: 800,
              lineHeight: 1.05,
              letterSpacing: '-0.03em',
              marginBottom: '24px',
              color: '#f0f9ff',
            }}
          >
            AI-Powered{' '}
            <span
              style={{
                background: 'linear-gradient(135deg, #00d4ff 0%, #0ea5e9 50%, #8b5cf6 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Driver Safety
            </span>
            <br />
            Intelligence Platform
          </h1>
          <p
            style={{
              fontSize: 'clamp(16px, 2vw, 20px)',
              color: '#94a3b8',
              maxWidth: '680px',
              margin: '0 auto',
              lineHeight: 1.7,
              fontWeight: 400,
            }}
          >
            Analyze dashcam footage with computer vision and deep learning.
            Detect road objects, understand your driving environment, and receive
            an intelligent risk assessment — powered by YOLO, PyTorch, and XGBoost.
          </p>
        </div>

        {/* CTA Buttons */}
        <div
          className="animate-fade-in-up delay-200"
          style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', justifyContent: 'center' }}
        >
          <button
            id="hero-cta-upload"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              background: 'linear-gradient(135deg, #00d4ff, #0ea5e9, #6366f1)',
              border: 'none',
              color: '#fff',
              padding: '15px 32px',
              borderRadius: '12px',
              fontSize: '16px',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.25s',
              fontFamily: 'inherit',
              boxShadow: '0 0 40px rgba(0,212,255,0.30), 0 4px 20px rgba(0,0,0,0.3)',
              letterSpacing: '0.01em',
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget;
              el.style.transform = 'translateY(-2px) scale(1.02)';
              el.style.boxShadow = '0 0 60px rgba(0,212,255,0.45), 0 8px 30px rgba(0,0,0,0.4)';
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget;
              el.style.transform = 'translateY(0) scale(1)';
              el.style.boxShadow = '0 0 40px rgba(0,212,255,0.30), 0 4px 20px rgba(0,0,0,0.3)';
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17,8 12,3 7,8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload Dashcam Footage
          </button>

          <button
            id="hero-cta-learn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              background: 'rgba(14,30,60,0.6)',
              border: '1px solid rgba(0,212,255,0.20)',
              color: '#f0f9ff',
              padding: '14px 28px',
              borderRadius: '12px',
              fontSize: '16px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.25s',
              fontFamily: 'inherit',
              backdropFilter: 'blur(12px)',
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget;
              el.style.borderColor = 'rgba(0,212,255,0.45)';
              el.style.background = 'rgba(0,212,255,0.08)';
              el.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget;
              el.style.borderColor = 'rgba(0,212,255,0.20)';
              el.style.background = 'rgba(14,30,60,0.6)';
              el.style.transform = 'translateY(0)';
            }}
          >
            How It Works
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Stats row */}
        <div
          className="animate-fade-in-up delay-300"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0',
            background: 'rgba(8,18,40,0.6)',
            border: '1px solid rgba(0,212,255,0.10)',
            borderRadius: '16px',
            backdropFilter: 'blur(20px)',
            overflow: 'hidden',
            flexWrap: 'wrap',
          }}
        >
          {stats.map((stat, i) => (
            <div
              key={stat.label}
              style={{
                padding: '20px 32px',
                borderRight: i < stats.length - 1 ? '1px solid rgba(0,212,255,0.08)' : 'none',
                textAlign: 'center',
                minWidth: '130px',
              }}
            >
              <div
                style={{
                  fontFamily: 'Space Grotesk, sans-serif',
                  fontSize: '18px',
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #00d4ff, #8b5cf6)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  marginBottom: '4px',
                }}
              >
                {stat.value}
              </div>
              <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Hero image */}
        <div
          className="animate-fade-in-up delay-400 animate-float"
          style={{
            width: '100%',
            maxWidth: '900px',
            borderRadius: '20px',
            overflow: 'hidden',
            border: '1px solid rgba(0,212,255,0.15)',
            boxShadow: '0 0 80px rgba(0,212,255,0.12), 0 40px 80px rgba(0,0,0,0.5)',
            position: 'relative',
          }}
        >
          {/* Scan line animation overlay */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'linear-gradient(180deg, transparent 0%, rgba(0,212,255,0.04) 50%, transparent 100%)',
              backgroundSize: '100% 6px',
              zIndex: 1,
              pointerEvents: 'none',
              mixBlendMode: 'overlay',
            }}
          />
          <Image
            src="/hero-dashboard.png"
            alt="AI Driver Safety Dashboard — dashcam analysis with object detection overlays and risk scoring"
            width={900}
            height={560}
            style={{ width: '100%', height: 'auto', display: 'block' }}
            priority
          />
        </div>
      </div>
    </section>
  );
}
