'use client';

const steps = [
  {
    step: '01',
    title: 'Upload Dashcam Footage',
    description:
      'Upload your dashcam video through the secure interface. Files are encrypted and stored in Supabase Storage with row-level security.',
    icon: '📤',
    color: '#00d4ff',
  },
  {
    step: '02',
    title: 'Video Processing',
    description:
      'OpenCV extracts frames at configurable FPS. Optical flow analysis estimates vehicle speed, acceleration, and relative motion vectors.',
    icon: '🎬',
    color: '#0ea5e9',
  },
  {
    step: '03',
    title: 'Object Detection',
    description:
      'YOLOv8 detects and classifies road objects — vehicles, pedestrians, cyclists, traffic signs — with bounding boxes on every frame.',
    icon: '🎯',
    color: '#6366f1',
  },
  {
    step: '04',
    title: 'Feature Extraction',
    description:
      'CNN and Transfer Learning models extract scene features. The risk engine computes engineered features: following distance, lane discipline, headway time.',
    icon: '🧠',
    color: '#8b5cf6',
  },
  {
    step: '05',
    title: 'Risk Assessment',
    description:
      'XGBoost and ANN models score the trip and classify risk (Low / Medium / High / Critical). SHAP values explain which factors drive the score.',
    icon: '📊',
    color: '#10b981',
  },
  {
    step: '06',
    title: 'Insights & Recommendations',
    description:
      'Results stream to your dashboard in real-time via Supabase Realtime. Receive personalized safety recommendations and an annotated video.',
    icon: '💡',
    color: '#f59e0b',
  },
];

export default function HowItWorks() {
  return (
    <section
      id="how-it-works"
      style={{
        padding: '120px 24px',
        background: 'linear-gradient(180deg, #050d1f 0%, #060f28 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Glow accent */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: '1000px',
          height: '600px',
          background: 'radial-gradient(ellipse, rgba(99,102,241,0.05) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      <div style={{ maxWidth: '1280px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
        {/* Section header */}
        <div style={{ textAlign: 'center', marginBottom: '80px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: 'rgba(0,212,255,0.08)',
              border: '1px solid rgba(0,212,255,0.18)',
              borderRadius: '100px',
              padding: '6px 16px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#00d4ff',
              marginBottom: '20px',
            }}
          >
            ✦ The Pipeline
          </div>
          <h2
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: 'clamp(32px, 4vw, 52px)',
              fontWeight: 800,
              color: '#f0f9ff',
              letterSpacing: '-0.03em',
              lineHeight: 1.1,
              marginBottom: '20px',
            }}
          >
            From Video to{' '}
            <span
              style={{
                background: 'linear-gradient(135deg, #00d4ff, #10b981)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Risk Assessment
            </span>
          </h2>
          <p style={{ fontSize: '17px', color: '#64748b', maxWidth: '520px', margin: '0 auto', lineHeight: 1.7 }}>
            Six stages transform raw dashcam footage into an actionable, explainable safety report.
          </p>
        </div>

        {/* Steps */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
            gap: '20px',
          }}
        >
          {steps.map((s, i) => (
            <div
              key={s.step}
              className="animate-fade-in-up"
              style={{ animationDelay: `${i * 0.08}s` }}
            >
              <div
                style={{
                  background: 'rgba(8,18,40,0.60)',
                  border: '1px solid rgba(0,212,255,0.08)',
                  borderRadius: '20px',
                  padding: '28px',
                  backdropFilter: 'blur(16px)',
                  transition: 'all 0.3s',
                  position: 'relative',
                  overflow: 'hidden',
                }}
                onMouseEnter={(e) => {
                  const el = e.currentTarget;
                  el.style.borderColor = `${s.color}44`;
                  el.style.transform = 'translateY(-4px)';
                  el.style.boxShadow = `0 16px 48px rgba(0,0,0,0.35), 0 0 24px ${s.color}18`;
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget;
                  el.style.borderColor = 'rgba(0,212,255,0.08)';
                  el.style.transform = 'translateY(0)';
                  el.style.boxShadow = 'none';
                }}
              >
                {/* Step number */}
                <div
                  style={{
                    position: 'absolute',
                    top: '20px',
                    right: '20px',
                    fontFamily: 'Space Grotesk, sans-serif',
                    fontSize: '36px',
                    fontWeight: 800,
                    color: 'rgba(255,255,255,0.04)',
                    lineHeight: 1,
                    pointerEvents: 'none',
                  }}
                >
                  {s.step}
                </div>

                {/* Icon */}
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '14px',
                    background: `${s.color}18`,
                    border: `1px solid ${s.color}30`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '22px',
                    marginBottom: '18px',
                  }}
                >
                  {s.icon}
                </div>

                <h3
                  style={{
                    fontFamily: 'Space Grotesk, Inter, sans-serif',
                    fontSize: '17px',
                    fontWeight: 700,
                    color: '#f0f9ff',
                    marginBottom: '10px',
                    letterSpacing: '-0.01em',
                  }}
                >
                  {s.title}
                </h3>
                <p style={{ fontSize: '14px', color: '#64748b', lineHeight: 1.65 }}>
                  {s.description}
                </p>

                {/* Step indicator */}
                <div
                  style={{
                    marginTop: '20px',
                    height: '2px',
                    borderRadius: '1px',
                    background: `linear-gradient(90deg, ${s.color}60, transparent)`,
                    width: '60%',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
