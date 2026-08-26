'use client';

const techStack = [
  {
    category: 'Frontend',
    color: '#00d4ff',
    items: [
      { name: 'Next.js 15', desc: 'React framework with App Router' },
      { name: 'TypeScript', desc: 'Type-safe development' },
      { name: 'Tailwind CSS', desc: 'Utility-first styling' },
    ],
  },
  {
    category: 'Backend',
    color: '#6366f1',
    items: [
      { name: 'FastAPI', desc: 'High-performance Python API' },
      { name: 'Pydantic v2', desc: 'Data validation & settings' },
      { name: 'Uvicorn', desc: 'ASGI server with async support' },
    ],
  },
  {
    category: 'AI / ML',
    color: '#8b5cf6',
    items: [
      { name: 'PyTorch', desc: 'Deep learning framework' },
      { name: 'YOLO (Ultralytics)', desc: 'Real-time object detection' },
      { name: 'OpenCV', desc: 'Computer vision & video processing' },
      { name: 'scikit-learn', desc: 'Classical ML algorithms' },
      { name: 'XGBoost', desc: 'Gradient boosted risk scoring' },
    ],
  },
  {
    category: 'Database / Cloud',
    color: '#10b981',
    items: [
      { name: 'Supabase PostgreSQL', desc: 'Relational database with RLS' },
      { name: 'Supabase Auth', desc: 'Secure user authentication' },
      { name: 'Supabase Storage', desc: 'Video & file storage' },
      { name: 'Supabase Realtime', desc: 'Live analysis updates' },
    ],
  },
];

export default function TechStack() {
  return (
    <section
      id="tech-stack"
      style={{
        padding: '120px 24px',
        background: 'linear-gradient(180deg, #060f28 0%, #020810 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent)',
        }}
      />

      <div style={{ maxWidth: '1280px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
        {/* Section header */}
        <div style={{ textAlign: 'center', marginBottom: '72px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: 'rgba(16,185,129,0.08)',
              border: '1px solid rgba(16,185,129,0.20)',
              borderRadius: '100px',
              padding: '6px 16px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#10b981',
              marginBottom: '20px',
            }}
          >
            ✦ Technology
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
            Built on a{' '}
            <span
              style={{
                background: 'linear-gradient(135deg, #10b981, #00d4ff)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Modern Stack
            </span>
          </h2>
          <p style={{ fontSize: '17px', color: '#64748b', maxWidth: '520px', margin: '0 auto', lineHeight: 1.7 }}>
            Every technology in the stack was chosen for production-readiness, scalability,
            and AI/ML capability.
          </p>
        </div>

        {/* Tech grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '20px',
          }}
        >
          {techStack.map((category) => (
            <div
              key={category.category}
              style={{
                background: 'rgba(8,18,40,0.60)',
                border: `1px solid ${category.color}18`,
                borderRadius: '20px',
                padding: '28px',
                backdropFilter: 'blur(16px)',
                transition: 'all 0.3s',
              }}
              onMouseEnter={(e) => {
                const el = e.currentTarget;
                el.style.borderColor = `${category.color}35`;
                el.style.transform = 'translateY(-4px)';
                el.style.boxShadow = `0 16px 48px rgba(0,0,0,0.3), 0 0 20px ${category.color}12`;
              }}
              onMouseLeave={(e) => {
                const el = e.currentTarget;
                el.style.borderColor = `${category.color}18`;
                el.style.transform = 'translateY(0)';
                el.style.boxShadow = 'none';
              }}
            >
              {/* Category header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                <div
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: category.color,
                    boxShadow: `0 0 10px ${category.color}`,
                  }}
                />
                <h3
                  style={{
                    fontFamily: 'Space Grotesk, Inter, sans-serif',
                    fontSize: '14px',
                    fontWeight: 700,
                    color: category.color,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                  }}
                >
                  {category.category}
                </h3>
              </div>

              {/* Items */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {category.items.map((item) => (
                  <div key={item.name} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: '#e2e8f0' }}>
                      {item.name}
                    </span>
                    <span style={{ fontSize: '12px', color: '#475569', lineHeight: 1.5 }}>
                      {item.desc}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
