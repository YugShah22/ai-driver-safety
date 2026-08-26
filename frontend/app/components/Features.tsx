'use client';

import FeatureCard from './FeatureCard';
import type { Feature } from '@/types';

const features: Feature[] = [
  {
    title: 'YOLO Object Detection',
    description:
      'Real-time detection of vehicles, pedestrians, cyclists, traffic signs, and road obstacles from dashcam frames using state-of-the-art YOLO models.',
    icon: '🎯',
    gradient: 'linear-gradient(135deg, #00d4ff, #0ea5e9)',
  },
  {
    title: 'CNN Scene Understanding',
    description:
      'Custom Convolutional Neural Networks classify driving scenes — highway, urban, rural, adverse weather — providing context for risk assessment.',
    icon: '🧠',
    gradient: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
  },
  {
    title: 'Transfer Learning',
    description:
      'Fine-tuned ResNet50 and EfficientNet models pre-trained on ImageNet, adapted for dashcam-specific feature extraction with domain expertise.',
    icon: '⚡',
    gradient: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
  },
  {
    title: 'Optical Flow Analysis',
    description:
      'OpenCV-powered optical flow estimates vehicle speed, relative motion, and following distance — key inputs to the risk scoring engine.',
    icon: '🌊',
    gradient: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
  },
  {
    title: 'XGBoost Risk Scoring',
    description:
      'Gradient-boosted ensemble model combines engineered driving features into a calibrated risk score with SHAP-based explainability.',
    icon: '📊',
    gradient: 'linear-gradient(135deg, #10b981, #06b6d4)',
  },
  {
    title: 'ANN Risk Classifier',
    description:
      'Artificial Neural Network classifies trips into Low / Medium / High / Critical risk levels, trained on labeled dashcam datasets.',
    icon: '🔮',
    gradient: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
  },
  {
    title: 'Supabase Realtime',
    description:
      'Analysis progress and results stream to the dashboard in real-time via Supabase Realtime subscriptions — no polling required.',
    icon: '📡',
    gradient: 'linear-gradient(135deg, #f59e0b, #10b981)',
  },
  {
    title: 'Secure Storage & Auth',
    description:
      'Dashcam videos are stored securely in Supabase Storage with row-level security. Auth ensures only you access your footage and reports.',
    icon: '🔐',
    gradient: 'linear-gradient(135deg, #ef4444, #f59e0b)',
  },
  {
    title: 'Risk Recommendations',
    description:
      'Beyond scores, the platform generates personalized, actionable safety recommendations based on the specific risk patterns detected in your footage.',
    icon: '💡',
    gradient: 'linear-gradient(135deg, #00d4ff, #10b981)',
  },
];

export default function Features() {
  return (
    <section
      id="features"
      style={{
        padding: '120px 24px',
        background: 'linear-gradient(180deg, #020810 0%, #050d1f 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background grid */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(0,212,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.02) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
          pointerEvents: 'none',
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
              background: 'rgba(99,102,241,0.10)',
              border: '1px solid rgba(99,102,241,0.22)',
              borderRadius: '100px',
              padding: '6px 16px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#8b5cf6',
              marginBottom: '20px',
            }}
          >
            ✦ Platform Capabilities
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
            Every Layer of the{' '}
            <span
              style={{
                background: 'linear-gradient(135deg, #00d4ff, #8b5cf6)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              AI Pipeline
            </span>
          </h2>
          <p style={{ fontSize: '17px', color: '#64748b', maxWidth: '560px', margin: '0 auto', lineHeight: 1.7 }}>
            From raw video frames to actionable risk assessments — a complete,
            production-quality AI stack for driver safety intelligence.
          </p>
        </div>

        {/* Feature grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
            gap: '20px',
          }}
        >
          {features.map((feature, i) => (
            <FeatureCard key={feature.title} feature={feature} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
