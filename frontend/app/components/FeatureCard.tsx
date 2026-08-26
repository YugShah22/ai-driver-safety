'use client';

import { useState } from 'react';
import type { Feature } from '@/types';

interface FeatureCardProps {
  feature: Feature;
  index: number;
}

export default function FeatureCard({ feature, index }: FeatureCardProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      className="animate-fade-in-up"
      style={{ animationDelay: `${index * 0.1}s` }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        style={{
          background: hovered
            ? 'rgba(14,30,60,0.85)'
            : 'rgba(8,18,40,0.60)',
          border: `1px solid ${hovered ? 'rgba(0,212,255,0.28)' : 'rgba(0,212,255,0.10)'}`,
          borderRadius: '20px',
          padding: '32px',
          height: '100%',
          backdropFilter: 'blur(20px)',
          transition: 'all 0.3s cubic-bezier(0.4,0,0.2,1)',
          transform: hovered ? 'translateY(-6px)' : 'translateY(0)',
          boxShadow: hovered
            ? '0 20px 60px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,255,0.08)'
            : '0 4px 24px rgba(0,0,0,0.25)',
          cursor: 'default',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Gradient orb on hover */}
        {hovered && (
          <div
            style={{
              position: 'absolute',
              top: '-40px',
              right: '-40px',
              width: '160px',
              height: '160px',
              background: feature.gradient,
              borderRadius: '50%',
              opacity: 0.08,
              pointerEvents: 'none',
              filter: 'blur(30px)',
            }}
          />
        )}

        {/* Icon */}
        <div
          style={{
            width: '52px',
            height: '52px',
            borderRadius: '14px',
            background: feature.gradient,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '20px',
            fontSize: '24px',
            boxShadow: hovered ? `0 0 24px rgba(0,212,255,0.25)` : 'none',
            transition: 'box-shadow 0.3s',
            flexShrink: 0,
          }}
        >
          {feature.icon}
        </div>

        {/* Title */}
        <h3
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '18px',
            fontWeight: 700,
            color: '#f0f9ff',
            marginBottom: '10px',
            letterSpacing: '-0.01em',
          }}
        >
          {feature.title}
        </h3>

        {/* Description */}
        <p
          style={{
            fontSize: '14px',
            color: hovered ? '#94a3b8' : '#64748b',
            lineHeight: 1.65,
            transition: 'color 0.3s',
          }}
        >
          {feature.description}
        </p>
      </div>
    </div>
  );
}
