'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

const navLinks = [
  { label: 'Features', href: '#features' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Tech Stack', href: '#tech-stack' },
  { label: 'Docs', href: '/docs' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav
      id="navbar"
      role="navigation"
      aria-label="Main navigation"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        transition: 'all 0.3s cubic-bezier(0.4,0,0.2,1)',
        background: scrolled
          ? 'rgba(2,8,16,0.88)'
          : 'transparent',
        backdropFilter: scrolled ? 'blur(24px)' : 'none',
        borderBottom: scrolled
          ? '1px solid rgba(0,212,255,0.10)'
          : '1px solid transparent',
      }}
    >
      <div
        style={{
          maxWidth: '1280px',
          margin: '0 auto',
          padding: '0 24px',
          height: '72px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Logo */}
        <Link
          href="/"
          id="nav-logo"
          style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}
        >
          {/* Shield Icon */}
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 20px rgba(0,212,255,0.35)',
            }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2L3 6.5V12c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V6.5L12 2z"
                fill="white"
                opacity="0.9"
              />
              <path
                d="M9 12l2 2 4-4"
                stroke="rgba(0,212,255,0.8)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <span
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontWeight: 700,
              fontSize: '17px',
              background: 'linear-gradient(90deg, #f0f9ff, #00d4ff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.02em',
            }}
          >
            DriveAI
          </span>
        </Link>

        {/* Desktop Nav Links */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
          className="hidden-mobile"
        >
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              id={`nav-link-${link.label.toLowerCase().replace(/\s+/g, '-')}`}
              style={{
                color: 'rgba(148,163,184,0.9)',
                textDecoration: 'none',
                fontSize: '14px',
                fontWeight: 500,
                padding: '8px 14px',
                borderRadius: '8px',
                transition: 'all 0.2s',
                letterSpacing: '0.01em',
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLElement).style.color = '#f0f9ff';
                (e.target as HTMLElement).style.background = 'rgba(0,212,255,0.08)';
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLElement).style.color = 'rgba(148,163,184,0.9)';
                (e.target as HTMLElement).style.background = 'transparent';
              }}
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* CTA */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            id="nav-cta-login"
            style={{
              background: 'transparent',
              border: '1px solid rgba(0,212,255,0.25)',
              color: '#00d4ff',
              padding: '8px 18px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: 'inherit',
            }}
            onMouseEnter={(e) => {
              (e.target as HTMLElement).style.background = 'rgba(0,212,255,0.08)';
              (e.target as HTMLElement).style.borderColor = 'rgba(0,212,255,0.5)';
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLElement).style.background = 'transparent';
              (e.target as HTMLElement).style.borderColor = 'rgba(0,212,255,0.25)';
            }}
          >
            Sign In
          </button>
          <button
            id="nav-cta-start"
            style={{
              background: 'linear-gradient(135deg, #00d4ff, #6366f1)',
              border: 'none',
              color: '#fff',
              padding: '9px 20px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: 'inherit',
              boxShadow: '0 0 20px rgba(0,212,255,0.25)',
            }}
            onMouseEnter={(e) => {
              (e.target as HTMLElement).style.transform = 'translateY(-1px)';
              (e.target as HTMLElement).style.boxShadow = '0 0 30px rgba(0,212,255,0.4)';
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLElement).style.transform = 'translateY(0)';
              (e.target as HTMLElement).style.boxShadow = '0 0 20px rgba(0,212,255,0.25)';
            }}
          >
            Get Started
          </button>

          {/* Mobile menu toggle */}
          <button
            id="nav-mobile-toggle"
            aria-label="Toggle menu"
            onClick={() => setMenuOpen(!menuOpen)}
            style={{
              display: 'none',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#94a3b8',
              padding: '4px',
            }}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {menuOpen ? (
                <>
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </>
              ) : (
                <>
                  <line x1="3" y1="7" x2="21" y2="7" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="17" x2="21" y2="17" />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div
          id="nav-mobile-menu"
          style={{
            background: 'rgba(2,8,20,0.97)',
            backdropFilter: 'blur(24px)',
            borderTop: '1px solid rgba(0,212,255,0.1)',
            padding: '16px 24px 24px',
          }}
        >
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onClick={() => setMenuOpen(false)}
              style={{
                display: 'block',
                color: '#94a3b8',
                textDecoration: 'none',
                fontSize: '15px',
                fontWeight: 500,
                padding: '12px 0',
                borderBottom: '1px solid rgba(0,212,255,0.06)',
              }}
            >
              {link.label}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
}
