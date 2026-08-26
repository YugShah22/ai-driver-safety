import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import HowItWorks from './components/HowItWorks';
import TechStack from './components/TechStack';
import Footer from './components/Footer';

/**
 * Landing page — assembles all sections.
 * Server component: no 'use client' here.
 */
export default function HomePage() {
  return (
    <main>
      <Navbar />
      <Hero />
      <Features />
      <HowItWorks />
      <TechStack />
      <Footer />
    </main>
  );
}
