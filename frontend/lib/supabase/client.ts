/**
 * Supabase BROWSER client.
 * Use ONLY in Client Components ('use client').
 * Uses the public anon key — service-role key is NEVER here.
 */
import { createBrowserClient } from '@supabase/ssr';
import type { Database } from '@/types/database';

export function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    if (typeof window !== 'undefined') {
      alert("⚠️ Supabase is not configured yet!\n\nPlease add NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY to your frontend/.env.local file to enable Sign Up and Login.");
    }
    // Return a dummy client to prevent crashes before the alert shows
    return {} as any;
  }

  return createBrowserClient<Database>(supabaseUrl, supabaseKey);
}
