'use client';

import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { UploadStatus, Trip } from '@/types';

const MAX_FILE_SIZE_MB = 500;
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/avi'];

export default function UploadPage() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);

  const [title, setTitle]           = useState('');
  const [file, setFile]             = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress]     = useState(0);
  const [error, setError]           = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;
    setError(null);

    if (!selected) { setFile(null); return; }

    if (!ALLOWED_TYPES.includes(selected.type)) {
      setError('Unsupported file type. Please upload MP4, MOV, AVI, or WebM.');
      setFile(null);
      return;
    }
    if (selected.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`);
      setFile(null);
      return;
    }
    setFile(selected);
    if (!title) setTitle(selected.name.replace(/\.[^/.]+$/, ''));
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !title.trim()) return;

    setError(null);
    setUploadStatus('uploading');
    setProgress(0);

    try {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');

      // ─── Upload video to Supabase Storage ─────────────────
      const ext       = file.name.split('.').pop();
      const filePath  = `${user.id}/${Date.now()}.${ext}`;

      const { error: uploadError } = await supabase.storage
        .from('videos')
        .upload(filePath, file, {
          cacheControl: '3600',
          upsert: false,
        });

      if (uploadError) throw uploadError;

      setProgress(70);
      setUploadStatus('creating');

      // ─── Create trip record in DB ──────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: trip, error: dbError } = await (supabase as any)
        .from('trips')
        .insert({
          user_id:    user.id,
          title:      title.trim(),
          video_path: filePath,
          status:     'UPLOADED',
        })
        .select()
        .single();

      if (dbError) throw dbError;
      const typedTrip = trip as Trip;

      setProgress(100);
      setUploadStatus('success');

      // Navigate to the new trip page
      setTimeout(() => router.push(`/trips/${typedTrip.id}`), 800);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
      setUploadStatus('error');
      setProgress(0);
    }
  }

  const isLoading = uploadStatus === 'uploading' || uploadStatus === 'creating';

  return (
    <div style={{ maxWidth: '600px' }}>
      <div style={{ marginBottom: '32px' }}>
        <h1
          style={{
            fontFamily: 'Space Grotesk, Inter, sans-serif',
            fontSize: '26px',
            fontWeight: 800,
            color: '#f0f9ff',
            letterSpacing: '-0.02em',
            marginBottom: '6px',
          }}
        >
          Upload Dashcam Video
        </h1>
        <p style={{ fontSize: '14px', color: '#64748b' }}>
          Upload a video to start AI analysis. Max {MAX_FILE_SIZE_MB} MB.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            background: 'rgba(239,68,68,0.10)',
            border: '1px solid rgba(239,68,68,0.22)',
            borderRadius: '12px',
            padding: '14px 18px',
            marginBottom: '24px',
            fontSize: '14px',
            color: '#fca5a5',
          }}
        >
          {error}
        </div>
      )}

      {/* Success */}
      {uploadStatus === 'success' && (
        <div
          style={{
            background: 'rgba(16,185,129,0.10)',
            border: '1px solid rgba(16,185,129,0.22)',
            borderRadius: '12px',
            padding: '14px 18px',
            marginBottom: '24px',
            fontSize: '14px',
            color: '#6ee7b7',
          }}
        >
          ✓ Trip created successfully! Redirecting…
        </div>
      )}

      <form
        onSubmit={handleUpload}
        style={{
          background: 'rgba(8,18,40,0.70)',
          border: '1px solid rgba(0,212,255,0.10)',
          borderRadius: '20px',
          padding: '32px',
          display: 'flex',
          flexDirection: 'column',
          gap: '22px',
        }}
      >
        {/* Trip title */}
        <div>
          <label htmlFor="upload-title" style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: '#94a3b8', marginBottom: '7px' }}>
            Trip name
          </label>
          <input
            id="upload-title"
            type="text"
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Morning commute, June 10"
            style={{
              width: '100%',
              background: 'rgba(14,30,60,0.6)',
              border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: '10px',
              padding: '11px 14px',
              fontSize: '14px',
              color: '#f0f9ff',
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>

        {/* File picker */}
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: '#94a3b8', marginBottom: '7px' }}>
            Video file
          </label>
          <div
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${file ? 'rgba(0,212,255,0.35)' : 'rgba(0,212,255,0.14)'}`,
              borderRadius: '12px',
              padding: '32px',
              textAlign: 'center',
              cursor: 'pointer',
              background: file ? 'rgba(0,212,255,0.04)' : 'transparent',
              transition: 'all 0.2s',
            }}
          >
            {file ? (
              <div>
                <p style={{ fontSize: '24px', marginBottom: '8px' }}>📹</p>
                <p style={{ fontSize: '14px', fontWeight: 600, color: '#e2e8f0', marginBottom: '4px' }}>{file.name}</p>
                <p style={{ fontSize: '12px', color: '#64748b' }}>{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <div>
                <p style={{ fontSize: '32px', marginBottom: '12px' }}>📂</p>
                <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '4px' }}>Click to select video</p>
                <p style={{ fontSize: '12px', color: '#475569' }}>MP4, MOV, AVI, WebM — up to {MAX_FILE_SIZE_MB} MB</p>
              </div>
            )}
          </div>
          <input
            ref={fileRef}
            id="upload-file"
            type="file"
            accept={ALLOWED_TYPES.join(',')}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        {/* Progress bar */}
        {isLoading && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', marginBottom: '6px' }}>
              <span>{uploadStatus === 'uploading' ? 'Uploading video…' : 'Creating trip record…'}</span>
              <span>{progress}%</span>
            </div>
            <div style={{ height: '4px', background: 'rgba(0,212,255,0.10)', borderRadius: '2px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${progress}%`,
                  background: 'linear-gradient(90deg, #00d4ff, #6366f1)',
                  borderRadius: '2px',
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>
        )}

        {/* Submit */}
        <button
          id="upload-submit"
          type="submit"
          disabled={isLoading || !file || !title.trim() || uploadStatus === 'success'}
          style={{
            background: (isLoading || !file || !title.trim())
              ? 'rgba(0,212,255,0.2)'
              : 'linear-gradient(135deg, #00d4ff, #6366f1)',
            border: 'none',
            borderRadius: '10px',
            padding: '13px',
            fontSize: '15px',
            fontWeight: 600,
            color: '#fff',
            cursor: (isLoading || !file || !title.trim()) ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
            transition: 'all 0.2s',
          }}
        >
          {isLoading ? 'Uploading…' : 'Upload & Create Trip'}
        </button>
      </form>
    </div>
  );
}
