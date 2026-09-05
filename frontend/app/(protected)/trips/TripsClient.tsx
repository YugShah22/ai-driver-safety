'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import type { Trip } from '@/types';

type SortKey = 'date_desc' | 'date_asc' | 'name_asc';
type StatusFilter = 'ALL' | 'UPLOADED' | 'PROCESSING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';

export default function TripsClient({ initialTrips }: { initialTrips: Trip[] }) {
  const [search, setSearch]             = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('ALL');
  const [sort, setSort]                 = useState<SortKey>('date_desc');
  const [page, setPage]                 = useState(1);
  const PER_PAGE = 8;

  const filtered = useMemo(() => {
    let result = initialTrips;
    
    // Search
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((t) => t.title.toLowerCase().includes(q));
    }
    
    // Filter
    if (statusFilter !== 'ALL') {
      result = result.filter((t) => t.status === statusFilter);
    }
    
    // Sort
    switch (sort) {
      case 'date_asc':  
        result = [...result].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()); 
        break;
      case 'date_desc': 
        result = [...result].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()); 
        break;
      case 'name_asc':  
        result = [...result].sort((a, b) => a.title.localeCompare(b.title)); 
        break;
    }
    return result;
  }, [initialTrips, search, statusFilter, sort]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PER_PAGE));
  const paginated  = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  const statusColors: Record<string, { bg: string; text: string }> = {
    UPLOADED:   { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' },
    PROCESSING: { bg: 'rgba(245,158,11,0.12)',  text: '#fbbf24' },
    ANALYZING:  { bg: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
    COMPLETED:  { bg: 'rgba(16,185,129,0.12)',  text: '#34d399' },
    FAILED:     { bg: 'rgba(239,68,68,0.12)',   text: '#f87171' },
  };

  const selectStyle = {
    padding: '10px 14px',
    borderRadius: '10px',
    fontSize: '13px',
    background: 'rgba(14,30,60,0.6)',
    border: '1px solid rgba(0,212,255,0.15)',
    color: '#f0f9ff',
    outline: 'none',
    cursor: 'pointer'
  };

  return (
    <div>
      {/* Controls Bar */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '24px', background: 'rgba(8,18,40,0.60)', padding: '16px', borderRadius: '16px', border: '1px solid rgba(0,212,255,0.08)' }}>
        <input
          type="text"
          placeholder="Search trips..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          style={{
            flex: '1',
            minWidth: '200px',
            background: 'rgba(14,30,60,0.6)',
            border: '1px solid rgba(0,212,255,0.15)',
            borderRadius: '10px',
            padding: '10px 14px',
            fontSize: '13px',
            color: '#f0f9ff',
            outline: 'none',
          }}
        />
        <select 
          value={statusFilter} 
          onChange={(e) => { setStatusFilter(e.target.value as StatusFilter); setPage(1); }} 
          style={selectStyle}
        >
          <option value="ALL">All Statuses</option>
          <option value="UPLOADED">Uploaded</option>
          <option value="PROCESSING">Processing</option>
          <option value="ANALYZING">Analyzing</option>
          <option value="COMPLETED">Completed</option>
          <option value="FAILED">Failed</option>
        </select>
        <select 
          value={sort} 
          onChange={(e) => setSort(e.target.value as SortKey)} 
          style={selectStyle}
        >
          <option value="date_desc">Newest First</option>
          <option value="date_asc">Oldest First</option>
          <option value="name_asc">Name (A-Z)</option>
        </select>
      </div>

      {/* Trip List */}
      {paginated.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '48px 0', color: '#64748b', fontSize: '14px' }}>
          No trips match your current filters.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {paginated.map((trip) => {
            const { bg, text } = statusColors[trip.status] ?? { bg: 'transparent', text: '#94a3b8' };
            return (
              <Link
                key={trip.id}
                href={`/trips/${trip.id}`}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr auto auto',
                  alignItems: 'center',
                  gap: '16px',
                  background: 'rgba(8,18,40,0.65)',
                  border: '1px solid rgba(0,212,255,0.08)',
                  borderRadius: '14px',
                  padding: '18px 22px',
                  textDecoration: 'none',
                  transition: 'all 0.18s',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.22)';
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,212,255,0.08)';
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
                }}
              >
                <div>
                  <p style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0', marginBottom: '4px' }}>
                    {trip.title}
                  </p>
                  <p style={{ fontSize: '12px', color: '#475569' }}>
                    {new Date(trip.created_at).toLocaleString()}
                    {trip.duration ? ` · ${Math.round(trip.duration)}s` : ''}
                  </p>
                </div>
                <span
                  style={{
                    background: bg,
                    color: text,
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '4px 10px',
                    borderRadius: '6px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {trip.status}
                </span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
            );
          })}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: '32px' }}>
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{
              background: 'rgba(8,18,40,0.6)',
              border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: '8px',
              padding: '6px 12px',
              color: page === 1 ? '#475569' : '#e2e8f0',
              cursor: page === 1 ? 'not-allowed' : 'pointer',
              fontSize: '13px'
            }}
          >
            ← Prev
          </button>
          
          <span style={{ fontSize: '13px', color: '#64748b', padding: '0 8px' }}>
            Page {page} of {totalPages}
          </span>

          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            style={{
              background: 'rgba(8,18,40,0.6)',
              border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: '8px',
              padding: '6px 12px',
              color: page === totalPages ? '#475569' : '#e2e8f0',
              cursor: page === totalPages ? 'not-allowed' : 'pointer',
              fontSize: '13px'
            }}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
