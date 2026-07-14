'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  ScrollText, RefreshCw, Filter, AlertTriangle,
  Info, CheckCircle, AlertCircle, Clock
} from 'lucide-react';
import { getAuditLogs, type AuditLog } from '@/lib/api';

const EVENT_TYPES = ['', 'TRAINING', 'DEPLOYMENT', 'DRIFT', 'RETRAINING', 'SYSTEM'];
const SEVERITIES = ['', 'INFO', 'WARNING', 'ERROR'];

const SEVERITY_STYLES: Record<string, { badge: string; icon: React.ElementType; color: string }> = {
  INFO: { badge: 'badge-info', icon: Info, color: '#3b82f6' },
  WARNING: { badge: 'badge-warning', icon: AlertTriangle, color: '#f59e0b' },
  ERROR: { badge: 'badge-error', icon: AlertCircle, color: '#ef4444' },
};

const EVENT_BADGE: Record<string, string> = {
  TRAINING: 'badge-info',
  DEPLOYMENT: 'badge-success',
  DRIFT: 'badge-warning',
  RETRAINING: 'badge-warning',
  SYSTEM: 'badge-neutral',
};

export default function AuditLogs() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [eventType, setEventType] = useState('');
  const [severity, setSeverity] = useState('');
  const [limit, setLimit] = useState(50);

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getAuditLogs({
        event_type: eventType || undefined,
        severity: severity || undefined,
        limit,
      });
      setLogs(result);
    } catch {/* ignore */} finally {
      setLoading(false);
    }
  }, [eventType, severity, limit]);

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  const infoCount = logs.filter(l => l.severity === 'INFO').length;
  const warnCount = logs.filter(l => l.severity === 'WARNING').length;
  const errCount = logs.filter(l => l.severity === 'ERROR').length;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient-blue">Audit Logs</h1>
          <p className="text-sm mt-1" style={{ color: '#64748b' }}>
            Full chronological history of all MLOps pipeline events.
          </p>
        </div>
        <button
          id="btn-refresh-logs"
          onClick={fetchLogs}
          className="btn-secondary"
          disabled={loading}
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Info Events', count: infoCount, icon: Info, color: '#3b82f6', bg: 'rgba(59,130,246,0.08)' },
          { label: 'Warnings', count: warnCount, icon: AlertTriangle, color: '#f59e0b', bg: 'rgba(245,158,11,0.08)' },
          { label: 'Errors', count: errCount, icon: AlertCircle, color: '#ef4444', bg: 'rgba(239,68,68,0.08)' },
        ].map(({ label, count, icon: Icon, color, bg }) => (
          <div key={label} className="glass-card p-4 flex items-center gap-3" style={{ background: bg }}>
            <Icon size={18} style={{ color }} />
            <div>
              <div className="text-xs" style={{ color: '#64748b' }}>{label}</div>
              <div className="font-bold text-xl mono" style={{ color }}>{count}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Filter size={14} style={{ color: '#64748b' }} />
          <span className="text-xs font-medium" style={{ color: '#94a3b8' }}>Filters</span>
        </div>
        <div className="flex flex-wrap gap-3">
          <div>
            <label className="text-xs block mb-1" style={{ color: '#64748b' }}>Event Type</label>
            <select
              id="filter-event-type"
              value={eventType}
              onChange={e => setEventType(e.target.value)}
              className="text-sm rounded-lg px-3 py-2 outline-none"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid #1e2d47', color: '#e2e8f0' }}
            >
              {EVENT_TYPES.map(t => <option key={t} value={t} style={{ background: '#0a0f1e' }}>{t || 'All Events'}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs block mb-1" style={{ color: '#64748b' }}>Severity</label>
            <select
              id="filter-severity"
              value={severity}
              onChange={e => setSeverity(e.target.value)}
              className="text-sm rounded-lg px-3 py-2 outline-none"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid #1e2d47', color: '#e2e8f0' }}
            >
              {SEVERITIES.map(s => <option key={s} value={s} style={{ background: '#0a0f1e' }}>{s || 'All Severities'}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs block mb-1" style={{ color: '#64748b' }}>Limit</label>
            <select
              id="filter-limit"
              value={limit}
              onChange={e => setLimit(Number(e.target.value))}
              className="text-sm rounded-lg px-3 py-2 outline-none"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid #1e2d47', color: '#e2e8f0' }}
            >
              {[25, 50, 100, 200].map(n => <option key={n} value={n} style={{ background: '#0a0f1e' }}>{n} rows</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Logs Table */}
      <div className="glass-card p-5 fade-in">
        <div className="flex items-center gap-2 mb-4">
          <ScrollText size={16} style={{ color: '#64748b' }} />
          <span className="section-title">Event Log</span>
          <span className="text-xs ml-auto" style={{ color: '#475569' }}>{logs.length} events</span>
        </div>

        {loading ? (
          <div className="flex flex-col items-center gap-4 py-12">
            <div className="w-10 h-10 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
            <div className="text-sm" style={{ color: '#64748b' }}>Loading logs…</div>
          </div>
        ) : logs.length === 0 ? (
          <div className="flex flex-col items-center gap-4 py-12 text-center">
            <div className="w-12 h-12 rounded-full flex items-center justify-center" style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)' }}>
              <CheckCircle size={20} style={{ color: '#3b82f6' }} />
            </div>
            <div className="text-sm" style={{ color: '#64748b' }}>No log events found for the current filters.</div>
          </div>
        ) : (
          <div className="space-y-1">
            {logs.map(log => {
              const sev = SEVERITY_STYLES[log.severity] ?? SEVERITY_STYLES['INFO'];
              const SevIcon = sev.icon;
              return (
                <div
                  key={log.id}
                  className="flex items-start gap-3 px-3 py-3 rounded-lg transition-all duration-150 hover:bg-white/[0.02]"
                  style={{ borderBottom: '1px solid #0d1a2d' }}
                >
                  <SevIcon size={14} style={{ color: sev.color, marginTop: 2, flexShrink: 0 }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className={EVENT_BADGE[log.event_type] ?? 'badge-neutral'}>{log.event_type}</span>
                      <span className={sev.badge}>{log.severity}</span>
                    </div>
                    <div className="text-sm" style={{ color: '#94a3b8', lineHeight: '1.5' }}>{log.message}</div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0" style={{ color: '#475569' }}>
                    <Clock size={11} />
                    <span className="text-xs">{new Date(log.created_at).toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
