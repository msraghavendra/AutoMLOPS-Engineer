'use client';

import { useEffect, useState, useCallback } from 'react';
import { BarChart2, Zap, TrendingUp, Activity, Shield, RefreshCw, Clock, Database, AlertTriangle, CheckCircle } from 'lucide-react';
import { getActiveDeployment, getModels, getAuditLogs, type Deployment, type TrainedModel, type AuditLog } from '@/lib/api';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const CHART_COLORS = { blue: '#3b82f6', violet: '#8b5cf6', emerald: '#10b981', amber: '#f59e0b', rose: '#f43f5e', cyan: '#06b6d4' };

function MetricKPI({ label, value, sub, icon: Icon, color }: { label: string; value: string | number; sub?: string; icon: React.ElementType; color: string }) {
  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium uppercase tracking-wider mb-1" style={{ color: '#64748b' }}>{label}</div>
          <div className="text-2xl font-bold mt-1" style={{ color: 'var(--text-primary)' }}>{value}</div>
          {sub && <div className="text-xs mt-1" style={{ color: '#64748b' }}>{sub}</div>}
        </div>
        <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: `${color}1a`, border: `1px solid ${color}30` }}>
          <Icon size={18} style={{ color }} />
        </div>
      </div>
    </div>
  );
}

function Sparkline({ color }: { color: string }) {
  const data = Array.from({ length: 20 }, (_, i) => ({ v: Math.floor(Math.random() * 60) + 20 + i * 2 }));
  return (
    <ResponsiveContainer width="100%" height={60}>
      <AreaChart data={data} margin={{ top: 4, bottom: 4, left: 0, right: 0 }}>
        <defs>
          <linearGradient id={`grad-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} fill={`url(#grad-${color})`} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-3 py-2 text-xs" style={{ border: '1px solid #2a3f5f' }}>
      <div style={{ color: '#94a3b8' }}>{label}</div>
      <div className="font-semibold" style={{ color: '#e2e8f0' }}>{payload[0].value}</div>
    </div>
  );
}

export default function OverviewDashboard() {
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [d, m, l] = await Promise.allSettled([getActiveDeployment(), getModels(), getAuditLogs({ limit: 10 })]);
      if (d.status === 'fulfilled') setDeployment(d.value);
      if (m.status === 'fulfilled') setModels(m.value);
      if (l.status === 'fulfilled') setLogs(l.value);
    } finally {
      setLoading(false);
      setLastRefresh(new Date());
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const activeModel = deployment?.model;
  const doneModels = models.filter(m => m.status === 'DONE');
  const trainingModels = models.filter(m => m.status === 'TRAINING' || m.status === 'PENDING');

  // Simulate prediction trend data
  const predTrend = Array.from({ length: 12 }, (_, i) => ({
    h: `${i * 2}h`,
    preds: Math.floor(Math.random() * 120) + 30
  }));

  const metricKey = activeModel?.metrics ? (Object.keys(activeModel.metrics).find(k => ['f1_score', 'r2_score'].includes(k))) : null;
  const metricVal = metricKey && activeModel?.metrics ? (activeModel.metrics[metricKey] * 100).toFixed(1) + '%' : '—';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient-blue">AutoMLOps Dashboard</h1>
          <p className="text-sm mt-1" style={{ color: '#64748b' }}>Full lifecycle ML automation — train, deploy, monitor, retrain.</p>
        </div>
        <button onClick={load} disabled={loading} className="btn-secondary" id="btn-refresh-dashboard">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricKPI label="Active Model" value={activeModel ? `v${activeModel.version}` : 'None'} sub={activeModel?.algorithm || 'No deployment'} icon={Zap} color={CHART_COLORS.blue} />
        <MetricKPI label="Predictions Served" value={deployment?.prediction_count.toLocaleString() ?? '0'} sub="Total since deployment" icon={TrendingUp} color={CHART_COLORS.emerald} />
        <MetricKPI label={metricKey?.replace('_', ' ').toUpperCase() ?? 'Performance'} value={metricVal} sub="On validation set" icon={BarChart2} color={CHART_COLORS.violet} />
        <MetricKPI label="Models Trained" value={doneModels.length} sub={`${trainingModels.length} pending/running`} icon={Activity} color={CHART_COLORS.amber} />
      </div>

      {/* Charts + Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Prediction Trend */}
        <div className="glass-card p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="section-title">Prediction Volume</div>
              <div className="text-xs" style={{ color: '#64748b' }}>Simulated hourly prediction rate</div>
            </div>
            <span className="badge-info">Live</span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={predTrend}>
              <defs>
                <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={CHART_COLORS.blue} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={CHART_COLORS.blue} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" vertical={false} />
              <XAxis dataKey="h" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} width={35} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="preds" stroke={CHART_COLORS.blue} strokeWidth={2} fill="url(#predGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Deployment Status */}
        <div className="glass-card p-5 space-y-4">
          <div className="section-title">Deployment Status</div>
          {deployment ? (
            <>
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
                <span className="text-sm font-medium" style={{ color: '#34d399' }}>Active</span>
              </div>
              <div className="space-y-3 text-sm">
                {[
                  { label: 'Model', value: activeModel?.model_name ?? '—' },
                  { label: 'Algorithm', value: activeModel?.algorithm ?? '—' },
                  { label: 'Problem Type', value: deployment.model?.metrics ? Object.keys(deployment.model.metrics).includes('f1_score') ? 'Classification' : 'Regression' : '—' },
                  { label: 'Drift Threshold', value: `${(deployment.drift_threshold * 100).toFixed(0)}%` },
                  { label: 'Perf. Threshold', value: `${(deployment.performance_threshold * 100).toFixed(0)}%` },
                ].map(r => (
                  <div key={r.label} className="flex justify-between">
                    <span style={{ color: '#64748b' }}>{r.label}</span>
                    <span className="font-medium mono text-xs" style={{ color: '#e2e8f0' }}>{r.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-32 text-center">
              <Shield size={28} className="mb-2" style={{ color: '#475569' }} />
              <div className="text-sm" style={{ color: '#64748b' }}>No model deployed yet.</div>
              <div className="text-xs mt-1" style={{ color: '#475569' }}>Train a model and deploy it from the Model Registry tab.</div>
            </div>
          )}
        </div>
      </div>

      {/* Model Performance Bar Chart */}
      {doneModels.length > 0 && (
        <div className="glass-card p-5">
          <div className="section-title">Model Comparison</div>
          <div className="text-xs mb-4" style={{ color: '#64748b' }}>Performance scores across all trained models</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={doneModels.slice(0, 8).map(m => ({
              name: `v${m.version} ${m.algorithm.split(' ')[0]}`,
              score: m.metrics ? Math.round(((m.metrics.f1_score ?? m.metrics.r2_score ?? 0)) * 100) : 0,
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} domain={[0, 100]} width={30} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="score" fill={CHART_COLORS.violet} radius={[4, 4, 0, 0]} maxBarSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent Audit Logs */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="section-title">Recent Events</div>
          <span className="text-xs" style={{ color: '#64748b' }}>Last refresh: {lastRefresh.toLocaleTimeString()}</span>
        </div>
        {logs.length === 0 ? (
          <div className="text-sm text-center py-8" style={{ color: '#475569' }}>No events logged yet.</div>
        ) : (
          <div className="space-y-2">
            {logs.map(log => (
              <div key={log.id} className="flex items-start gap-3 p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <div className="mt-0.5">
                  {log.severity === 'ERROR' ? <AlertTriangle size={14} style={{ color: '#f43f5e' }} /> :
                   log.severity === 'WARNING' ? <AlertTriangle size={14} style={{ color: '#f59e0b' }} /> :
                   <CheckCircle size={14} style={{ color: '#10b981' }} />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={log.severity === 'ERROR' ? 'badge-error' : log.severity === 'WARNING' ? 'badge-warning' : 'badge-success'}>{log.event_type}</span>
                    <span className="text-xs mono" style={{ color: '#475569' }}>{new Date(log.created_at).toLocaleTimeString()}</span>
                  </div>
                  <div className="text-sm truncate" style={{ color: '#94a3b8' }}>{log.message}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
