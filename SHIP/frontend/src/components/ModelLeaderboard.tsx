'use client';

import { useState, useCallback, useEffect } from 'react';
import { Trophy, Zap, RefreshCw, Rocket, ChevronDown, ChevronUp, Search } from 'lucide-react';
import { getModels, deployModel, getActiveDeployment, type TrainedModel, type Deployment } from '@/lib/api';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

const ALGO_COLORS: Record<string, string> = {
  'Random Forest': '#3b82f6',
  'Gradient Boosting': '#8b5cf6',
  'Logistic Regression': '#06b6d4',
  'SVM': '#f59e0b',
  'Ridge Regression': '#10b981',
  'Lasso Regression': '#f43f5e',
  'Linear Regression': '#ec4899',
  'Decision Tree': '#a78bfa',
};

function statusBadge(status: TrainedModel['status']) {
  if (status === 'DONE') return <span className="badge-success">✓ Done</span>;
  if (status === 'TRAINING') return <span className="badge-warning animate-pulse-slow">⚡ Training</span>;
  if (status === 'PENDING') return <span className="badge-neutral">⏳ Pending</span>;
  return <span className="badge-error">✗ Failed</span>;
}

function MetricRadar({ model }: { model: TrainedModel }) {
  if (!model.metrics) return null;
  const data = Object.entries(model.metrics).map(([k, v]) => ({
    metric: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    value: Math.round(v * 100),
  }));
  if (data.length < 3) return null;
  const color = ALGO_COLORS[model.algorithm] ?? '#3b82f6';
  return (
    <ResponsiveContainer width="100%" height={180}>
      <RadarChart data={data}>
        <PolarGrid stroke="#1e2d47" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#64748b' }} />
        <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
        <Radar dataKey="value" stroke={color} fill={color} fillOpacity={0.15} strokeWidth={1.5} />
        <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e2d47', borderRadius: 8, fontSize: 12 }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

export default function ModelLeaderboard() {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState<number | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [deployMessage, setDeployMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [m, d] = await Promise.allSettled([getModels(), getActiveDeployment()]);
      if (m.status === 'fulfilled') setModels(m.value);
      if (d.status === 'fulfilled') setDeployment(d.value);
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDeploy = async (modelId: number) => {
    setDeploying(modelId);
    setDeployMessage(null);
    try {
      const d = await deployModel(modelId);
      setDeployment(d);
      setDeployMessage({ type: 'success', text: `Model successfully deployed! (Deployment ID: ${d.id})` });
      await load();
    } catch (e: unknown) {
      setDeployMessage({ type: 'error', text: e instanceof Error ? e.message : 'Deploy failed.' });
    } finally { setDeploying(null); }
  };

  const doneModels = models
    .filter(m => m.status === 'DONE')
    .filter(m => !search || m.algorithm.toLowerCase().includes(search.toLowerCase()) || m.model_name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const getScore = (m: TrainedModel) => m.metrics ? (m.metrics.f1_score ?? m.metrics.r2_score ?? 0) : 0;
      return getScore(b) - getScore(a);
    });

  const otherModels = models.filter(m => m.status !== 'DONE');

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient-blue">Model Registry</h1>
          <p className="text-sm mt-1" style={{ color: '#64748b' }}>Compare, inspect, and deploy your trained models.</p>
        </div>
        <button onClick={load} disabled={loading} className="btn-secondary" id="btn-refresh-models">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {deployMessage && (
        <div className={`glass-card p-4 flex items-center gap-3 ${deployMessage.type === 'success' ? 'border-emerald-500/30' : 'border-rose-500/30'}`}
          style={{ borderColor: deployMessage.type === 'success' ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)' }}>
          <span>{deployMessage.type === 'success' ? '✓' : '✗'}</span>
          <span className="text-sm" style={{ color: deployMessage.type === 'success' ? '#34d399' : '#fb7185' }}>{deployMessage.text}</span>
        </div>
      )}

      {/* Active Deployment Banner */}
      {deployment && (
        <div className="glass-card p-4 flex items-center gap-4 glow-blue" style={{ borderColor: 'rgba(59,130,246,0.3)' }}>
          <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.25)' }}>
            <Rocket size={16} style={{ color: '#3b82f6' }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold" style={{ color: '#93c5fd' }}>Active Deployment</div>
            <div className="text-xs mt-0.5" style={{ color: '#64748b' }}>
              Model ID {deployment.model_id} · {deployment.prediction_count} predictions served · deployed at {new Date(deployment.deployed_at).toLocaleString()}
            </div>
          </div>
          <span className="badge-success shrink-0">Live</span>
        </div>
      )}

      {/* Search */}
      {doneModels.length > 0 && (
        <div className="relative max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: '#64748b' }} />
          <input
            type="text"
            placeholder="Search models…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input-field pl-9"
            id="model-search"
          />
        </div>
      )}

      {/* Leaderboard */}
      {loading ? (
        <div className="glass-card p-12 flex items-center justify-center gap-3">
          <RefreshCw size={20} className="animate-spin" style={{ color: '#3b82f6' }} />
          <span style={{ color: '#64748b' }}>Loading models…</span>
        </div>
      ) : doneModels.length === 0 ? (
        <div className="glass-card p-12 flex flex-col items-center gap-3">
          <Trophy size={36} style={{ color: '#475569' }} />
          <div className="text-center">
            <div className="font-medium" style={{ color: '#94a3b8' }}>No trained models yet</div>
            <div className="text-sm mt-1" style={{ color: '#64748b' }}>Upload a dataset and start training to populate the leaderboard.</div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {doneModels.map((model, rank) => {
            const score = model.metrics ? (model.metrics.f1_score ?? model.metrics.r2_score ?? 0) : 0;
            const isActive = deployment?.model_id === model.id;
            const isExpanded = expanded === model.id;
            const color = ALGO_COLORS[model.algorithm] ?? '#3b82f6';

            return (
              <div key={model.id}
                className={`glass-card overflow-hidden transition-all duration-300 ${isActive ? 'glow-blue' : ''}`}
                style={isActive ? { borderColor: 'rgba(59,130,246,0.35)' } : {}}>
                <div className="p-4 flex items-center gap-4">
                  {/* Rank */}
                  <div className="w-8 text-center">
                    {rank === 0 ? <span className="text-xl">🥇</span> :
                     rank === 1 ? <span className="text-xl">🥈</span> :
                     rank === 2 ? <span className="text-xl">🥉</span> :
                     <span className="font-bold text-sm" style={{ color: '#64748b' }}>#{rank + 1}</span>}
                  </div>

                  {/* Color dot */}
                  <div className="w-3 h-3 rounded-full shrink-0" style={{ background: color, boxShadow: `0 0 8px ${color}60` }} />

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-sm" style={{ color: '#e2e8f0' }}>{model.algorithm}</span>
                      {statusBadge(model.status)}
                      {isActive && <span className="badge-info">Active</span>}
                    </div>
                    <div className="text-xs mt-0.5 mono" style={{ color: '#64748b' }}>{model.model_name} · v{model.version}</div>
                  </div>

                  {/* Score bar */}
                  <div className="hidden md:flex flex-col items-end gap-1 min-w-[120px]">
                    <span className="mono text-sm font-bold" style={{ color }}>{(score * 100).toFixed(2)}%</span>
                    <div className="progress-bar w-28">
                      <div className="progress-fill" style={{ width: `${score * 100}%`, background: `linear-gradient(90deg, ${color}99, ${color})` }} />
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 shrink-0">
                    {!isActive && (
                      <button
                        id={`btn-deploy-${model.id}`}
                        onClick={() => handleDeploy(model.id)}
                        disabled={deploying === model.id}
                        className="btn-primary text-xs py-1.5 px-3"
                      >
                        {deploying === model.id ? <RefreshCw size={12} className="animate-spin" /> : <Zap size={12} />}
                        Deploy
                      </button>
                    )}
                    <button
                      onClick={() => setExpanded(isExpanded ? null : model.id)}
                      className="btn-secondary text-xs py-1.5 px-2"
                      id={`btn-expand-${model.id}`}
                    >
                      {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>
                </div>

                {/* Expanded metrics */}
                {isExpanded && (
                  <div className="border-t px-4 py-4 flex gap-6 fade-in" style={{ borderColor: '#1e2d47' }}>
                    <div className="flex-1">
                      <div className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: '#64748b' }}>Metrics</div>
                      <div className="space-y-2">
                        {model.metrics && Object.entries(model.metrics).map(([k, v]) => (
                          <div key={k} className="flex items-center gap-3">
                            <span className="text-sm w-36 capitalize" style={{ color: '#94a3b8' }}>{k.replace(/_/g, ' ')}</span>
                            <div className="progress-bar flex-1">
                              <div className="progress-fill" style={{ width: `${v * 100}%`, background: color }} />
                            </div>
                            <span className="mono text-xs font-bold w-12 text-right" style={{ color }}>{(v * 100).toFixed(2)}%</span>
                          </div>
                        ))}
                      </div>
                      {model.mlflow_run_id && (
                        <div className="mt-3 text-xs mono" style={{ color: '#475569' }}>
                          MLflow Run: {model.mlflow_run_id.slice(0, 16)}…
                        </div>
                      )}
                      <div className="mt-2 text-xs" style={{ color: '#475569' }}>
                        Trained: {new Date(model.created_at).toLocaleString()}
                      </div>
                    </div>
                    {model.metrics && Object.keys(model.metrics).length >= 3 && (
                      <div className="w-52 shrink-0">
                        <MetricRadar model={model} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* In-progress / other models */}
      {otherModels.length > 0 && (
        <div className="glass-card p-5">
          <div className="section-title">Training Jobs</div>
          <div className="space-y-2 mt-3">
            {otherModels.map(m => (
              <div key={m.id} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                {statusBadge(m.status)}
                <span className="text-sm font-medium" style={{ color: '#94a3b8' }}>{m.algorithm}</span>
                <span className="mono text-xs" style={{ color: '#475569' }}>{m.model_name}</span>
                {m.status_message && <span className="text-xs ml-auto" style={{ color: '#64748b' }}>{m.status_message}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
