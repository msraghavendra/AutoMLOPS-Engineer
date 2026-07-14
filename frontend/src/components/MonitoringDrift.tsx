'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Activity, AlertTriangle, CheckCircle, RefreshCw, ShieldAlert,
  TrendingUp, TrendingDown, Cpu, Wifi, BarChart2, Info
} from 'lucide-react';
import { checkDrift, updateThresholds, type DriftResult } from '@/lib/api';

export default function MonitoringDrift() {
  const [drift, setDrift] = useState<DriftResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [driftThreshold, setDriftThreshold] = useState(0.3);
  const [perfThreshold, setPerfThreshold] = useState(0.7);
  const [savingThresholds, setSavingThresholds] = useState(false);
  const [lastChecked, setLastChecked] = useState<string | null>(null);

  const runDriftCheck = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await checkDrift();
      setDrift(result);
      setLastChecked(new Date().toLocaleTimeString());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Drift check failed.');
    } finally {
      setLoading(false);
    }
  }, []);

  const saveThresholds = async () => {
    setSavingThresholds(true);
    try {
      await updateThresholds(driftThreshold, perfThreshold);
    } catch {/* ignore */} finally {
      setSavingThresholds(false);
    }
  };

  useEffect(() => { runDriftCheck(); }, [runDriftCheck]);

  const featureEntries = drift?.metrics ? Object.entries(drift.metrics) : [];
  const driftedCount = featureEntries.filter(([, v]) => v.drift_detected).length;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient-blue">Monitoring & Drift</h1>
          <p className="text-sm mt-1" style={{ color: '#64748b' }}>
            Real-time data drift analysis across deployed model features.
          </p>
        </div>
        <button
          id="btn-run-drift-check"
          onClick={runDriftCheck}
          className="btn-primary"
          disabled={loading}
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          {loading ? 'Checking…' : 'Run Drift Check'}
        </button>
      </div>

      {/* Status Banner */}
      {drift && (
        <div
          className="flex items-center gap-4 px-5 py-4 rounded-xl fade-in"
          style={{
            background: drift.has_drift ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)',
            border: `1px solid ${drift.has_drift ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)'}`,
          }}
        >
          {drift.has_drift
            ? <ShieldAlert size={24} style={{ color: '#ef4444' }} />
            : <CheckCircle size={24} style={{ color: '#10b981' }} />
          }
          <div className="flex-1">
            <div className="font-semibold" style={{ color: drift.has_drift ? '#ef4444' : '#10b981' }}>
              {drift.has_drift ? 'Drift Detected!' : 'No Drift Detected'}
            </div>
            <div className="text-sm" style={{ color: '#64748b' }}>{drift.message}</div>
          </div>
          <div className="text-right">
            <div className="text-xs" style={{ color: '#475569' }}>Drift Score</div>
            <div className="text-2xl font-bold mono" style={{ color: drift.has_drift ? '#ef4444' : '#10b981' }}>
              {(drift.drift_score * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl" style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)' }}>
          <AlertTriangle size={16} style={{ color: '#f43f5e' }} />
          <span className="text-sm" style={{ color: '#fb7185' }}>{error}</span>
        </div>
      )}

      {/* Stats Row */}
      {drift && featureEntries.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Features Checked', value: featureEntries.length, icon: BarChart2, color: '#3b82f6' },
            { label: 'Drifted Features', value: driftedCount, icon: TrendingUp, color: '#ef4444' },
            { label: 'Stable Features', value: featureEntries.length - driftedCount, icon: TrendingDown, color: '#10b981' },
            { label: 'Overall Drift', value: `${(drift.drift_score * 100).toFixed(1)}%`, icon: Activity, color: drift.has_drift ? '#ef4444' : '#10b981' },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="glass-card p-4 flex items-center gap-3">
              <Icon size={18} style={{ color }} className="shrink-0" />
              <div>
                <div className="text-xs" style={{ color: '#64748b' }}>{label}</div>
                <div className="font-bold text-lg mono" style={{ color }}>{value}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Feature Drift Table */}
      {featureEntries.length > 0 && (
        <div className="glass-card p-5 fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="section-title">Feature Drift Analysis</div>
            {lastChecked && (
              <span className="text-xs" style={{ color: '#475569' }}>Last checked: {lastChecked}</span>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  {['Feature', 'Type', 'Test', 'Drift Score', 'Status'].map(h => (
                    <th key={h} className="table-header">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {featureEntries.map(([feature, info]) => (
                  <tr key={feature} className="table-row">
                    <td className="table-cell mono font-medium" style={{ color: '#e2e8f0' }}>{feature}</td>
                    <td className="table-cell">
                      <span className={info.feature_type === 'numerical' ? 'badge-success' : 'badge-warning'}>
                        {info.feature_type}
                      </span>
                    </td>
                    <td className="table-cell text-xs" style={{ color: '#64748b' }}>{info.test_name}</td>
                    <td className="table-cell">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: '#1e2d47', minWidth: '60px' }}>
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.min(100, info.drift_score * 100)}%`,
                              background: info.drift_detected ? '#ef4444' : '#10b981',
                            }}
                          />
                        </div>
                        <span className="mono text-xs font-medium" style={{ color: info.drift_detected ? '#ef4444' : '#10b981' }}>
                          {(info.drift_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="table-cell">
                      {info.drift_detected
                        ? <span className="badge-error flex items-center gap-1 w-fit"><AlertTriangle size={10} />Drift</span>
                        : <span className="badge-success flex items-center gap-1 w-fit"><CheckCircle size={10} />Stable</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* No data state */}
      {!loading && drift && featureEntries.length === 0 && (
        <div className="glass-card p-10 flex flex-col items-center gap-4 text-center fade-in">
          <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}>
            <Info size={24} style={{ color: '#3b82f6' }} />
          </div>
          <div>
            <div className="font-semibold" style={{ color: '#e2e8f0' }}>No Predictions Yet</div>
            <div className="text-sm mt-1 max-w-md" style={{ color: '#64748b' }}>
              Run predictions against the active deployed model first. Drift analysis compares live prediction features against the baseline training dataset.
            </div>
          </div>
        </div>
      )}

      {/* Threshold Settings */}
      <div className="glass-card p-5">
        <div className="section-title mb-4">Alert Thresholds</div>
        <div className="grid md:grid-cols-2 gap-5">
          <div>
            <label className="text-xs font-medium block mb-2" style={{ color: '#94a3b8' }}>
              Drift Threshold <span className="mono" style={{ color: '#3b82f6' }}>{(driftThreshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range" min={0} max={1} step={0.05}
              value={driftThreshold}
              onChange={e => setDriftThreshold(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-xs mt-1" style={{ color: '#475569' }}>
              <span>0%</span><span>100%</span>
            </div>
          </div>
          <div>
            <label className="text-xs font-medium block mb-2" style={{ color: '#94a3b8' }}>
              Performance Threshold <span className="mono" style={{ color: '#8b5cf6' }}>{(perfThreshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range" min={0} max={1} step={0.05}
              value={perfThreshold}
              onChange={e => setPerfThreshold(Number(e.target.value))}
              className="w-full accent-violet-500"
            />
            <div className="flex justify-between text-xs mt-1" style={{ color: '#475569' }}>
              <span>0%</span><span>100%</span>
            </div>
          </div>
        </div>
        <div className="mt-4">
          <button
            id="btn-save-thresholds"
            onClick={saveThresholds}
            className="btn-secondary"
            disabled={savingThresholds}
          >
            <Cpu size={14} />
            {savingThresholds ? 'Saving…' : 'Save Thresholds'}
          </button>
        </div>
      </div>
    </div>
  );
}
