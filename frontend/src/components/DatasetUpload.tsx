'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, ChevronRight, RefreshCw, X, Target, Layers } from 'lucide-react';
import { uploadDataset, getDatasetPreview, getDatasets, triggerTraining, getModelStatus, type Dataset, type TrainedModel } from '@/lib/api';

type Step = 'idle' | 'uploading' | 'preview' | 'training' | 'done' | 'error';

const PROBLEM_BADGE: Record<string, string> = {
  classification: 'badge-info',
  regression: 'badge-warning',
};

export default function DatasetUpload() {
  const [step, setStep] = useState<Step>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<{ columns: string[]; rows: Record<string, unknown>[] } | null>(null);
  const [model, setModel] = useState<TrainedModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<string>('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadHistory = useCallback(async () => {
    try { setDatasets(await getDatasets()); setShowHistory(true); } catch {}
  }, []);

  const handleFile = async (f: File) => {
    if (!f.name.endsWith('.csv')) { setError('Please upload a CSV file.'); return; }
    setFile(f);
    setError(null);
    setStep('uploading');
    setProgress('Uploading and analyzing dataset…');
    try {
      const ds = await uploadDataset(f);
      setDataset(ds);
      setProgress('Loading data preview…');
      const pv = await getDatasetPreview(ds.id);
      setPreview(pv);
      setStep('preview');
      setProgress('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed.');
      setStep('error');
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleTrain = async () => {
    if (!dataset) return;
    setStep('training');
    setProgress('Submitting training job…');
    try {
      const m = await triggerTraining(dataset.id);
      setModel(m);
      setProgress('Training in progress — polling status…');
      // Poll until done
      let current = m;
      while (current.status === 'PENDING' || current.status === 'TRAINING') {
        await new Promise(r => setTimeout(r, 3000));
        current = await getModelStatus(m.id);
        setModel(current);
      }
      setStep('done');
      setProgress('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Training failed.');
      setStep('error');
    }
  };

  const reset = () => { setStep('idle'); setFile(null); setDataset(null); setPreview(null); setModel(null); setError(null); setProgress(''); };

  // ---- Render Sections ----
  const renderDropzone = () => (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileRef.current?.click()}
      className={`relative flex flex-col items-center justify-center gap-4 cursor-pointer rounded-xl py-16 px-8 transition-all duration-300 ${dragging ? 'glow-blue' : ''}`}
      style={{
        border: `2px dashed ${dragging ? '#3b82f6' : '#1e2d47'}`,
        background: dragging ? 'rgba(59,130,246,0.05)' : 'rgba(255,255,255,0.02)',
      }}
      id="dropzone"
    >
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
        style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}>
        <Upload size={28} style={{ color: '#3b82f6' }} />
      </div>
      <div className="text-center">
        <div className="font-semibold text-lg" style={{ color: '#e2e8f0' }}>Drop your CSV file here</div>
        <div className="text-sm mt-1" style={{ color: '#64748b' }}>or click to browse — CSV files only</div>
      </div>
      <div className="flex gap-3 text-xs" style={{ color: '#475569' }}>
        <span>✓ Auto target detection</span>
        <span>✓ Feature analysis</span>
        <span>✓ Type inference</span>
      </div>
      <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
    </div>
  );

  const renderUploading = () => (
    <div className="flex flex-col items-center gap-6 py-20">
      <div className="w-14 h-14 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
      <div>
        <div className="text-center font-medium" style={{ color: '#e2e8f0' }}>{progress}</div>
        <div className="text-center text-sm mt-1" style={{ color: '#64748b' }}>{file?.name}</div>
      </div>
    </div>
  );

  const renderPreview = () => {
    if (!dataset || !preview) return null;
    const featMeta = dataset.features_metadata ?? {};
    const numCols = Object.values(featMeta).filter(v => v.type === 'numerical').length;
    const catCols = Object.values(featMeta).filter(v => v.type === 'categorical').length;
    const nullTotal = Object.values(featMeta).reduce((a, v) => a + (v.null_count ?? 0), 0);
    const aiDescription = dataset.description;

    return (
      <div className="space-y-5 fade-in">
        {/* AI Description Banner */}
        {aiDescription && (
          <div className="flex gap-3 px-4 py-4 rounded-xl fade-in" style={{ background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.2)' }}>
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: 'rgba(59,130,246,0.15)' }}>
              <Target size={15} style={{ color: '#3b82f6' }} />
            </div>
            <div>
              <div className="text-xs font-semibold mb-1" style={{ color: '#3b82f6' }}>AI Dataset Profile</div>
              <div className="text-sm" style={{ color: '#94a3b8' }}>{aiDescription}</div>
            </div>
          </div>
        )}
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Rows', value: dataset.row_count.toLocaleString(), icon: Layers },
            { label: 'Features', value: dataset.column_count - 1, icon: FileText },
            { label: 'Null Values', value: nullTotal, icon: AlertCircle },
            { label: 'Problem Type', value: dataset.problem_type, icon: Target },
          ].map(({ label, value, icon: Icon }) => (
            <div key={label} className="glass-card p-4 flex items-center gap-3">
              <Icon size={16} style={{ color: '#3b82f6' }} className="shrink-0" />
              <div>
                <div className="text-xs" style={{ color: '#64748b' }}>{label}</div>
                <div className="font-semibold text-sm capitalize" style={{ color: '#e2e8f0' }}>{String(value)}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Feature Details */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="section-title">Feature Analysis</div>
              <div className="text-xs" style={{ color: '#64748b' }}>{numCols} numerical · {catCols} categorical · Target: <span className="mono">{dataset.target_column}</span></div>
            </div>
            <span className={PROBLEM_BADGE[dataset.problem_type] ?? 'badge-neutral'}>{dataset.problem_type}</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  {['Column', 'Role', 'Type', 'Null Count', 'Dtype', 'AI Insight'].map(h => <th key={h} className="table-header">{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {Object.entries(featMeta).map(([col, meta]) => {
                  const role = meta.role ?? 'feature';
                  const explanation = meta.explanation ?? '';
                  const roleBadge = role === 'target' ? 'badge-info' : role === 'ignore' ? 'badge-error' : 'badge-success';
                  return (
                    <tr key={col} className="table-row">
                      <td className="table-cell">
                        <div className="flex items-center gap-1.5">
                          <span className="mono font-medium" style={{ color: col === dataset.target_column ? '#3b82f6' : '#e2e8f0' }}>{col}</span>
                          {col === dataset.target_column && <span className="badge-info text-xs">target</span>}
                        </div>
                      </td>
                      <td className="table-cell">
                        <span className={roleBadge}>{role}</span>
                      </td>
                      <td className="table-cell">
                        <span className={meta.type === 'numerical' ? 'badge-success' : 'badge-warning'}>{meta.type}</span>
                      </td>
                      <td className="table-cell">
                        <span style={{ color: meta.null_count > 0 ? '#f59e0b' : '#10b981' }}>{meta.null_count}</span>
                      </td>
                      <td className="table-cell mono text-xs">{meta.dtype}</td>
                      <td className="table-cell text-xs" style={{ color: '#64748b', maxWidth: '200px' }}>{explanation}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Data Preview */}
        <div className="glass-card p-5">
          <div className="section-title">Data Preview <span className="text-sm font-normal" style={{ color: '#64748b' }}>(first 10 rows)</span></div>
          <div className="overflow-x-auto mt-3">
            <table className="w-full text-sm">
              <thead>
                <tr>{preview.columns.map(c => <th key={c} className="table-header mono">{c}</th>)}</tr>
              </thead>
              <tbody>
                {preview.rows.slice(0, 10).map((row, i) => (
                  <tr key={i} className="table-row">
                    {preview.columns.map(c => (
                      <td key={c} className="table-cell mono text-xs">{String(row[c] ?? '—')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button id="btn-start-training" onClick={handleTrain} className="btn-primary">
            <ChevronRight size={16} />
            Start AutoML Training
          </button>
          <button onClick={reset} className="btn-secondary">
            <X size={14} />
            Upload Different File
          </button>
        </div>
      </div>
    );
  };

  const renderTraining = () => (
    <div className="flex flex-col items-center gap-6 py-12 fade-in">
      <div className="relative w-20 h-20">
        <div className="w-20 h-20 rounded-full border-2 border-violet-500/30 border-t-violet-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(139,92,246,0.15)' }}>
            <RefreshCw size={14} style={{ color: '#8b5cf6' }} />
          </div>
        </div>
      </div>
      <div className="text-center">
        <div className="font-semibold text-lg" style={{ color: '#e2e8f0' }}>AutoML Training in Progress</div>
        <div className="text-sm mt-1" style={{ color: '#64748b' }}>{progress}</div>
        {model && (
          <div className="mt-3">
            <span className={model.status === 'TRAINING' ? 'badge-warning' : model.status === 'PENDING' ? 'badge-neutral' : 'badge-info'}>
              {model.status} — {model.algorithm}
            </span>
          </div>
        )}
      </div>
      <div className="glass-card p-4 max-w-sm w-full text-center">
        <div className="text-xs" style={{ color: '#64748b' }}>Training 4+ algorithms with hyperparameter tuning via RandomizedSearchCV. This may take a few minutes.</div>
      </div>
    </div>
  );

  const renderDone = () => (
    <div className="flex flex-col items-center gap-5 py-12 fade-in">
      <div className="w-16 h-16 rounded-full flex items-center justify-center glow-emerald"
        style={{ background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.3)' }}>
        <CheckCircle size={32} style={{ color: '#10b981' }} />
      </div>
      <div className="text-center">
        <div className="font-bold text-xl" style={{ color: '#e2e8f0' }}>Training Complete!</div>
        <div className="text-sm mt-1" style={{ color: '#64748b' }}>Head to the <strong>Model Registry</strong> tab to see results and deploy your model.</div>
      </div>
      {model?.metrics && (
        <div className="glass-card p-5 max-w-sm w-full">
          <div className="font-medium mb-3 text-sm" style={{ color: '#94a3b8' }}>{model.model_name} — v{model.version}</div>
          <div className="space-y-2">
            {Object.entries(model.metrics).map(([k, v]) => (
              <div key={k} className="flex justify-between text-sm">
                <span className="capitalize" style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}</span>
                <span className="mono font-semibold" style={{ color: '#10b981' }}>{(v * 100).toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="flex gap-3">
        <button onClick={reset} className="btn-primary" id="btn-upload-another">
          <Upload size={14} />
          Upload Another Dataset
        </button>
      </div>
    </div>
  );

  const renderError = () => (
    <div className="flex flex-col items-center gap-4 py-12 fade-in">
      <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: 'rgba(244,63,94,0.12)', border: '1px solid rgba(244,63,94,0.2)' }}>
        <AlertCircle size={28} style={{ color: '#f43f5e' }} />
      </div>
      <div className="text-center">
        <div className="font-semibold" style={{ color: '#fb7185' }}>Something went wrong</div>
        <div className="text-sm mt-1 max-w-md" style={{ color: '#64748b' }}>{error}</div>
      </div>
      <button onClick={reset} className="btn-secondary">Try Again</button>
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient-blue">Dataset Upload</h1>
          <p className="text-sm mt-1" style={{ color: '#64748b' }}>Upload a CSV → auto-detect features → train all suitable models.</p>
        </div>
        {step === 'idle' && (
          <button onClick={loadHistory} className="btn-secondary" id="btn-show-history">
            <FileText size={14} />
            View History
          </button>
        )}
      </div>

      <div className="glass-card p-6">
        {step === 'idle' && renderDropzone()}
        {step === 'uploading' && renderUploading()}
        {step === 'preview' && renderPreview()}
        {step === 'training' && renderTraining()}
        {step === 'done' && renderDone()}
        {step === 'error' && renderError()}
      </div>

      {showHistory && datasets.length > 0 && (
        <div className="glass-card p-5 fade-in">
          <div className="section-title">Previously Uploaded Datasets</div>
          <table className="w-full mt-3">
            <thead>
              <tr>
                {['Name', 'Rows', 'Columns', 'Problem', 'Target', 'Uploaded'].map(h => <th key={h} className="table-header">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {datasets.map(d => (
                <tr key={d.id} className="table-row">
                  <td className="table-cell font-medium">{d.name}</td>
                  <td className="table-cell">{d.row_count.toLocaleString()}</td>
                  <td className="table-cell">{d.column_count}</td>
                  <td className="table-cell"><span className={PROBLEM_BADGE[d.problem_type] ?? 'badge-neutral'}>{d.problem_type}</span></td>
                  <td className="table-cell mono text-xs">{d.target_column}</td>
                  <td className="table-cell text-xs" style={{ color: '#64748b' }}>{new Date(d.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
