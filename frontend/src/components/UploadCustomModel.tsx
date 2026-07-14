'use client';

import { useState, useCallback, useRef } from 'react';
import {
  PackagePlus, Upload, Database, CheckCircle, AlertCircle,
  RefreshCw, X, ChevronRight, FileType2, Cpu, ArrowRight
} from 'lucide-react';
import {
  getDatasets, uploadCustomModel, getModelStatus,
  type Dataset, type TrainedModel
} from '@/lib/api';

type Step = 'idle' | 'loading' | 'configure' | 'retraining' | 'done' | 'error';

const ROLE_COLORS: Record<string, string> = {
  target: '#3b82f6',
  feature: '#10b981',
  ignore: '#ef4444',
};
const ROLE_BADGES: Record<string, string> = {
  target: 'badge-info',
  feature: 'badge-success',
  ignore: 'badge-error',
};

export default function UploadCustomModel() {
  const [step, setStep] = useState<Step>('idle');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState('');
  const [model, setModel] = useState<TrainedModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const loadDatasets = useCallback(async () => {
    setStep('loading');
    setError(null);
    try {
      const ds = await getDatasets();
      setDatasets(ds);
      setStep('configure');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load datasets.');
      setStep('error');
    }
  }, []);

  const handleFile = (f: File) => {
    if (!f.name.endsWith('.joblib')) {
      setError('Only .joblib model files are supported.');
      return;
    }
    setModelFile(f);
    setError(null);
    if (!modelName) setModelName(f.name.replace('.joblib', ''));
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleSubmit = async () => {
    if (!modelFile || !selectedDataset) return;
    setStep('retraining');
    setProgress('Submitting retraining job…');
    try {
      const m = await uploadCustomModel(modelFile, selectedDataset.id, modelName || modelFile.name);
      setModel(m);
      setProgress('Retraining in progress — polling status…');
      let current = m;
      while (current.status === 'PENDING' || current.status === 'TRAINING') {
        await new Promise(r => setTimeout(r, 3000));
        current = await getModelStatus(m.id);
        setModel(current);
      }
      setStep('done');
      setProgress('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Retraining failed.');
      setStep('error');
    }
  };

  const reset = () => {
    setStep('idle');
    setDatasets([]);
    setSelectedDataset(null);
    setModelFile(null);
    setModelName('');
    setModel(null);
    setError(null);
    setProgress('');
  };

  // ---- Renders ----

  const renderIdle = () => (
    <div className="flex flex-col items-center gap-8 py-16">
      <div className="w-20 h-20 rounded-2xl flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2))', border: '1px solid rgba(139,92,246,0.3)' }}>
        <PackagePlus size={36} style={{ color: '#8b5cf6' }} />
      </div>
      <div className="text-center">
        <h2 className="text-xl font-bold mb-2" style={{ color: '#e2e8f0' }}>Upload & Retrain a Custom Model</h2>
        <p className="text-sm max-w-md" style={{ color: '#64748b' }}>
          Upload an existing scikit-learn <span className="mono">.joblib</span> pipeline and retrain it on one of your uploaded datasets. The best model is auto-evaluated and can be promoted to active deployment.
        </p>
      </div>
      <div className="grid grid-cols-3 gap-4 max-w-lg w-full text-center">
        {[
          { icon: Upload, label: 'Upload .joblib', color: '#8b5cf6' },
          { icon: Database, label: 'Select Dataset', color: '#3b82f6' },
          { icon: Cpu, label: 'Auto-Retrain', color: '#10b981' },
        ].map(({ icon: Icon, label, color }) => (
          <div key={label} className="glass-card p-4 flex flex-col items-center gap-2">
            <Icon size={20} style={{ color }} />
            <span className="text-xs font-medium" style={{ color: '#94a3b8' }}>{label}</span>
          </div>
        ))}
      </div>
      <button id="btn-start-custom" onClick={loadDatasets} className="btn-primary">
        <ArrowRight size={16} />
        Get Started
      </button>
    </div>
  );

  const renderLoading = () => (
    <div className="flex flex-col items-center gap-6 py-20">
      <div className="w-12 h-12 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
      <div className="text-sm" style={{ color: '#94a3b8' }}>Loading available datasets…</div>
    </div>
  );

  const renderConfigure = () => (
    <div className="space-y-6 fade-in">
      {/* Step 1: Pick dataset */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: 'rgba(59,130,246,0.2)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.3)' }}>1</div>
          <span className="font-semibold text-sm" style={{ color: '#e2e8f0' }}>Select a Dataset to Train On</span>
        </div>
        {datasets.length === 0 ? (
          <div className="text-sm text-center py-8" style={{ color: '#64748b' }}>No datasets uploaded yet. Please upload a CSV first.</div>
        ) : (
          <div className="grid gap-2">
            {datasets.map(ds => (
              <button
                key={ds.id}
                id={`select-dataset-${ds.id}`}
                onClick={() => setSelectedDataset(ds)}
                className={`w-full text-left p-4 rounded-xl transition-all duration-200 flex items-center justify-between ${selectedDataset?.id === ds.id ? 'glow-blue' : ''}`}
                style={{
                  background: selectedDataset?.id === ds.id ? 'rgba(59,130,246,0.1)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${selectedDataset?.id === ds.id ? '#3b82f6' : '#1e2d47'}`,
                }}
              >
                <div className="flex items-center gap-3">
                  <Database size={16} style={{ color: selectedDataset?.id === ds.id ? '#3b82f6' : '#475569' }} />
                  <div>
                    <div className="font-medium text-sm" style={{ color: '#e2e8f0' }}>{ds.name}</div>
                    <div className="text-xs mt-0.5" style={{ color: '#64748b' }}>
                      {ds.row_count.toLocaleString()} rows · Target: <span className="mono">{ds.target_column}</span>
                    </div>
                  </div>
                </div>
                <span className={`text-xs px-2 py-1 rounded-full font-medium ${ds.problem_type === 'classification' ? 'badge-info' : 'badge-warning'}`}>
                  {ds.problem_type}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Step 2: Drop .joblib */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: 'rgba(139,92,246,0.2)', color: '#8b5cf6', border: '1px solid rgba(139,92,246,0.3)' }}>2</div>
          <span className="font-semibold text-sm" style={{ color: '#e2e8f0' }}>Upload Your .joblib Model File</span>
        </div>
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          id="custom-model-dropzone"
          className={`relative flex flex-col items-center justify-center gap-4 cursor-pointer rounded-xl py-10 px-8 transition-all duration-300`}
          style={{
            border: `2px dashed ${dragging ? '#8b5cf6' : modelFile ? '#10b981' : '#1e2d47'}`,
            background: dragging ? 'rgba(139,92,246,0.05)' : modelFile ? 'rgba(16,185,129,0.04)' : 'rgba(255,255,255,0.02)',
          }}
        >
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
            style={{ background: modelFile ? 'rgba(16,185,129,0.1)' : 'rgba(139,92,246,0.1)', border: `1px solid ${modelFile ? 'rgba(16,185,129,0.2)' : 'rgba(139,92,246,0.2)'}` }}>
            {modelFile ? <CheckCircle size={24} style={{ color: '#10b981' }} /> : <FileType2 size={24} style={{ color: '#8b5cf6' }} />}
          </div>
          <div className="text-center">
            {modelFile ? (
              <>
                <div className="font-semibold text-sm" style={{ color: '#10b981' }}>Model file selected</div>
                <div className="text-xs mt-1 mono" style={{ color: '#64748b' }}>{modelFile.name} · {(modelFile.size / 1024).toFixed(1)} KB</div>
              </>
            ) : (
              <>
                <div className="font-semibold text-sm" style={{ color: '#e2e8f0' }}>Drop your .joblib file here</div>
                <div className="text-xs mt-1" style={{ color: '#64748b' }}>or click to browse — scikit-learn pipelines supported</div>
              </>
            )}
          </div>
          <input ref={fileRef} type="file" accept=".joblib" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        </div>
      </div>

      {/* Step 3: Model Name */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: 'rgba(16,185,129,0.2)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }}>3</div>
          <span className="font-semibold text-sm" style={{ color: '#e2e8f0' }}>Name Your Model</span>
        </div>
        <input
          id="custom-model-name-input"
          type="text"
          placeholder="e.g. My Custom RF Pipeline"
          value={modelName}
          onChange={e => setModelName(e.target.value)}
          className="w-full px-4 py-3 rounded-xl text-sm transition-all duration-200 outline-none"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid #1e2d47',
            color: '#e2e8f0',
          }}
          onFocus={e => { e.target.style.borderColor = '#3b82f6'; }}
          onBlur={e => { e.target.style.borderColor = '#1e2d47'; }}
        />
      </div>

      {/* Summary + Submit */}
      {selectedDataset && modelFile && (
        <div className="glass-card p-4 flex items-center justify-between fade-in">
          <div className="text-sm" style={{ color: '#94a3b8' }}>
            <span className="mono font-medium" style={{ color: '#e2e8f0' }}>{modelFile.name}</span>
            <span style={{ color: '#475569' }}> → </span>
            <span className="mono font-medium" style={{ color: '#3b82f6' }}>{selectedDataset.name}</span>
          </div>
          <button id="btn-submit-custom" onClick={handleSubmit} className="btn-primary" style={{ padding: '8px 20px' }}>
            <ChevronRight size={14} />
            Retrain
          </button>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl fade-in" style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)' }}>
          <AlertCircle size={16} style={{ color: '#f43f5e' }} />
          <span className="text-sm" style={{ color: '#fb7185' }}>{error}</span>
        </div>
      )}
    </div>
  );

  const renderRetraining = () => (
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
        <div className="font-semibold text-lg" style={{ color: '#e2e8f0' }}>Retraining Custom Model</div>
        <div className="text-sm mt-1" style={{ color: '#64748b' }}>{progress}</div>
        {model && (
          <div className="mt-3">
            <span className={`${model.status === 'TRAINING' ? 'badge-warning' : model.status === 'PENDING' ? 'badge-neutral' : 'badge-info'}`}>
              {model.status} — {model.algorithm}
            </span>
          </div>
        )}
      </div>
      <div className="glass-card p-4 max-w-sm w-full text-center">
        <div className="text-xs" style={{ color: '#64748b' }}>
          Loading your pipeline, fitting on the selected dataset, and evaluating metrics. This may take a few minutes.
        </div>
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
        <div className="font-bold text-xl" style={{ color: '#e2e8f0' }}>Retraining Complete!</div>
        <div className="text-sm mt-1" style={{ color: '#64748b' }}>Head to the <strong>Model Registry</strong> tab to deploy your custom model.</div>
      </div>
      {model?.metrics && (
        <div className="glass-card p-5 max-w-sm w-full">
          <div className="font-medium mb-3 text-sm" style={{ color: '#94a3b8' }}>{model.model_name}</div>
          <div className="space-y-2">
            {Object.entries(model.metrics)
              .filter(([k]) => k !== 'label_mapping')
              .map(([k, v]) => (
                <div key={k} className="flex justify-between text-sm">
                  <span className="capitalize" style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}</span>
                  <span className="mono font-semibold" style={{ color: '#10b981' }}>
                    {typeof v === 'number' ? (Math.abs(v) <= 1 ? `${(v * 100).toFixed(2)}%` : v.toFixed(4)) : String(v)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
      <div className="flex gap-3">
        <button onClick={reset} className="btn-primary" id="btn-upload-another-custom">
          <PackagePlus size={14} />
          Upload Another Model
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
      <div>
        <h1 className="text-2xl font-bold" style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Upload Custom Model
        </h1>
        <p className="text-sm mt-1" style={{ color: '#64748b' }}>
          Upload an existing <span className="mono">.joblib</span> scikit-learn pipeline → select a dataset → auto-retrain & evaluate.
        </p>
      </div>

      <div className="glass-card p-6">
        {step === 'idle' && renderIdle()}
        {step === 'loading' && renderLoading()}
        {step === 'configure' && renderConfigure()}
        {step === 'retraining' && renderRetraining()}
        {step === 'done' && renderDone()}
        {step === 'error' && renderError()}
      </div>

      {/* How-it-works hint when idle */}
      {step === 'idle' && (
        <div className="glass-card p-5 fade-in">
          <div className="section-title mb-3">How Custom Model Retraining Works</div>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            {[
              { step: '1', title: 'Upload Pipeline', desc: 'Provide any scikit-learn Pipeline or raw estimator serialized as .joblib.' },
              { step: '2', title: 'Select Dataset', desc: 'Pick from your already-uploaded CSV datasets. AI-inferred column roles are applied automatically.' },
              { step: '3', title: 'Retrain & Evaluate', desc: 'The engine refits your model, measures F1/R² on a test split, logs to MLflow, and auto-promotes if it beats the current active model.' },
            ].map(({ step, title, desc }) => (
              <div key={step} className="flex gap-3">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 text-xs font-bold"
                  style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', border: '1px solid rgba(139,92,246,0.2)' }}>
                  {step}
                </div>
                <div>
                  <div className="font-semibold text-xs mb-1" style={{ color: '#e2e8f0' }}>{title}</div>
                  <div className="text-xs" style={{ color: '#64748b' }}>{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
