export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'ship_it_ml_secret_api_key_2026';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Request failed: ${res.status}`);
  }
  return res.json();
}

// --- Datasets ---
export async function getDatasets() {
  return request<Dataset[]>('/api/datasets');
}

export async function uploadDataset(file: File, targetColumn?: string) {
  const form = new FormData();
  form.append('file', file);
  if (targetColumn) form.append('target_column', targetColumn);
  const res = await fetch(`${API_BASE}/api/datasets/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<Dataset>;
}

export async function getDatasetPreview(id: number) {
  return request<{ columns: string[]; rows: Record<string, unknown>[]; features_metadata: FeaturesMetadata }>(`/api/datasets/${id}/preview`);
}

// --- Models ---
export async function getModels() {
  return request<TrainedModel[]>('/api/models');
}

export async function triggerTraining(datasetId: number) {
  const form = new FormData();
  form.append('algorithm', 'Random Forest');
  const res = await fetch(`${API_BASE}/api/models/train/${datasetId}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<TrainedModel>;
}

export async function downloadModel(modelId: number, format: 'joblib' | 'pkl', filename: string) {
  const res = await fetch(`${API_BASE}/api/models/download/${modelId}?format=${format}`);
  if (!res.ok) throw new Error(await res.text() || `Download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function uploadCustomModel(file: File, datasetId: number, modelName: string) {
  const form = new FormData();
  form.append('file', file);
  form.append('dataset_id', String(datasetId));
  form.append('model_name', modelName);
  const res = await fetch(`${API_BASE}/api/models/upload-custom`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<TrainedModel>;
}

export async function getModelStatus(modelId: number) {
  return request<TrainedModel>(`/api/models/train/status/${modelId}`);
}

// --- Deployments ---
export async function deployModel(modelId: number) {
  return request<Deployment>(`/api/deployments/deploy/${modelId}`, { method: 'POST' });
}

export async function getActiveDeployment() {
  return request<Deployment>('/api/deployments/active');
}

export async function predict(inputs: Record<string, unknown>) {
  return request<PredictionResult>('/api/deployments/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
    body: JSON.stringify({ inputs }),
  });
}

// --- Monitoring ---
export async function checkDrift() {
  return request<DriftResult>('/api/monitoring/drift');
}

export async function updateThresholds(driftThreshold: number, performanceThreshold: number) {
  return request('/api/monitoring/thresholds', {
    method: 'POST',
    body: JSON.stringify({ drift_threshold: driftThreshold, performance_threshold: performanceThreshold }),
  });
}

// --- Logs ---
export async function getAuditLogs(params?: { event_type?: string; severity?: string; limit?: number }) {
  const qs = new URLSearchParams();
  if (params?.event_type) qs.set('event_type', params.event_type);
  if (params?.severity) qs.set('severity', params.severity);
  if (params?.limit) qs.set('limit', String(params.limit));
  return request<AuditLog[]>(`/api/logs?${qs.toString()}`);
}

// --- Types ---
export interface FeatureMeta {
  type: 'numerical' | 'categorical';
  null_count: number;
  dtype: string;
  stats: Record<string, number>;
  role?: string;
  explanation?: string;
}
export type FeaturesMetadata = Record<string, FeatureMeta>;

export interface Dataset {
  id: number;
  name: string;
  row_count: number;
  column_count: number;
  target_column: string;
  problem_type: string;
  features_metadata: FeaturesMetadata;
  description?: string;
  ai_analysis?: {
    description?: string;
    suggested_target?: string;
    suggested_problem_type?: string;
    column_analysis?: Record<string, { role: string; type: string; explanation: string }>;
  };
  created_at: string;
}

export interface TrainedModel {
  id: number;
  dataset_id: number;
  model_name: string;
  algorithm: string;
  metrics: Record<string, number> | null;
  file_path: string | null;
  version: number;
  mlflow_run_id: string | null;
  created_at: string;
  status: 'PENDING' | 'TRAINING' | 'DONE' | 'FAILED';
  status_message: string | null;
}

export interface Deployment {
  id: number;
  model_id: number;
  status: string;
  deployed_at: string;
  prediction_count: number;
  drift_threshold: number;
  performance_threshold: number;
  model?: TrainedModel;
}

export interface PredictionResult {
  prediction_id: number;
  prediction: string | number;
  raw_prediction: string | number;
  problem_type: string;
}

export interface DriftResult {
  drift_score: number;
  metrics: Record<string, { drift_detected: boolean; drift_score: number; test_name: string; feature_type: string }>;
  has_drift: boolean;
  message: string;
}

export interface AuditLog {
  id: number;
  event_type: string;
  message: string;
  severity: string;
  created_at: string;
}
