import type {
  AnnotationVersion,
  Dataset,
  DatasetDetail,
  Health,
  InferenceResult,
  InferenceStatus,
  ModelActivation,
  ModelVersion,
  OnnxExportResult,
  Run,
  RunArtifacts,
  RunConfig,
  RunEvent,
  SnapshotResult,
  TrainingSnapshot,
} from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const API_TOKEN_KEY = 'defectdock.api-token'

export function setApiToken(token: string): void {
  window.sessionStorage.setItem(API_TOKEN_KEY, token)
}

export function clearApiToken(): void {
  window.sessionStorage.removeItem(API_TOKEN_KEY)
}

export function hasApiToken(): boolean {
  return typeof window !== 'undefined' && Boolean(window.sessionStorage.getItem(API_TOKEN_KEY))
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, { ...init, headers: authenticatedHeaders(init?.headers) })
  } catch {
    throw new ApiError(0, '无法连接 DefectDock API，请确认后端服务已启动。')
  }

  const contentType = response.headers.get('content-type') ?? ''
  const payload: unknown = contentType.includes('application/json')
    ? await response.json()
    : await response.text()
  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(payload, response.status))
  }
  return payload as T
}

async function requestBlob(path: string): Promise<Blob> {
  let response: Response
  try {
    response = await fetch(path, { headers: authenticatedHeaders() })
  } catch {
    throw new ApiError(0, '无法连接 DefectDock API，请确认后端服务已启动。')
  }
  if (!response.ok) {
    throw new ApiError(response.status, `图片加载失败（HTTP ${response.status}）`)
  }
  return response.blob()
}

function authenticatedHeaders(source?: HeadersInit): Headers {
  const headers = new Headers(source)
  const token = typeof window === 'undefined' ? null : window.sessionStorage.getItem(API_TOKEN_KEY)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return headers
}

function errorMessage(payload: unknown, status: number): string {
  if (typeof payload === 'string' && payload.trim()) return payload
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object' && 'message' in detail) {
      return String((detail as { message: unknown }).message)
    }
    return JSON.stringify(detail)
  }
  return `请求失败（HTTP ${status}）`
}

function jsonRequest<T>(path: string, method: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export const api = {
  health: () => request<Health>('/api/health'),
  inferenceStatus: () => request<InferenceStatus>('/api/inference/status'),
  listDatasets: () => request<Dataset[]>('/api/datasets?limit=100'),
  dataset: (datasetId: string) => request<DatasetDetail>(`/api/datasets/${datasetId}`),
  datasetImage: (previewUrl: string) => requestBlob(previewUrl),
  createDataset: (name: string, scene: string, labels: string, files: File[]) => {
    const form = new FormData()
    form.set('name', name)
    form.set('scene', scene)
    form.set('labels', labels)
    files.forEach((file) => form.append('files', file))
    return request<{ dataset: Dataset }>('/api/datasets', { method: 'POST', body: form })
  },
  uploadAnnotations: (datasetId: string, files: File[]) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return request(`/api/datasets/${datasetId}/annotations`, { method: 'POST', body: form })
  },
  autoAnnotate: (datasetId: string) =>
    jsonRequest<{ annotation_version: AnnotationVersion; detection_count: number }>(
      `/api/datasets/${datasetId}/auto-annotations`,
      'POST',
      { confidence: 0.5, max_detections: 100, device: 'auto' },
    ),
  approveAnnotations: (datasetId: string, versionId: string) =>
    request<AnnotationVersion>(
      `/api/datasets/${datasetId}/annotation-versions/${versionId}/approve`,
      { method: 'POST' },
    ),
  createCvatTask: (datasetId: string) =>
    request<{ dataset: Dataset; task_id: number; task_url: string }>(
      `/api/datasets/${datasetId}/cvat-task`,
      { method: 'POST' },
    ),
  syncCvatAnnotations: (datasetId: string) =>
    request<{ dataset: Dataset; annotation_version: AnnotationVersion }>(
      `/api/datasets/${datasetId}/cvat-sync`,
      { method: 'POST' },
    ),
  freezeDataset: (datasetId: string) =>
    request<Dataset>(`/api/datasets/${datasetId}/freeze`, { method: 'POST' }),
  createSnapshot: (datasetId: string) =>
    jsonRequest<SnapshotResult>(`/api/datasets/${datasetId}/training-snapshot`, 'POST', {
      seed: 42,
      val_ratio: 0.2,
    }),
  listSnapshots: (datasetId: string) =>
    request<TrainingSnapshot[]>(`/api/datasets/${datasetId}/training-snapshots`),
  snapshot: (datasetId: string, snapshotId: string) =>
    request<TrainingSnapshot>(
      `/api/datasets/${datasetId}/training-snapshots/${encodeURIComponent(snapshotId)}`,
    ),
  listRuns: () => request<Run[]>('/api/runs?limit=100'),
  run: (runId: string) => request<Run>(`/api/runs/${runId}`),
  runEvents: (runId: string) => request<RunEvent[]>(`/api/runs/${runId}/events`),
  runArtifacts: (runId: string) =>
    request<RunArtifacts>(`/api/runs/${runId}/artifacts`),
  submitRun: (config: RunConfig) => jsonRequest<Run>('/api/runs', 'POST', config),
  cancelRun: (runId: string) =>
    request<Run>(`/api/runs/${runId}/cancel`, { method: 'POST' }),
  activateRun: (runId: string) =>
    request(`/api/runs/${runId}/activate`, { method: 'POST' }),
  listModels: () => request<ModelVersion[]>('/api/models?limit=100'),
  activationHistory: () => request<ModelActivation[]>('/api/models/activation-history?limit=100'),
  activateModel: (modelVersionId: string) =>
    request(`/api/models/${modelVersionId}/activate`, { method: 'POST' }),
  approveModel: (modelVersionId: string) =>
    request<ModelVersion>(`/api/models/${modelVersionId}/approve`, { method: 'POST' }),
  rollbackModel: () => request('/api/models/rollback', { method: 'POST' }),
  exportOnnx: (modelVersionId: string) =>
    jsonRequest<OnnxExportResult>(`/api/models/${modelVersionId}/exports/onnx`, 'POST', {
      opset: 18,
      warmup_runs: 2,
      benchmark_runs: 10,
    }),
  detect: (file: File) => {
    const form = new FormData()
    form.set('file', file)
    return request<InferenceResult>('/api/inference/detect', { method: 'POST', body: form })
  },
}
