import type {
  Dataset,
  DatasetDetail,
  Health,
  InferenceResult,
  InferenceStatus,
  Run,
  RunArtifacts,
  RunConfig,
  RunEvent,
  SnapshotResult,
} from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
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
  freezeDataset: (datasetId: string) =>
    request<Dataset>(`/api/datasets/${datasetId}/freeze`, { method: 'POST' }),
  createSnapshot: (datasetId: string) =>
    jsonRequest<SnapshotResult>(`/api/datasets/${datasetId}/training-snapshot`, 'POST', {
      seed: 42,
      val_ratio: 0.2,
    }),
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
  detect: (file: File) => {
    const form = new FormData()
    form.set('file', file)
    return request<InferenceResult>('/api/inference/detect', { method: 'POST', body: form })
  },
}
