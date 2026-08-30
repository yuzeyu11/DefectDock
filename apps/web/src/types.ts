export type Health = {
  status: string
  service: string
  version: string
  training_submission_enabled: boolean
  dataset_upload_enabled: boolean
  inference_ready: boolean
}

export type AnnotationVersion = {
  annotation_version_id: string
  dataset_id: string
  source: string
  format: string
  manifest_sha256: string
  labeled_count: number
  unlabeled_count: number
  created_at: string
  is_current: boolean
}

export type Dataset = {
  dataset_id: string
  name: string
  scene: string
  labels: string[]
  status: 'draft' | 'annotating' | 'frozen'
  image_count: number
  total_bytes: number
  cvat_task_id: number | null
  annotation_url?: string
  created_at: string
  updated_at: string
}

export type DatasetImage = {
  image_id: string
  original_name: string
  width: number
  height: number
  preview_url: string
  boxes: Array<{
    class_id: number
    cx: number
    cy: number
    w: number
    h: number
  }>
}

export type DatasetDetail = Dataset & {
  current_annotation_version: AnnotationVersion | null
  images: DatasetImage[]
}

export type SnapshotResult = {
  snapshot: {
    snapshot_id: string
    data_yaml: string
    image_count: number
    annotation_version: string
    annotation_manifest_sha256: string
  }
  quality: {
    ok: boolean
    errors: string[]
    warnings: string[]
  }
  stats: Record<string, unknown>
}

export type RunStatus =
  | 'created'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export type Run = {
  run_id: string
  project: string
  task: string
  engine: string
  model: string
  dataset: string
  dataset_version: string
  config_hash: string
  config: RunConfig
  status: RunStatus
  output_dir: string
  metrics: Record<string, unknown> | null
  error: string | null
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
}

export type RunEvent = {
  event: string
  epoch?: number
  epochs?: number
  device?: string
  metrics?: Record<string, unknown>
}

export type RunArtifacts = {
  run_id: string
  best_model: string | null
  last_model: string | null
  metrics: string | null
}

export type RunConfig = {
  schema_version: 1
  project: string
  task: 'object-detection'
  engine: 'torchvision'
  model: 'fasterrcnn-resnet50-fpn-v2'
  dataset: { path: string; version: string }
  train: {
    epochs: number
    imgsz: number
    batch: number
    device: string
    workers: number
    learning_rate: number
    momentum: number
    weight_decay: number
    step_size: number
    gamma: number
    seed: number
    pretrained: boolean
    score_threshold: number
    iou_threshold: number
  }
  output_root: string
}

export type InferenceStatus = {
  configured: boolean
  available: boolean
  loaded: boolean
  model: string | null
  engine: string
  confidence: number
  max_detections: number
  device: string
}

export type Detection = {
  class_id: number
  class_name: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}

export type InferenceResult = {
  filename: string
  width: number
  height: number
  model: string | null
  timing_ms: { inference: number; total: number }
  detections: Detection[]
}
