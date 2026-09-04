import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { api } from './api'
import { Icon, ImagePicker } from './ui'
import { useDialog } from './ui-utils'
import type {
  Dataset,
  DatasetDetail,
  DatasetImage,
  Health,
  InferenceResult,
  InferenceStatus,
  Run,
  RunArtifacts,
  RunConfig,
  RunEvent,
  SnapshotResult,
} from './types'

const statusText: Record<string, string> = {
  draft: '待标注',
  annotating: '标注中',
  frozen: '已冻结',
  created: '已创建',
  queued: '排队中',
  running: '训练中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

const terminalStatuses = new Set(['succeeded', 'failed', 'cancelled'])

export function Overview({ health, datasets, runs, inference, loading = false }: { health: Health | null; datasets: Dataset[]; runs: Run[]; inference: InferenceStatus | null; loading?: boolean }) {
  const running = runs.filter((run) => !terminalStatuses.has(run.status)).length
  const failed = runs.filter((run) => run.status === 'failed').length
  if (loading) return <section className="summary-grid" aria-label="正在加载运行摘要" aria-busy="true">{[0, 1, 2, 3].map((item) => <article className="skeleton-card" key={item}><span /><strong /><small /></article>)}</section>
  return (
    <section className="summary-grid" aria-label="运行摘要">
      <article><span><Icon name="folder" />数据集</span><strong>{datasets.length}</strong><small>{datasets.filter((item) => item.status === 'frozen').length} 个已冻结</small></article>
      <article><span><Icon name="activity" />训练运行</span><strong>{runs.length}</strong><small>{running} 个正在处理</small></article>
      <article><span><Icon name="layers" />异常运行</span><strong className={failed ? 'danger-text' : ''}>{failed}</strong><small>{failed ? '进入运行详情查看原因' : '目前没有异常运行'}</small></article>
      <article><span><Icon name="scan" />推理服务</span><strong className={`word-value ${inference?.available ? 'is-ready' : 'is-idle'}`}>{inference?.available ? '已就绪' : '待激活'}</strong><small>{health?.training_submission_enabled ? 'GPU 训练栈可用' : '当前为轻量运行时'}</small></article>
    </section>
  )
}

export function DatasetList({ datasets, selectedId, onSelect, onCreate }: { datasets: Dataset[]; selectedId: string | null; onSelect: (id: string) => void; onCreate: () => void }) {
  return (
    <div className="panel list-panel">
      <div className="panel-heading"><strong>数据集</strong><button className="text-button" onClick={onCreate}>＋ 新建</button></div>
      {datasets.length === 0 ? <Empty title="还没有数据集" text="先上传一批代表性图片。" action="创建数据集" onAction={onCreate} /> : datasets.map((dataset) => (
        <button className={`select-row ${selectedId === dataset.dataset_id ? 'selected' : ''}`} key={dataset.dataset_id} onClick={() => onSelect(dataset.dataset_id)}>
          <span className="row-symbol">DS</span>
          <span><strong>{dataset.name}</strong><small>{dataset.image_count} 张 · {dataset.labels.join(' / ')}</small></span>
          <em className={`status-chip ${dataset.status}`}>{statusText[dataset.status]}</em>
        </button>
      ))}
    </div>
  )
}

export function DatasetWorkspace({ detail, busy, onAnnotations, onAutoAnnotations, onApproveAnnotations, onCreateCvat, onSyncCvat, onSnapshot }: { detail: DatasetDetail | null; busy: string | null; onAnnotations: (files: File[]) => void; onAutoAnnotations: () => void; onApproveAnnotations: () => void; onCreateCvat: () => void; onSyncCvat: () => void; onSnapshot: () => void }) {
  if (!detail) return <div className="panel"><Empty title="选择一个数据集" text="查看图片、标注版本和冻结状态。" /></div>
  const version = detail.current_annotation_version
  return (
    <div className="panel detail-panel">
      <div className="detail-title">
        <div><p className="eyebrow">{detail.dataset_id}</p><h3>{detail.name}</h3></div>
        <span className={`status-chip ${detail.status}`}>{statusText[detail.status]}</span>
      </div>
      <div className="metadata-grid">
        <div><span>图片</span><strong>{detail.image_count}</strong></div>
        <div><span>类别</span><strong>{detail.labels.length}</strong></div>
        <div><span>数据量</span><strong>{formatBytes(detail.total_bytes)}</strong></div>
        <div><span>场景</span><strong>{detail.scene}</strong></div>
      </div>
      <div className="version-card">
        <div><span>当前标注版本</span><strong>{version?.annotation_version_id ?? '尚未导入'}</strong></div>
        {version && <small>{version.labeled_count} 已标注 · {version.unlabeled_count} 未标注 · {version.review_status === 'candidate' ? '等待人工批准' : '已批准'} · SHA {version.manifest_sha256.slice(0, 10)}</small>}
      </div>
      <div className="image-strip">
        {detail.images.slice(0, 12).map((image) => <AuthenticatedImage image={image} key={image.image_id} />)}
        {detail.images.length > 12 && <span>＋{detail.images.length - 12}</span>}
      </div>
      <div className="action-bar">
        {detail.status !== 'frozen' && <button disabled={Boolean(busy)} onClick={() => void onAutoAnnotations()}>{busy === 'auto-annotations' ? '自动标注中…' : '使用当前模型自动标注'}</button>}
        {detail.status !== 'frozen' && detail.cvat_task_id === null && <button disabled={Boolean(busy)} onClick={() => void onCreateCvat()}>{busy === 'cvat-create' ? '正在创建…' : '创建 CVAT 任务'}</button>}
        {detail.status !== 'frozen' && detail.cvat_task_id !== null && <button disabled={Boolean(busy)} onClick={() => void onSyncCvat()}>{busy === 'cvat-sync' ? '正在同步…' : '同步 CVAT 标注'}</button>}
        {detail.annotation_url && <a className="button-like" href={detail.annotation_url} target="_blank" rel="noreferrer">打开 CVAT</a>}
        {detail.status !== 'frozen' && <label className={`button-like ${busy === 'annotations' ? 'disabled' : ''}`}>
          {busy === 'annotations' ? '正在导入…' : '上传 YOLO 标注'}
          <input type="file" accept=".txt,text/plain" multiple disabled={Boolean(busy)} onChange={(event) => { const files = Array.from(event.target.files ?? []); event.target.value = ''; void onAnnotations(files) }} />
        </label>}
        {version?.review_status === 'candidate' && <button className="primary" disabled={Boolean(busy)} onClick={() => void onApproveAnnotations()}>{busy === 'approve-annotations' ? '正在批准…' : '批准候选标注'}</button>}
        <button className="primary" disabled={Boolean(busy) || !version || version.review_status !== 'approved'} onClick={() => void onSnapshot()}>{busy === 'snapshot' ? '正在生成…' : detail.status === 'frozen' ? '重新生成训练快照' : '冻结并生成快照'}</button>
      </div>
      {!version && <p className="helper warning">训练前需上传与图片同名的 YOLO `.txt` 标注文件。</p>}
      {version?.review_status === 'candidate' && <p className="helper warning">这是模型生成的候选框。请检查预览；如需修正可上传新的 YOLO 标注，确认无误后再批准。</p>}
    </div>
  )
}

function AuthenticatedImage({ image }: { image: DatasetImage }) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    let createdUrl: string | null = null
    void api.datasetImage(image.preview_url).then((blob) => {
      if (!active) return
      createdUrl = URL.createObjectURL(blob)
      setObjectUrl(createdUrl)
    }).catch(() => setObjectUrl(null))
    return () => {
      active = false
      if (createdUrl) URL.revokeObjectURL(createdUrl)
    }
  }, [image.preview_url])
  return (
    <div className="annotated-preview" style={{ aspectRatio: `${image.width} / ${image.height}` }} title={image.original_name}>
      {objectUrl ? <img src={objectUrl} alt={image.original_name} /> : <span>IMG</span>}
      {objectUrl && image.boxes.length > 0 && <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
        {image.boxes.map((box, index) => <rect key={`${box.class_id}-${index}`} x={box.cx - box.w / 2} y={box.cy - box.h / 2} width={box.w} height={box.h} />)}
      </svg>}
    </div>
  )
}

export function TrainingForm({ snapshot, dataset, enabled, busy, onSubmit }: { snapshot: SnapshotResult | null; dataset: DatasetDetail | null; enabled: boolean; busy: string | null; onSubmit: (config: RunConfig) => void }) {
  const [project, setProject] = useState('defect-project')
  const [epochs, setEpochs] = useState(20)
  const [device, setDevice] = useState('auto')
  const [pretrained, setPretrained] = useState(false)

  useEffect(() => {
    if (dataset) setProject(toProjectSlug(dataset.name))
  }, [dataset])

  const config = useMemo<RunConfig | null>(() => snapshot ? {
    schema_version: 1,
    project,
    task: 'object-detection',
    engine: 'torchvision',
    model: 'fasterrcnn-resnet50-fpn-v2',
    dataset: { path: snapshot.snapshot.data_yaml, version: snapshot.snapshot.snapshot_id },
    train: { epochs, imgsz: 640, batch: 2, device, workers: 0, learning_rate: 0.005, momentum: 0.9, weight_decay: 0.0005, step_size: 8, gamma: 0.1, seed: 42, pretrained, score_threshold: 0.25, iou_threshold: 0.5 },
    output_root: 'outputs',
  } : null, [device, epochs, pretrained, project, snapshot])

  return (
    <div className="panel training-panel">
      <div className="form-grid">
        <label><span>项目标识</span><input value={project} pattern="[A-Za-z0-9][A-Za-z0-9_-]{1,63}" onChange={(event) => setProject(event.target.value)} /></label>
        <label><span>训练轮数</span><input type="number" min="1" max="10000" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
        <label><span>运行设备</span><select value={device} onChange={(event) => setDevice(event.target.value)}><option value="auto">自动选择</option><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option></select></label>
        <label className="toggle-field"><span>预训练权重</span><button type="button" role="switch" aria-checked={pretrained} className={`toggle ${pretrained ? 'on' : ''}`} onClick={() => setPretrained((value) => !value)}><i /></button></label>
      </div>
      <div className="training-contract">
        <div><span>引擎</span><strong>TorchVision / Faster R-CNN</strong></div>
        <div><span>数据版本</span><strong>{snapshot?.snapshot.snapshot_id ?? '等待训练快照'}</strong></div>
        <div><span>数据质量</span><strong className={snapshot?.quality.ok ? 'success-text' : ''}>{snapshot ? snapshot.quality.ok ? '已通过' : '未通过' : '—'}</strong></div>
      </div>
      <div className="submit-row">
        <p>{!enabled ? '当前轻量运行时未安装训练依赖。请使用 GPU 镜像启动。' : !snapshot ? '先在上一步冻结数据并生成训练快照。' : '提交后可离开页面，后台任务会持续运行。'}</p>
        <button className="primary" disabled={!enabled || !config || Boolean(busy) || !/^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$/.test(project)} onClick={() => config && void onSubmit(config)}>{busy === 'submit-run' ? '正在提交…' : '提交训练'}</button>
      </div>
    </div>
  )
}

export function RunList({ runs, selectedId, onSelect }: { runs: Run[]; selectedId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="panel list-panel">
      <div className="panel-heading"><strong>训练运行</strong><small>{runs.length} 条记录</small></div>
      {runs.length === 0 ? <Empty title="暂无训练记录" text="生成快照后即可从工作台提交。" /> : runs.map((run) => (
        <button className={`select-row ${selectedId === run.run_id ? 'selected' : ''}`} key={run.run_id} onClick={() => onSelect(run.run_id)}>
          <span className="row-symbol">RUN</span>
          <span><strong>{run.project}</strong><small>{run.run_id.slice(-16)}</small></span>
          <em className={`status-chip ${run.status}`}>{statusText[run.status]}</em>
        </button>
      ))}
    </div>
  )
}

export function RunDetail({ run, events, artifacts, busy, onCancel, onActivate }: { run: Run | null; events: RunEvent[]; artifacts: RunArtifacts | null; busy: string | null; onCancel: () => void; onActivate: () => void }) {
  if (!run) return <div className="panel"><Empty title="选择一个训练运行" text="查看进度、指标和模型产物。" /></div>
  const latestEpoch = [...events].reverse().find((event) => event.event === 'epoch_end')
  const standard = run.metrics?.standard as Record<string, number> | undefined
  return (
    <div className="panel detail-panel">
      <div className="detail-title"><div><p className="eyebrow">{run.run_id}</p><h3>{run.project}</h3></div><span className={`status-chip ${run.status}`}>{statusText[run.status]}</span></div>
      <div className="metadata-grid run-metrics">
        <div><span>当前轮次</span><strong>{latestEpoch?.epoch ?? '—'}{latestEpoch?.epochs ? ` / ${latestEpoch.epochs}` : ''}</strong></div>
        <div><span>检出率</span><strong>{formatMetric(standard?.recall)}</strong></div>
        <div><span>精确率</span><strong>{formatMetric(standard?.precision)}</strong></div>
        <div><span>设备</span><strong>{events.find((event) => event.event === 'training_started')?.device ?? run.config.train.device}</strong></div>
      </div>
      <div className="artifact-list">
        <div><span>最佳模型</span><code>{artifacts?.best_model ?? '尚未生成'}</code></div>
        <div><span>指标文件</span><code>{artifacts?.metrics ?? '尚未生成'}</code></div>
        <div><span>配置哈希</span><code>{run.config_hash}</code></div>
      </div>
      {run.error && <div className="inline-error"><strong>失败原因</strong><span>{run.error}</span></div>}
      <div className="action-bar align-end">
        {!terminalStatuses.has(run.status) && <button className="danger" disabled={Boolean(busy)} onClick={() => void onCancel()}>{busy === 'cancel-run' ? '正在取消…' : '取消训练'}</button>}
        {run.status === 'succeeded' && <button className="primary" disabled={Boolean(busy) || !artifacts?.best_model} onClick={() => void onActivate()}>{busy === 'activate-run' ? '正在激活…' : '激活最佳模型'}</button>}
      </div>
    </div>
  )
}

export function InferenceWorkspace({ status, busy, perform }: { status: InferenceStatus | null; busy: string | null; perform: (label: string, action: () => Promise<void>) => Promise<unknown> }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResult | null>(null)
  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  const detect = () => file && perform('detect', async () => setResult(await api.detect(file)))
  return (
    <section className="workspace-section" id="inference">
      <SectionTitle eyebrow="MODEL VERIFICATION" title="模型推理" note={status?.available ? `已激活 · ${status.device}` : '等待激活模型'} />
      <div className="panel inference-panel">
        <div className="inference-input">
          <ImagePicker files={file ? [file] : []} preview={preview} disabled={Boolean(busy)} onChange={(files) => { setFile(files[0] ?? null); setResult(null) }} />
          <button className="primary" disabled={!status?.available || !file || Boolean(busy)} onClick={() => void detect()}>{busy === 'detect' ? '正在推理…' : '运行检测'}</button>
        </div>
        <div className="inference-result" aria-busy={busy === 'detect'} aria-live="polite">
          {!result ? <Empty title={status?.available ? '等待检测图片' : '尚未激活模型'} text={status?.available ? '检测结果与耗时将在这里显示。' : '在已完成的训练运行中激活最佳模型。'} /> : <>
            <div className="result-head"><div><span>检测数量</span><strong>{result.detections.length}</strong></div><div><span>GPU/CPU 推理</span><strong>{result.timing_ms.inference.toFixed(1)} ms</strong></div></div>
            <div className="detection-list">{result.detections.slice(0, 8).map((item, index) => <div key={`${item.class_id}-${index}`}><span>{item.class_name}</span><strong>{(item.confidence * 100).toFixed(1)}%</strong></div>)}</div>
            {result.detections.length > 8 && <small>另有 {result.detections.length - 8} 个检测结果未展开</small>}
          </>}
        </div>
      </div>
    </section>
  )
}

export type CreateDatasetInput = { name: string; scene: string; labels: string; files: File[] }

export function CreateDatasetDialog({ busy, onClose, onSubmit, maxBytes = 511 * 1024 * 1024 }: { busy: boolean; onClose: () => void; onSubmit: (input: CreateDatasetInput) => void; maxBytes?: number }) {
  const [name, setName] = useState('')
  const [scene, setScene] = useState('board')
  const [labels, setLabels] = useState('pit,scratch')
  const [files, setFiles] = useState<File[]>([])
  const dialogRef = useDialog(busy, onClose)
  const valid = name.trim().length >= 2 && labels.split(/[,，]/).some((label) => label.trim()) && files.length > 0
  const submit = (event: FormEvent) => { event.preventDefault(); if (valid && !busy) void onSubmit({ name: name.trim(), scene, labels, files }) }
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}>
      <form className="dialog" ref={dialogRef} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="create-dataset-title" aria-busy={busy} onSubmit={submit}>
        <div className="dialog-head"><div><p className="eyebrow">A FRESH START</p><h2 id="create-dataset-title">创建数据集</h2></div><button type="button" className="icon-button" onClick={onClose} disabled={busy} aria-label="关闭"><Icon name="close" /></button></div>
        <p className="dialog-intro">从一组图片开始，为下一次模型训练做好准备。</p>
        <label><span>数据集名称</span><input required minLength={2} maxLength={100} disabled={busy} value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：一号线板材缺陷" /></label>
        <div className="form-grid two"><label><span>场景</span><select value={scene} disabled={busy} onChange={(event) => setScene(event.target.value)}><option value="board">板材</option><option value="surface">表面检测</option><option value="component">零部件</option><option value="custom">其他</option></select></label><label><span>缺陷类别 · 逗号分隔</span><input required disabled={busy} value={labels} onChange={(event) => setLabels(event.target.value)} placeholder="pit,scratch" /></label></div>
        <ImagePicker files={files} onChange={setFiles} multiple disabled={busy} maxBytes={maxBytes} />
        <div className="dialog-actions"><span className="dialog-hint">{busy ? '正在校验与上传，请稍候' : '可按 Esc 关闭'}</span><button type="button" onClick={onClose} disabled={busy}>取消</button><button className="primary" disabled={busy || !valid}>{busy && <span className="spinner" aria-hidden="true" />}{busy ? '正在上传…' : '创建并上传'}</button></div>
      </form>
    </div>
  )
}

export function NetworkAccessDialog({ busy, error, onSubmit }: { busy: boolean; error: string | null; onSubmit: (token: string) => void }) {
  const [token, setToken] = useState('')
  const dialogRef = useDialog(busy)
  const submit = (event: FormEvent) => {
    event.preventDefault()
    void onSubmit(token)
  }
  return (
    <div className="dialog-backdrop" role="presentation">
      <form className="dialog access-dialog" ref={dialogRef} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="access-title" onSubmit={submit}>
        <div className="dialog-head"><div><p className="eyebrow">NETWORK SECURITY</p><h2 id="access-title">连接受保护的工作台</h2></div></div>
        <p>服务正在网络模式下运行。请输入管理员提供的 API Token；凭据只保存在当前浏览器会话中。</p>
        <label><span>API Token</span><input required minLength={32} disabled={busy} type="password" autoComplete="current-password" value={token} onChange={(event) => setToken(event.target.value)} /></label>
        {error && <div className="inline-error" role="alert"><strong>验证失败</strong><span>{error}</span></div>}
        <div className="dialog-actions"><button className="primary" disabled={busy || token.length < 32}>{busy ? '正在验证…' : '连接'}</button></div>
      </form>
    </div>
  )
}

export function SectionTitle({ eyebrow, title, note }: { eyebrow: string; title: string; note: string }) {
  return <div className="section-title"><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div><span>{note}</span></div>
}

export function Feedback({ error, message, onClose }: { error: string | null; message: string | null; onClose: () => void }) {
  const [paused, setPaused] = useState(false)
  const closeRef = useRef(onClose)
  useEffect(() => { closeRef.current = onClose }, [onClose])
  useEffect(() => {
    if (!message || error || paused) return
    const timer = window.setTimeout(() => closeRef.current(), 7000)
    return () => window.clearTimeout(timer)
  }, [error, message, paused])
  const text = error ?? message
  if (!text) return null
  return <div className={`notice ${error ? 'error' : 'success'}`} role={error ? 'alert' : 'status'} onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)} onFocus={() => setPaused(true)} onBlur={() => setPaused(false)}><span className="notice-symbol" aria-hidden="true">{error ? '!' : <Icon name="check" />}</span><div><strong>{error ? '操作未完成' : '操作成功'}</strong><span>{text}</span></div><button onClick={onClose} aria-label="关闭提示"><Icon name="close" /></button></div>
}

function Empty({ title, text, action, onAction }: { title: string; text: string; action?: string; onAction?: () => void }) {
  return <div className="empty"><strong>{title}</strong><span>{text}</span>{action && onAction && <button onClick={onAction}>{action}</button>}</div>
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

function formatMetric(value: number | undefined): string {
  return value === undefined ? '—' : `${(value * 100).toFixed(1)}%`
}

function toProjectSlug(value: string): string {
  const ascii = value.normalize('NFKD').replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')
  return ascii.length >= 2 ? ascii.slice(0, 64) : 'defect-project'
}
