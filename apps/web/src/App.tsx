import { useCallback, useEffect, useState } from 'react'
import { ApiError, api } from './api'
import {
  CreateDatasetDialog,
  DatasetList,
  DatasetWorkspace,
  Feedback,
  InferenceWorkspace,
  Overview,
  RunDetail,
  RunList,
  SectionTitle,
  TrainingForm,
  type CreateDatasetInput,
} from './components'
import type {
  Dataset,
  DatasetDetail,
  Health,
  InferenceStatus,
  Run,
  RunArtifacts,
  RunConfig,
  RunEvent,
  SnapshotResult,
} from './types'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [inference, setInference] = useState<InferenceStatus | null>(null)
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(null)
  const [snapshot, setSnapshot] = useState<SnapshotResult | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedRun, setSelectedRun] = useState<Run | null>(null)
  const [runEvents, setRunEvents] = useState<RunEvent[]>([])
  const [artifacts, setArtifacts] = useState<RunArtifacts | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const refreshOverview = useCallback(async (quiet = false) => {
    try {
      const [nextHealth, nextDatasets, nextRuns, nextInference] = await Promise.all([
        api.health(),
        api.listDatasets(),
        api.listRuns(),
        api.inferenceStatus(),
      ])
      setHealth(nextHealth)
      setDatasets(nextDatasets)
      setRuns(nextRuns)
      setInference(nextInference)
      setSelectedDatasetId((current) => current ?? nextDatasets[0]?.dataset_id ?? null)
      setSelectedRunId((current) => current ?? nextRuns[0]?.run_id ?? null)
      if (!quiet) setError(null)
    } catch (cause) {
      if (!quiet) setError(toMessage(cause))
    }
  }, [])

  const refreshDataset = useCallback(async (datasetId: string) => {
    try {
      setDatasetDetail(await api.dataset(datasetId))
    } catch (cause) {
      setError(toMessage(cause))
    }
  }, [])

  const refreshRun = useCallback(async (runId: string) => {
    try {
      const [record, events, runArtifacts] = await Promise.all([
        api.run(runId),
        api.runEvents(runId),
        api.runArtifacts(runId),
      ])
      setSelectedRun(record)
      setRunEvents(events)
      setArtifacts(runArtifacts)
    } catch (cause) {
      setError(toMessage(cause))
    }
  }, [])

  useEffect(() => {
    void refreshOverview()
    const timer = window.setInterval(() => void refreshOverview(true), 5000)
    return () => window.clearInterval(timer)
  }, [refreshOverview])

  useEffect(() => {
    if (selectedDatasetId) void refreshDataset(selectedDatasetId)
  }, [refreshDataset, selectedDatasetId])

  useEffect(() => {
    if (!selectedRunId) return
    void refreshRun(selectedRunId)
    const timer = window.setInterval(() => void refreshRun(selectedRunId), 3000)
    return () => window.clearInterval(timer)
  }, [refreshRun, selectedRunId])

  const perform = async (label: string, action: () => Promise<void>) => {
    setBusy(label)
    setError(null)
    setMessage(null)
    try {
      await action()
    } catch (cause) {
      setError(toMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const createDataset = async (input: CreateDatasetInput) => {
    await perform('create-dataset', async () => {
      const result = await api.createDataset(input.name, input.scene, input.labels, input.files)
      setShowCreate(false)
      setSelectedDatasetId(result.dataset.dataset_id)
      setSnapshot(null)
      await refreshOverview(true)
      await refreshDataset(result.dataset.dataset_id)
      setMessage(`数据集“${result.dataset.name}”已创建，可继续上传标注。`)
    })
  }

  const uploadAnnotations = async (files: File[]) => {
    if (!selectedDatasetId || files.length === 0) return
    await perform('annotations', async () => {
      await api.uploadAnnotations(selectedDatasetId, files)
      await refreshDataset(selectedDatasetId)
      setMessage(`已导入 ${files.length} 个标注文件并建立新版本。`)
    })
  }

  const freezeAndSnapshot = async () => {
    if (!selectedDatasetId) return
    await perform('snapshot', async () => {
      if (datasetDetail?.status !== 'frozen') await api.freezeDataset(selectedDatasetId)
      const result = await api.createSnapshot(selectedDatasetId)
      setSnapshot(result)
      await refreshOverview(true)
      await refreshDataset(selectedDatasetId)
      setMessage('不可变训练快照已生成，可提交训练。')
      document.querySelector('#training')?.scrollIntoView({ behavior: 'smooth' })
    })
  }

  const submitTraining = async (config: RunConfig) => {
    await perform('submit-run', async () => {
      const record = await api.submitRun(config)
      setSelectedRunId(record.run_id)
      await refreshOverview(true)
      setMessage(`训练 ${record.run_id} 已进入队列。`)
      document.querySelector('#runs')?.scrollIntoView({ behavior: 'smooth' })
    })
  }

  const cancelRun = async () => {
    if (!selectedRunId) return
    await perform('cancel-run', async () => {
      await api.cancelRun(selectedRunId)
      await refreshRun(selectedRunId)
      await refreshOverview(true)
      setMessage('已发送取消请求。')
    })
  }

  const activateRun = async () => {
    if (!selectedRunId) return
    await perform('activate-run', async () => {
      await api.activateRun(selectedRunId)
      await refreshOverview(true)
      setMessage('模型已激活，推理工作区现在可以使用。')
      document.querySelector('#inference')?.scrollIntoView({ behavior: 'smooth' })
    })
  }

  const completedStages = [
    datasets.length > 0,
    Boolean(datasetDetail?.current_annotation_version),
    runs.some((run) => run.status === 'succeeded'),
    Boolean(inference?.available),
  ].filter(Boolean).length

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="DefectDock 总览">
          <div className="brand-mark">D</div>
          <div><strong>DefectDock</strong><span>VISION OPERATIONS</span></div>
        </a>
        <nav aria-label="主导航">
          <a href="#overview"><span>01</span>运行总览</a>
          <a href="#datasets"><span>02</span>数据与标注</a>
          <a href="#training"><span>03</span>训练配置</a>
          <a href="#runs"><span>04</span>运行与产物</a>
          <a href="#inference"><span>05</span>模型推理</a>
        </nav>
        <div className="sidebar-footer">
          <span className={`status-dot ${health ? 'online' : ''}`} />
          <div><strong>{health ? '服务正常' : '等待连接'}</strong><small>API {health?.version ?? '—'}</small></div>
        </div>
      </aside>

      <main>
        <header>
          <div><p className="eyebrow">INDUSTRIAL VISION WORKBENCH</p><h1>交付工作台</h1></div>
          <div className="header-actions">
            <button className="ghost" onClick={() => window.open('/docs', '_blank')}>接口文档</button>
            <button className="primary" onClick={() => setShowCreate(true)} disabled={!health?.dataset_upload_enabled}>创建数据集</button>
          </div>
        </header>

        <Feedback error={error} message={message} onClose={() => { setError(null); setMessage(null) }} />

        <section className="hero" id="overview">
          <div>
            <span className="pill">PRODUCT SLICE · P1-1</span>
            <h2>一条链路，完成数据到模型。</h2>
            <p>上传图片与标注，冻结可追溯数据版本，提交训练并查看产物，最后激活模型完成推理验证。</p>
          </div>
          <div className="hero-metric">
            <span>当前闭环进度</span><strong>{completedStages} / 4</strong><small>按实际工作区状态计算</small>
            <div className="metric-line"><i style={{ width: `${completedStages * 25}%` }} /></div>
          </div>
        </section>

        <Overview health={health} datasets={datasets} runs={runs} inference={inference} />

        <section className="workspace-section" id="datasets">
          <SectionTitle eyebrow="DATA GOVERNANCE" title="数据与标注" note="上传 → 标注版本 → 冻结快照" />
          <div className="split-layout">
            <DatasetList datasets={datasets} selectedId={selectedDatasetId} onSelect={(id) => { setSelectedDatasetId(id); setSnapshot(null) }} onCreate={() => setShowCreate(true)} />
            <DatasetWorkspace detail={datasetDetail} busy={busy} onAnnotations={uploadAnnotations} onSnapshot={freezeAndSnapshot} />
          </div>
        </section>

        <section className="workspace-section" id="training">
          <SectionTitle eyebrow="REPRODUCIBLE TRAINING" title="训练配置" note="配置会随运行固化并计算哈希" />
          <TrainingForm snapshot={snapshot} dataset={datasetDetail} enabled={Boolean(health?.training_submission_enabled)} busy={busy} onSubmit={submitTraining} />
        </section>

        <section className="workspace-section" id="runs">
          <SectionTitle eyebrow="RUN OPERATIONS" title="运行与产物" note="状态每 3 秒自动刷新" />
          <div className="split-layout runs-layout">
            <RunList runs={runs} selectedId={selectedRunId} onSelect={setSelectedRunId} />
            <RunDetail run={selectedRun} events={runEvents} artifacts={artifacts} busy={busy} onCancel={cancelRun} onActivate={activateRun} />
          </div>
        </section>

        <InferenceWorkspace status={inference} busy={busy} perform={perform} />
      </main>

      {showCreate && <CreateDatasetDialog busy={busy === 'create-dataset'} onClose={() => setShowCreate(false)} onSubmit={createDataset} />}
    </div>
  )
}

function toMessage(cause: unknown): string {
  if (cause instanceof ApiError || cause instanceof Error) return cause.message
  return '操作失败，请查看后端日志后重试。'
}

export default App
