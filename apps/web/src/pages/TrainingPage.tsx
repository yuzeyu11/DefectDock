import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { SectionTitle, TrainingForm } from '../components'
import { toMessage } from '../errors'
import type { DatasetDetail, RunConfig, SnapshotResult } from '../types'
import { useWorkbench } from '../workbench'

type TrainingLocationState = { dataset?: DatasetDetail; snapshot?: SnapshotResult }

export function TrainingPage() {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const navigate = useNavigate()
  const { health, busy, perform, notify } = useWorkbench()
  const datasetId = searchParams.get('datasetId')
  const requestedSnapshotId = searchParams.get('snapshotId')
  const initial = location.state as TrainingLocationState | null
  const [dataset, setDataset] = useState<DatasetDetail | null>(initial?.dataset ?? null)
  const [snapshot, setSnapshot] = useState<SnapshotResult | null>(initial?.snapshot ?? null)
  const [loading, setLoading] = useState(Boolean(datasetId && (!initial?.dataset || !initial?.snapshot)))
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!datasetId || (dataset && snapshot)) return
    let active = true
    const recover = async () => {
      try {
        const [nextDataset, snapshotRecord] = await Promise.all([
          api.dataset(datasetId),
          requestedSnapshotId
            ? api.snapshot(datasetId, requestedSnapshotId)
            : api.listSnapshots(datasetId).then((items) => {
                if (!items[0]) throw new Error('该数据集还没有已登记的训练快照。')
                return items[0]
              }),
        ])
        if (!active) return
        const nextSnapshot: SnapshotResult = {
          snapshot: {
            snapshot_id: snapshotRecord.snapshot_id,
            data_yaml: snapshotRecord.data_yaml,
            image_count: snapshotRecord.image_count,
            annotation_version: snapshotRecord.annotation_version_id,
            annotation_manifest_sha256: '',
          },
          quality: { ok: true, errors: [], warnings: [] },
          stats: {},
        }
        setDataset(nextDataset)
        setSnapshot(nextSnapshot)
      } catch (cause) {
        if (active) setError(toMessage(cause))
      } finally {
        if (active) setLoading(false)
      }
    }
    void recover()
    return () => { active = false }
  }, [dataset, datasetId, requestedSnapshotId, snapshot])

  const submitTraining = async (config: RunConfig) => {
    const record = await perform('submit-run', () => api.submitRun(config))
    if (!record) return
    notify(`训练 ${record.run_id} 已进入队列。`)
    navigate(`/runs/${record.run_id}`)
  }

  if (!datasetId) {
    return <section className="route-section"><div className="panel empty-route"><strong>尚未选择训练数据</strong><span>请先从数据集详情冻结数据并生成训练快照。</span><Link to="/datasets">选择数据集</Link></div></section>
  }

  return (
    <section className="route-section">
      <div className="page-toolbar"><Link className="back-link" to={`/datasets/${datasetId}`}>← 返回数据集详情</Link></div>
      <SectionTitle eyebrow="REPRODUCIBLE TRAINING" title="配置训练任务" note="配置会随运行固化并计算哈希" />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在恢复训练快照…</div> : (
        <TrainingForm snapshot={snapshot} dataset={dataset} enabled={Boolean(health?.training_submission_enabled)} busy={busy} onSubmit={submitTraining} />
      )}
    </section>
  )
}
