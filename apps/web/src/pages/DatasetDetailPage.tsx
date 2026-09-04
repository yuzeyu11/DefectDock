import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { DatasetWorkspace, SectionTitle } from '../components'
import { toMessage } from '../errors'
import type { DatasetDetail } from '../types'
import { useWorkbench } from '../workbench'

export function DatasetDetailPage() {
  const { datasetId } = useParams()
  const navigate = useNavigate()
  const { busy, perform, notify } = useWorkbench()
  const [detail, setDetail] = useState<DatasetDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!datasetId) return
    try {
      setDetail(await api.dataset(datasetId))
      setError(null)
    } catch (cause) {
      setError(toMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => { void refresh() }, [refresh])

  const uploadAnnotations = async (files: File[]) => {
    if (!datasetId || files.length === 0) return
    const result = await perform('annotations', () => api.uploadAnnotations(datasetId, files))
    if (result === undefined) return
    await refresh()
    notify(`已导入 ${files.length} 个标注文件并建立新版本。`)
  }

  const freezeAndSnapshot = async () => {
    if (!datasetId || !detail) return
    const result = await perform('snapshot', async () => {
      if (detail.status !== 'frozen') await api.freezeDataset(datasetId)
      return api.createSnapshot(datasetId)
    })
    if (!result) return
    notify('不可变训练快照已生成，可以配置训练。')
    const query = new URLSearchParams({ datasetId, snapshotId: result.snapshot.snapshot_id })
    navigate(`/training/new?${query.toString()}`, { state: { dataset: detail, snapshot: result } })
  }

  const autoAnnotations = async () => {
    if (!datasetId) return
    const result = await perform('auto-annotations', () => api.autoAnnotate(datasetId))
    if (!result) return
    await refresh()
    notify(`模型生成了 ${result.detection_count} 个候选框，请人工检查后批准。`)
  }

  const approveAnnotations = async () => {
    const version = detail?.current_annotation_version
    if (!datasetId || !version || version.review_status !== 'candidate') return
    const result = await perform(
      'approve-annotations',
      () => api.approveAnnotations(datasetId, version.annotation_version_id),
    )
    if (!result) return
    await refresh()
    notify('候选标注已批准，现在可以冻结训练快照。')
  }

  const createCvatTask = async () => {
    if (!datasetId) return
    const result = await perform('cvat-create', () => api.createCvatTask(datasetId))
    if (!result) return
    await refresh()
    notify(`CVAT 任务 ${result.task_id} 已创建。`)
  }

  const syncCvatAnnotations = async () => {
    if (!datasetId) return
    const result = await perform('cvat-sync', () => api.syncCvatAnnotations(datasetId))
    if (!result) return
    await refresh()
    notify('CVAT 标注已同步并冻结为不可变版本。')
  }

  return (
    <section className="route-section">
      <div className="page-toolbar"><Link className="back-link" to="/datasets">← 返回数据集</Link></div>
      <SectionTitle eyebrow="DATASET DETAIL" title={detail?.name ?? '数据集详情'} note="图片、标注版本与训练快照" />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在加载数据集详情…</div> : (
        <DatasetWorkspace detail={detail} busy={busy} onAnnotations={uploadAnnotations} onAutoAnnotations={autoAnnotations} onApproveAnnotations={approveAnnotations} onCreateCvat={createCvatTask} onSyncCvat={syncCvatAnnotations} onSnapshot={freezeAndSnapshot} />
      )}
    </section>
  )
}
