import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { SectionTitle } from '../components'
import { toMessage } from '../errors'
import type { ModelActivation, ModelVersion } from '../types'
import { useWorkbench } from '../workbench'

export function ModelsPage() {
  const { busy, perform, notify } = useWorkbench()
  const [models, setModels] = useState<ModelVersion[]>([])
  const [history, setHistory] = useState<ModelActivation[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const [nextModels, nextHistory] = await Promise.all([
        api.listModels(),
        api.activationHistory(),
      ])
      setModels(nextModels)
      setHistory(nextHistory)
      setError(null)
    } catch (cause) {
      setError(toMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const activate = async (model: ModelVersion) => {
    const result = await perform('activate-model', () => api.activateModel(model.model_version_id))
    if (result === undefined) return
    await refresh()
    notify(`模型 ${model.model_version_id} 已通过完整性校验并激活。`)
  }

  const approve = async (model: ModelVersion) => {
    const result = await perform('approve-model', () => api.approveModel(model.model_version_id))
    if (!result) return
    await refresh()
    notify(`模型 ${model.model_version_id} 已批准，可进行激活。`)
  }

  const rollback = async () => {
    const result = await perform('rollback-model', () => api.rollbackModel())
    if (result === undefined) return
    await refresh()
    notify('已回滚到上一模型版本。')
  }

  const exportOnnx = async (model: ModelVersion) => {
    const result = await perform('export-onnx', () => api.exportOnnx(model.model_version_id))
    if (!result) return
    const latency = result.export.benchmark.median_ms
    notify(`ONNX ${result.created ? '导出完成' : '制品已存在并通过校验'}${latency === undefined ? '' : `，CPU 中位耗时 ${latency.toFixed(1)} ms`}。`)
  }

  const active = models.find((model) => model.is_active)
  const canRollback = Boolean(active && history.find((event) => event.model_version_id === active.model_version_id)?.previous_model_version_id)

  return (
    <section className="route-section">
      <SectionTitle eyebrow="MODEL GOVERNANCE" title="模型版本" note="注册 · 完整性校验 · 原子激活 · 回滚" />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在加载模型注册表…</div> : models.length === 0 ? (
        <div className="panel empty-route"><strong>尚无注册模型</strong><span>在成功的训练运行详情中激活最佳模型，系统会先登记不可变版本。</span></div>
      ) : (
        <div className="model-governance-grid">
          <div className="panel model-list">
            <div className="panel-heading"><strong>模型版本</strong><small>{models.length} 个版本</small></div>
            {models.map((model) => (
              <article key={model.model_version_id} className={model.is_active ? 'active' : undefined}>
                <div><strong>{model.project}</strong><small>{model.model_version_id}</small></div>
                <dl><dt>运行</dt><dd>{model.run_id}</dd><dt>数据</dt><dd>{model.dataset_version}</dd><dt>SHA</dt><dd>{model.artifact_sha256.slice(0, 16)}</dd><dt>审批</dt><dd>{model.approval_status === 'approved' ? `${model.approved_by} · 已批准` : '等待批准'}</dd></dl>
                <div className="model-actions">
                  {model.approval_status === 'candidate' && <button className="primary" disabled={Boolean(busy)} onClick={() => void approve(model)}>{busy === 'approve-model' ? '正在批准…' : '批准模型'}</button>}
                  {model.is_active ? <span className="status-chip succeeded">当前激活</span> : <button disabled={Boolean(busy) || model.approval_status !== 'approved'} onClick={() => void activate(model)}>激活</button>}
                  <button disabled={Boolean(busy)} onClick={() => void exportOnnx(model)}>{busy === 'export-onnx' ? '导出中…' : '导出 ONNX'}</button>
                </div>
              </article>
            ))}
          </div>
          <div className="panel activation-history">
            <div className="panel-heading"><strong>激活历史</strong><button className="danger" disabled={!canRollback || Boolean(busy)} onClick={() => void rollback()}>{busy === 'rollback-model' ? '正在回滚…' : '回滚上一版本'}</button></div>
            {history.map((event) => <div key={event.activation_id}><span>{event.action === 'rollback' ? '回滚' : '激活'}</span><strong>{event.model_version_id}</strong><small>{event.actor} · {new Date(event.created_at).toLocaleString()}</small></div>)}
          </div>
        </div>
      )}
    </section>
  )
}
