import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Overview } from '../components'
import { toMessage } from '../errors'
import type { Dataset, InferenceStatus, Run } from '../types'
import { useWorkbench } from '../workbench'
import { Icon } from '../ui'

export function OverviewPage() {
  const navigate = useNavigate()
  const { health, openCreateDataset, notify } = useWorkbench()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [inference, setInference] = useState<InferenceStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  const refresh = useCallback(async (quiet = false) => {
    try {
      const [nextDatasets, nextRuns, nextInference] = await Promise.all([
        api.listDatasets(),
        api.listRuns(),
        api.inferenceStatus(),
      ])
      setDatasets(nextDatasets)
      setRuns(nextRuns)
      setInference(nextInference)
      setError(null)
      setUpdatedAt(new Date())
      return true
    } catch (cause) {
      if (!quiet) setError(toMessage(cause))
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(true), 5000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const completedStages = [
    datasets.length > 0,
    datasets.some((dataset) => dataset.status === 'frozen'),
    runs.some((run) => run.status === 'succeeded'),
    Boolean(inference?.available),
  ].filter(Boolean).length

  const steps = [
    { title: '准备数据', description: '上传图片，建立数据集', to: '/datasets', done: datasets.length > 0, icon: 'folder' as const },
    { title: '标注与快照', description: '审核标注，固化训练数据', to: '/datasets', done: datasets.some((item) => item.status === 'frozen'), icon: 'layers' as const },
    { title: '训练模型', description: '跟进训练，查看运行结果', to: '/runs', done: runs.some((item) => item.status === 'succeeded'), icon: 'activity' as const },
    { title: '验证效果', description: '上传现场图片，验证检测', to: '/inference', done: Boolean(inference?.available), icon: 'scan' as const },
  ]

  return (
    <>
      {error && <div className="page-error" role="alert">{error}</div>}
      <section className="hero">
        <div>
          <span className="pill"><span className="tiny-dot" /> YOUR VISION WORKSPACE</span>
          <h2>从一张图片开始，<br />让检测更进一步。</h2>
          <p>整理数据、训练模型、验证效果。把每一次探索，变成可以追溯的进步。</p>
          <div className="overview-actions">
            <button className="primary" disabled={loading || (!datasets.length && !health?.dataset_upload_enabled)} onClick={() => datasets.length ? navigate('/datasets') : openCreateDataset()}>{datasets.length ? '继续我的工作' : '创建第一个数据集'}<Icon name="arrow" /></button>
            <button onClick={() => navigate('/runs')}>查看训练运行</button>
          </div>
        </div>
        <div className="hero-metric">
          <span>工作空间进度</span><strong>{loading ? '—' : `${completedStages}`}<em> / 4</em></strong><small>{completedStages === 4 ? '端到端流程已就绪' : '每一步，都让模型更近一点'}</small>
          <div className="metric-line" role="progressbar" aria-label="工作空间完成阶段" aria-valuenow={completedStages} aria-valuemin={0} aria-valuemax={4}><i style={{ width: `${completedStages * 25}%` }} /></div>
        </div>
      </section>
      <div className="overview-section-heading"><div><h2>工作空间一览</h2><span>{updatedAt ? `最近同步 ${updatedAt.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}` : '正在获取工作空间状态'}</span></div><button className="text-button" disabled={refreshing || loading} onClick={async () => { setRefreshing(true); const ok = await refresh(); setRefreshing(false); if (ok) notify('工作空间状态已更新。') }}>{refreshing ? <><span className="spinner" />正在刷新</> : '刷新状态 ↻'}</button></div>
      <Overview health={health} datasets={datasets} runs={runs} inference={inference} loading={loading} />
      <div className="overview-section-heading"><div><h2>下一步，从这里开始</h2><span>跟随流程，也可以随时回到任何一个环节</span></div></div>
      <section className="workflow-grid" aria-label="主工作流">
        {steps.map((step, index) => <button key={step.title} className={`workflow-step ${step.done ? 'is-complete' : ''}`} onClick={() => navigate(step.to)}><div className="step-top"><span className="step-symbol"><Icon name={step.icon} /></span><small>{step.done ? <><Icon name="check" />已就绪</> : `0${index + 1}`}</small></div><strong>{step.title}</strong><span>{step.description}</span><div className="step-link">进入工作区 <Icon name="arrow" /></div></button>)}
      </section>
    </>
  )
}
