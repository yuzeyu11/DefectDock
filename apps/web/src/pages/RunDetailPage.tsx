import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { RunDetail, SectionTitle } from '../components'
import { toMessage } from '../errors'
import type { Run, RunArtifacts, RunEvent } from '../types'
import { useWorkbench } from '../workbench'

const terminalStatuses = new Set(['succeeded', 'failed', 'cancelled'])

export function RunDetailPage() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const { busy, perform, notify } = useWorkbench()
  const [run, setRun] = useState<Run | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [artifacts, setArtifacts] = useState<RunArtifacts | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<number | null>(null)

  const refresh = useCallback(async (): Promise<Run | null> => {
    if (!runId) return null
    try {
      const [record, nextEvents, nextArtifacts] = await Promise.all([
        api.run(runId),
        api.runEvents(runId),
        api.runArtifacts(runId),
      ])
      setRun(record)
      setEvents(nextEvents)
      setArtifacts(nextArtifacts)
      setError(null)
      return record
    } catch (cause) {
      setError(toMessage(cause))
      return null
    } finally {
      setLoading(false)
    }
  }, [runId])

  useEffect(() => {
    let active = true
    const poll = async () => {
      const record = await refresh()
      if (active && record && !terminalStatuses.has(record.status)) {
        timerRef.current = window.setTimeout(() => void poll(), 3000)
      }
    }
    void poll()
    return () => {
      active = false
      if (timerRef.current !== null) window.clearTimeout(timerRef.current)
    }
  }, [refresh])

  const cancelRun = async () => {
    if (!runId) return
    const result = await perform('cancel-run', () => api.cancelRun(runId))
    if (!result) return
    await refresh()
    notify('已发送取消请求。')
  }

  const activateRun = async () => {
    if (!runId) return
    const result = await perform('activate-run', () => api.activateRun(runId))
    if (result === undefined) return
    notify('模型已激活，推理工作区现在可以使用。')
    navigate('/inference')
  }

  return (
    <section className="route-section">
      <div className="page-toolbar"><Link className="back-link" to="/runs">← 返回训练运行</Link></div>
      <SectionTitle eyebrow="RUN DETAIL" title={run?.project ?? '运行详情'} note={run && !terminalStatuses.has(run.status) ? '运行中每 3 秒自动刷新' : '指标、产物与失败信息'} />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在加载运行详情…</div> : (
        <RunDetail run={run} events={events} artifacts={artifacts} busy={busy} onCancel={cancelRun} onActivate={activateRun} />
      )}
    </section>
  )
}
