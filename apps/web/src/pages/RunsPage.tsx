import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { RunList, SectionTitle } from '../components'
import { toMessage } from '../errors'
import type { Run } from '../types'

export function RunsPage() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async (quiet = false) => {
    try {
      setRuns(await api.listRuns())
      setError(null)
    } catch (cause) {
      if (!quiet) setError(toMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(true), 5000)
    return () => window.clearInterval(timer)
  }, [refresh])

  return (
    <section className="route-section">
      <SectionTitle eyebrow="RUN OPERATIONS" title="全部训练运行" note="点击运行进入指标与产物详情" />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在加载训练运行…</div> : (
        <RunList runs={runs} selectedId={null} onSelect={(id) => navigate(`/runs/${id}`)} />
      )}
    </section>
  )
}
