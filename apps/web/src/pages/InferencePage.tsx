import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { InferenceWorkspace } from '../components'
import { toMessage } from '../errors'
import type { InferenceStatus } from '../types'
import { useWorkbench } from '../workbench'

export function InferencePage() {
  const { busy, perform } = useWorkbench()
  const [status, setStatus] = useState<InferenceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.inferenceStatus())
      setError(null)
    } catch (cause) {
      setError(toMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  if (loading) return <section className="route-section"><div className="panel route-loading">正在检查推理服务…</div></section>
  return <>{error && <div className="page-error" role="alert">{error}</div>}<InferenceWorkspace status={status} busy={busy} perform={perform} /></>
}
