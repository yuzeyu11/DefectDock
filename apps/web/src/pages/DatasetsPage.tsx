import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { DatasetList, SectionTitle } from '../components'
import { toMessage } from '../errors'
import type { Dataset } from '../types'
import { useWorkbench } from '../workbench'

export function DatasetsPage() {
  const navigate = useNavigate()
  const { openCreateDataset } = useWorkbench()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setDatasets(await api.listDatasets())
      setError(null)
    } catch (cause) {
      setError(toMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  return (
    <section className="route-section">
      <SectionTitle eyebrow="DATA GOVERNANCE" title="全部数据集" note="点击数据集进入独立详情页" />
      {error && <div className="page-error" role="alert">{error}</div>}
      {loading ? <div className="panel route-loading">正在加载数据集…</div> : (
        <DatasetList datasets={datasets} selectedId={null} onSelect={(id) => navigate(`/datasets/${id}`)} onCreate={openCreateDataset} />
      )}
    </section>
  )
}
