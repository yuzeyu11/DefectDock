import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { api, clearApiToken, hasApiToken, setApiToken } from '../api'
import { CreateDatasetDialog, Feedback, NetworkAccessDialog, type CreateDatasetInput } from '../components'
import { toMessage } from '../errors'
import type { Health } from '../types'
import type { WorkbenchContext } from '../workbench'
import { Icon, OperationFeedback, type IconName } from '../ui'

const navigation = [
  { to: '/overview', icon: 'grid', label: '运行总览' },
  { to: '/datasets', icon: 'folder', label: '数据集' },
  { to: '/runs', icon: 'activity', label: '训练运行' },
  { to: '/models', icon: 'layers', label: '模型注册' },
  { to: '/inference', icon: 'scan', label: '模型推理' },
] satisfies { to: string; icon: IconName; label: string }[]

export function WorkbenchLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const [health, setHealth] = useState<Health | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const operationInFlight = useRef(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [showNetworkAccess, setShowNetworkAccess] = useState(false)
  const [accessBusy, setAccessBusy] = useState(false)
  const [accessError, setAccessError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const result = await api.health()
        if (active) {
          setHealth(result)
          setShowNetworkAccess(result.authentication_required && !hasApiToken())
        }
      } catch {
        if (active) setHealth(null)
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 15000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const perform = useCallback(async <T,>(label: string, action: () => Promise<T>): Promise<T | undefined> => {
    if (operationInFlight.current) return undefined
    operationInFlight.current = true
    setBusy(label)
    setError(null)
    setMessage(null)
    try {
      return await action()
    } catch (cause) {
      setError(toMessage(cause))
      return undefined
    } finally {
      operationInFlight.current = false
      setBusy(null)
    }
  }, [])

  const notify = useCallback((text: string) => {
    setError(null)
    setMessage(text)
  }, [])

  const createDataset = async (input: CreateDatasetInput) => {
    const result = await perform('create-dataset', () => api.createDataset(input.name, input.scene, input.labels, input.files))
    if (!result) return
    setShowCreate(false)
    notify(`数据集“${result.dataset.name}”已创建，可继续上传标注。`)
    navigate(`/datasets/${result.dataset.dataset_id}`)
  }

  const connectNetwork = async (token: string) => {
    setAccessBusy(true)
    setAccessError(null)
    setApiToken(token)
    try {
      await api.listDatasets()
      setShowNetworkAccess(false)
      window.location.reload()
    } catch (cause) {
      clearApiToken()
      setAccessError(toMessage(cause))
    } finally {
      setAccessBusy(false)
    }
  }

  const disconnectNetwork = () => {
    clearApiToken()
    setShowNetworkAccess(true)
  }

  const meta = pageMeta(location.pathname)
  const canCreateDataset = location.pathname === '/overview' || location.pathname === '/datasets'
  const context = useMemo<WorkbenchContext>(() => ({
    health,
    busy,
    perform,
    notify,
    openCreateDataset: () => setShowCreate(true),
  }), [busy, health, notify, perform])

  useEffect(() => {
    document.title = `${meta.title} · DefectDock`
  }, [meta.title])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink className="brand" to="/overview" aria-label="DefectDock 总览">
          <div className="brand-mark">D</div>
          <div><strong>DefectDock</strong><span>VISION OPERATIONS</span></div>
        </NavLink>
        <p className="nav-caption">工作空间</p>
        <nav aria-label="主导航">
          {navigation.map((item) => (
            <NavLink key={item.to} to={item.to} className={({ isActive }) => isActive ? 'active' : undefined}>
              <Icon name={item.icon} />{item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className={`status-dot ${health ? 'online' : ''}`} />
          <div><strong>{health ? '服务正常' : '等待连接'}</strong><small>API {health?.version ?? '—'}</small></div>
        </div>
      </aside>

      <main>
        <header>
          <div><p className="eyebrow">{meta.eyebrow}</p><h1>{meta.title}</h1></div>
          <div className="header-actions">
            <button className="ghost" onClick={() => window.open('/docs', '_blank')}>接口文档</button>
            {health?.authentication_required && <button className="ghost" onClick={disconnectNetwork}>断开网络凭据</button>}
            {canCreateDataset && <button className="primary" onClick={() => setShowCreate(true)} disabled={!health?.dataset_upload_enabled || Boolean(busy)}><Icon name="plus" />创建数据集</button>}
          </div>
        </header>

        <Feedback key={error ?? message ?? 'idle'} error={error} message={message} onClose={() => { setError(null); setMessage(null) }} />
        <OperationFeedback operation={busy} />
        <div className="page-content"><Outlet context={context} /></div>
      </main>

      {showCreate && <CreateDatasetDialog busy={busy === 'create-dataset'} maxBytes={health ? Math.max(1, health.max_request_bytes - 1024 * 1024) : undefined} onClose={() => setShowCreate(false)} onSubmit={createDataset} />}
      {showNetworkAccess && <NetworkAccessDialog busy={accessBusy} error={accessError} onSubmit={connectNetwork} />}
    </div>
  )
}

function pageMeta(pathname: string): { eyebrow: string; title: string } {
  if (pathname === '/overview') return { eyebrow: 'INDUSTRIAL VISION WORKBENCH', title: '交付工作台' }
  if (pathname === '/datasets') return { eyebrow: 'DATA GOVERNANCE', title: '数据集' }
  if (pathname.startsWith('/datasets/')) return { eyebrow: 'DATA GOVERNANCE', title: '数据集详情' }
  if (pathname === '/training/new') return { eyebrow: 'REPRODUCIBLE TRAINING', title: '新建训练' }
  if (pathname === '/runs') return { eyebrow: 'RUN OPERATIONS', title: '训练运行' }
  if (pathname.startsWith('/runs/')) return { eyebrow: 'RUN OPERATIONS', title: '运行详情' }
  if (pathname === '/models') return { eyebrow: 'MODEL GOVERNANCE', title: '模型注册' }
  if (pathname === '/inference') return { eyebrow: 'MODEL VERIFICATION', title: '模型推理' }
  return { eyebrow: 'DEFECTDOCK', title: '页面不存在' }
}
