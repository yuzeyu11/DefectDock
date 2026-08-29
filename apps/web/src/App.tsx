import { useEffect, useState } from 'react'

type Health = {
  status: string
  version: string
  dataset_upload_enabled: boolean
  inference_ready: boolean
}

type Run = {
  run_id: string
  project: string
  model: string
  status: string
  created_at: string
}

type Config = {
  name: string
  project: string
  model: string
  dataset: { version: string }
}

const stages = [
  { number: '01', title: '数据接入', note: '上传、去重、版本冻结' },
  { number: '02', title: '标注协同', note: 'CVAT 任务与同步' },
  { number: '03', title: '训练验收', note: '配置、指标与漏检分析' },
  { number: '04', title: '部署运行', note: '模型激活与现场输入' },
]

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [configs, setConfigs] = useState<Config[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        const [healthResponse, runsResponse, configsResponse] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/runs?limit=5'),
          fetch('/api/configs/examples'),
        ])
        if (![healthResponse, runsResponse, configsResponse].every((item) => item.ok)) {
          throw new Error('API response was not successful')
        }
        setHealth((await healthResponse.json()) as Health)
        setRuns((await runsResponse.json()) as Run[])
        setConfigs((await configsResponse.json()) as Config[])
      } catch {
        setError('后端尚未连接。启动 DefectDock API 后刷新页面。')
      }
    }
    void load()
  }, [])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">D</div>
          <div>
            <strong>DefectDock</strong>
            <span>VISION OPERATIONS</span>
          </div>
        </div>
        <nav aria-label="主导航">
          <a className="active" href="#overview"><span>◫</span>总览</a>
          <a href="#workflow"><span>◇</span>项目工作流</a>
          <a href="#runs"><span>◎</span>训练运行</a>
          <a href="#models"><span>⬡</span>模型资产</a>
          <a href="#settings"><span>⚙</span>系统设置</a>
        </nav>
        <div className="sidebar-footer">
          <span className={`status-dot ${health ? 'online' : ''}`} />
          <div><strong>{health ? '服务正常' : '等待连接'}</strong><small>API {health?.version ?? '—'}</small></div>
        </div>
      </aside>

      <main>
        <header>
          <div><p className="eyebrow">INDUSTRIAL VISION WORKBENCH</p><h1>交付控制台</h1></div>
          <div className="header-actions"><button className="ghost">查看接口</button><button className="primary">新建项目</button></div>
        </header>

        {error && <div className="notice">{error}</div>}

        <section className="hero" id="overview">
          <div>
            <span className="pill">ENGINEERING BASELINE · 0.1.0</span>
            <h2>让视觉模型从数据<br />走到真实产线。</h2>
            <p>在一个可追溯的工作流中完成数据治理、训练、工业指标验收与部署准备。</p>
          </div>
          <div className="hero-metric">
            <span>交付链路</span><strong>4 / 4</strong><small>核心阶段已建立工程边界</small>
            <div className="metric-line"><i /></div>
          </div>
        </section>

        <section className="workflow" id="workflow">
          <div className="section-title"><div><p className="eyebrow">DELIVERY FLOW</p><h3>从原始图片到可验收模型</h3></div><span>可审计 · 可复现 · 可替换</span></div>
          <div className="stage-grid">
            {stages.map((stage, index) => (
              <article key={stage.number}>
                <div className="stage-top"><span>{stage.number}</span><i className={index === 0 ? 'ready' : ''} /></div>
                <h4>{stage.title}</h4><p>{stage.note}</p>
              </article>
            ))}
          </div>
        </section>

        <div className="content-grid">
          <section className="panel" id="runs">
            <div className="section-title"><div><p className="eyebrow">RECENT RUNS</p><h3>最近训练</h3></div><button className="text-button">查看全部 →</button></div>
            {runs.length ? runs.map((run) => (
              <div className="run-row" key={run.run_id}>
                <div className="run-icon">↗</div>
                <div className="run-name"><strong>{run.project}</strong><span>{run.model}</span></div>
                <code>{run.run_id.slice(-8)}</code>
                <span className={`run-status ${run.status}`}>{run.status}</span>
              </div>
            )) : <div className="empty"><strong>暂无训练记录</strong><span>验证配置后，从 CLI 发起第一条可复现训练。</span></div>}
          </section>

          <section className="panel" id="models">
            <div className="section-title"><div><p className="eyebrow">STARTING POINTS</p><h3>训练配置</h3></div></div>
            {configs.length ? configs.map((config) => (
              <div className="config-card" key={config.name}>
                <span className="config-symbol">F</span>
                <div><strong>{config.project}</strong><small>{config.model}</small></div>
                <em>{config.dataset.version}</em>
              </div>
            )) : <div className="empty compact"><strong>等待 API</strong><span>配置通过后端自动发现。</span></div>}
            <div className="boundary"><span>✓</span><div><strong>许可证边界已启用</strong><small>内置引擎：PyTorch / TorchVision</small></div></div>
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
