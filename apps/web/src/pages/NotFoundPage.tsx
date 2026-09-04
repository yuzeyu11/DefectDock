import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <section className="route-section">
      <div className="panel empty-route">
        <strong>找不到这个页面</strong>
        <span>地址可能已失效，或对应资源已经移动。</span>
        <Link to="/overview">返回工作台</Link>
      </div>
    </section>
  )
}
