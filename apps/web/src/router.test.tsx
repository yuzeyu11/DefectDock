import { renderToString } from 'react-dom/server'
import { createMemoryRouter, matchRoutes, RouterProvider } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { routes } from './routes'

function renderRoute(path: string): string {
  const memoryRouter = createMemoryRouter(routes, { initialEntries: [path] })
  return renderToString(<RouterProvider router={memoryRouter} />)
}

describe('workbench routes', () => {
  it.each([
    ['/overview', '交付工作台'],
    ['/datasets', '全部数据集'],
    ['/datasets/ds-example', '数据集详情'],
    ['/training/new', '尚未选择训练数据'],
    ['/runs', '全部训练运行'],
    ['/runs/run-example', '运行详情'],
    ['/models', '模型版本'],
    ['/inference', '正在检查推理服务'],
    ['/unknown', '找不到这个页面'],
  ])('renders %s as an independent page', (path, expectedText) => {
    expect(renderRoute(path)).toContain(expectedText)
  })

  it('matches the root redirect route', () => {
    const matches = matchRoutes(routes, '/')
    expect(matches?.at(-1)?.route.index).toBe(true)
  })
})
