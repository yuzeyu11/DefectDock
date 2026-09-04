import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, clearApiToken, setApiToken } from './api'

function memoryStorage(): Storage {
  const values = new Map<string, string>()
  return {
    get length() { return values.size },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => { values.delete(key) },
    setItem: (key, value) => { values.set(key, value) },
  }
}

describe('network API credentials', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    Object.defineProperty(globalThis, 'window', {
      configurable: true,
      value: { sessionStorage: memoryStorage() },
    })
  })

  it('adds the session token to API requests and removes it on disconnect', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } }),
    )
    setApiToken('x'.repeat(32))
    await api.listDatasets()
    const authenticatedHeaders = new Headers(fetchMock.mock.calls.at(0)?.[1]?.headers)
    expect(authenticatedHeaders.get('Authorization')).toBe(`Bearer ${'x'.repeat(32)}`)

    clearApiToken()
    await api.listDatasets()
    const disconnectedHeaders = new Headers(fetchMock.mock.calls.at(1)?.[1]?.headers)
    expect(disconnectedHeaders.has('Authorization')).toBe(false)
  })
})
