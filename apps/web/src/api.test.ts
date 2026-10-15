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

  it('shows the Retry-After delay without retrying a mutation', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({ detail: 'Request rate limit exceeded' }),
      { status: 429, headers: { 'Content-Type': 'application/json', 'Retry-After': '30' } },
    ))
    await expect(api.cancelRun('run-test')).rejects.toMatchObject({
      status: 429,
      retryAfterSeconds: 30,
      message: expect.stringContaining('等待 30 秒'),
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it.each([undefined, 'invalid-delay'])('handles missing or invalid Retry-After: %s', async (value) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('Rate limited', {
      status: 429,
      headers: value ? { 'Retry-After': value } : {},
    }))
    await expect(api.listDatasets()).rejects.toMatchObject({
      status: 429,
      retryAfterSeconds: null,
      message: expect.stringContaining('请稍后重试'),
    })
  })

  it('handles an HTTP-date retry header', async () => {
    vi.spyOn(Date, 'now').mockReturnValue(Date.parse('2026-09-04T00:00:00Z'))
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', {
      status: 429,
      headers: { 'Retry-After': 'Fri, 04 Sep 2026 00:00:45 GMT' },
    }))
    await expect(api.listDatasets()).rejects.toMatchObject({ retryAfterSeconds: 45 })
  })

  it('also preserves rate-limit feedback when loading image blobs', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', {
      status: 429,
      headers: { 'Retry-After': '10' },
    }))
    await expect(api.datasetImage('/api/datasets/test/images/test')).rejects.toMatchObject({
      status: 429,
      retryAfterSeconds: 10,
    })
  })
})
