import { ApiError } from './api'

export function toMessage(cause: unknown): string {
  if (cause instanceof ApiError || cause instanceof Error) return cause.message
  return '操作失败，请查看后端日志后重试。'
}
