import { useEffect, useRef } from 'react'

export function useDialog(busy: boolean, onClose?: () => void) {
  const ref = useRef<HTMLFormElement>(null)
  const latest = useRef({ busy, onClose })
  useEffect(() => { latest.current = { busy, onClose } }, [busy, onClose])
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const focusables = () => Array.from(ref.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex="0"]') ?? [])
    const input = ref.current?.querySelector<HTMLElement>('input:not([type="file"])')
    ;(input ?? focusables()[0] ?? ref.current)?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (!latest.current.busy) latest.current.onClose?.()
      }
      if (event.key === 'Tab') {
        const elements = focusables()
        const first = elements[0]
        const last = elements.at(-1)
        if (!first) { event.preventDefault(); ref.current?.focus(); return }
        if (event.shiftKey && (document.activeElement === first || !ref.current?.contains(document.activeElement))) {
          event.preventDefault(); last?.focus()
        } else if (!event.shiftKey && (document.activeElement === last || !ref.current?.contains(document.activeElement))) {
          event.preventDefault(); first.focus()
        }
      }
    }
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.body.style.overflow = overflow
      if (previous?.isConnected) previous.focus()
    }
  }, [])
  return ref
}

export function validateImageFiles(files: readonly Pick<File, 'name' | 'size'>[], multiple: boolean, maxBytes: number): string | null {
  if (!multiple && files.length > 1) return '一次只能检测一张图片，请重新选择。'
  if (files.some((file) => !/\.(jpe?g|png|bmp|tiff?|webp)$/i.test(file.name))) return '请选择 JPG、PNG、BMP、TIFF 或 WebP 图片。'
  if (files.some((file) => file.size === 0)) return '文件内容为空，请重新选择有效图片。'
  if (files.reduce((sum, file) => sum + file.size, 0) > maxBytes) return `所选文件超过 ${(maxBytes / 1024 / 1024).toFixed(0)} MB，请减少图片数量或压缩文件。`
  return null
}
