import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { CreateDatasetDialog, Feedback } from './components'
import { ImagePicker, OperationFeedback } from './ui'
import { validateImageFiles } from './ui-utils'

const noop = () => {}

describe('image selection feedback', () => {
  it('accepts supported image extensions without relying on browser MIME detection', () => {
    expect(validateImageFiles([{ name: 'sample.PNG', size: 100 }], false, 1000)).toBeNull()
  })
  it('rejects non-images', () => {
    expect(validateImageFiles([{ name: 'notes.txt', size: 100 }], false, 1000)).toContain('请选择')
  })
  it('rejects empty files', () => {
    expect(validateImageFiles([{ name: 'empty.png', size: 0 }], false, 1000)).toContain('内容为空')
  })
  it('rejects multiple images in single-image inference', () => {
    expect(validateImageFiles([{ name: 'one.png', size: 100 }, { name: 'two.png', size: 100 }], false, 1000)).toContain('一次只能')
  })
  it('checks the combined size, not only individual files', () => {
    expect(validateImageFiles([{ name: 'one.png', size: 600 }, { name: 'two.png', size: 600 }], true, 1000)).toContain('超过')
  })
  it('allows exactly the configured maximum', () => {
    expect(validateImageFiles([{ name: 'one.webp', size: 1000 }], true, 1000)).toBeNull()
  })
  it('provides a keyboard-accessible file input and size instructions', () => {
    const html = renderToStaticMarkup(<ImagePicker files={[]} onChange={noop} />)
    expect(html).toContain('aria-label="选择检测图片"')
    expect(html).toContain('25 MB')
    expect(html).toContain('将图片拖到这里')
  })
})

describe('operation and result feedback', () => {
  it('does not display a pending state when idle', () => {
    expect(renderToStaticMarkup(<OperationFeedback operation={null} />)).toBe('')
  })
  it('announces the real operation without claiming numeric progress', () => {
    const html = renderToStaticMarkup(<OperationFeedback operation="export-onnx" />)
    expect(html).toContain('role="status"')
    expect(html).toContain('正在导出与验证 ONNX')
    expect(html).not.toContain('aria-valuenow')
  })
  it('gives errors priority over stale success feedback', () => {
    const html = renderToStaticMarkup(<Feedback error="上传失败" message="已成功" onClose={noop} />)
    expect(html).toContain('role="alert"')
    expect(html).toContain('上传失败')
    expect(html).not.toContain('已成功')
  })
  it('announces successful actions and provides an accessible dismiss button', () => {
    const html = renderToStaticMarkup(<Feedback error={null} message="状态已更新" onClose={noop} />)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-label="关闭提示"')
  })
  it('marks the create form as a modal and prevents empty submissions', () => {
    const html = renderToStaticMarkup(<CreateDatasetDialog busy={false} onClose={noop} onSubmit={noop} />)
    expect(html).toContain('role="dialog"')
    expect(html).toContain('aria-modal="true"')
    expect(html).toContain('class="primary" disabled=""')
  })
})
