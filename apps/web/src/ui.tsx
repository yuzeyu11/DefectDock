import { useId, useState, type ReactNode } from 'react'
import { validateImageFiles } from './ui-utils'

export type IconName = 'grid' | 'folder' | 'activity' | 'layers' | 'scan' | 'upload' | 'check' | 'arrow' | 'plus' | 'close'

const paths: Record<IconName, ReactNode> = {
  grid: <><rect x="3" y="3" width="7" height="7" rx="2" /><rect x="14" y="3" width="7" height="7" rx="2" /><rect x="3" y="14" width="7" height="7" rx="2" /><rect x="14" y="14" width="7" height="7" rx="2" /></>,
  folder: <path d="M3 7V5a2 2 0 0 1 2-2h5l3 4h6a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />,
  activity: <><path d="M3 12h4l3-7 4 14 3-7h4" /></>,
  layers: <><path d="m12 3 10 5-10 5L2 8l10-5Zm-9 10 9 5 9-5M3 18l9 4 9-4" /></>,
  scan: <><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3M3 12h18" /></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
  check: <path d="m5 12 4 4L19 6" />,
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
}

export function Icon({ name }: { name: IconName }) {
  return <svg className="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

const operationLabels: Record<string, string> = {
  'create-dataset': '正在上传图片并创建数据集',
  annotations: '正在校验标注并建立版本',
  'auto-annotations': '模型正在生成候选标注',
  'approve-annotations': '正在保存标注审批结果',
  snapshot: '正在检查数据并生成训练快照',
  'cvat-create': '正在创建 CVAT 标注任务',
  'cvat-sync': '正在同步 CVAT 标注',
  'submit-run': '正在提交训练任务',
  'cancel-run': '正在请求取消训练',
  'activate-run': '正在校验并激活最佳模型',
  'activate-model': '正在切换模型版本',
  'approve-model': '正在保存模型审批结果',
  'rollback-model': '正在回滚到上一模型版本',
  'export-onnx': '正在导出与验证 ONNX 模型',
  detect: '正在分析图片，请稍候',
}

export function OperationFeedback({ operation }: { operation: string | null }) {
  if (!operation) return null
  return <div className="operation-feedback" role="status" aria-live="polite"><span className="spinner" aria-hidden="true" /><div><strong>{operationLabels[operation] ?? '正在处理操作'}</strong><small>完成后会更新结果，请勿重复提交。</small></div></div>
}

export function ImagePicker({ files, onChange, multiple = false, disabled = false, maxBytes = 25 * 1024 * 1024, preview }: { files: File[]; onChange: (files: File[]) => void; multiple?: boolean; disabled?: boolean; maxBytes?: number; preview?: string | null }) {
  const id = useId()
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const select = (selected: File[]) => {
    if (!selected.length || disabled) return
    const validation = validateImageFiles(selected, multiple, maxBytes)
    setError(validation)
    if (!validation) onChange(selected)
  }
  return <div className="image-picker">
    <label className={`file-picker ${dragging ? 'is-dragging' : ''} ${files.length ? 'has-files' : ''} ${disabled ? 'disabled' : ''}`} onDragOver={(event) => { event.preventDefault(); if (!disabled) setDragging(true) }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false) }} onDrop={(event) => { event.preventDefault(); setDragging(false); select(Array.from(event.dataTransfer.files)) }}>
      {preview ? <img className="picker-preview" src={preview} alt="待检测图片预览" /> : <div className="upload-symbol"><Icon name={files.length ? 'check' : 'upload'} /></div>}
      <strong>{dragging ? '松开鼠标，选择图片' : files.length ? `已选择 ${files.length} 张图片` : '将图片拖到这里，或点击选择'}</strong>
      <span>{files.length ? files.slice(0, 3).map((file) => file.name).join('、') : `${multiple ? '支持多选 · ' : ''}JPG / PNG / BMP / TIFF / WebP`}</span>
      <input aria-label={multiple ? '选择原始图片' : '选择检测图片'} aria-describedby={`${id}-hint`} aria-invalid={Boolean(error)} type="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp" multiple={multiple} disabled={disabled} onChange={(event) => { select(Array.from(event.target.files ?? [])); event.target.value = '' }} />
    </label>
    <div className="picker-meta"><small id={`${id}-hint`}>{files.length ? `${(files.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(2)} MB · 点击可重新选择` : `总大小不超过 ${(maxBytes / 1024 / 1024).toFixed(0)} MB，上传后校验图片内容`}</small>{files.length > 0 && <button type="button" className="text-button" disabled={disabled} onClick={() => { onChange([]); setError(null) }}>清空选择</button>}</div>
    {error && <p className="field-error" role="alert">{error}</p>}
  </div>
}
