import { useOutletContext } from 'react-router-dom'
import type { Health } from './types'

export type WorkbenchContext = {
  health: Health | null
  busy: string | null
  perform: <T>(label: string, action: () => Promise<T>) => Promise<T | undefined>
  notify: (message: string) => void
  openCreateDataset: () => void
}

export function useWorkbench(): WorkbenchContext {
  return useOutletContext<WorkbenchContext>()
}
