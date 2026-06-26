// Responsive master/detail layout: stacked on small screens, drag-resizable
// side-by-side on large ones. The chosen split is persisted per `autoSaveId`.
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'

import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable'

const DESKTOP_QUERY = '(min-width: 1024px)'

/** Track whether the viewport is at the large (lg) breakpoint or above. */
function useIsDesktop(): boolean {
  const [isDesktop, setIsDesktop] = useState(() =>
    typeof window === 'undefined' ? true : window.matchMedia(DESKTOP_QUERY).matches,
  )
  useEffect(() => {
    const mql = window.matchMedia(DESKTOP_QUERY)
    const onChange = () => setIsDesktop(mql.matches)
    mql.addEventListener('change', onChange)
    onChange()
    return () => mql.removeEventListener('change', onChange)
  }, [])
  return isDesktop
}

/** Master list and detail pane, side-by-side and resizable on lg+, stacked below. */
export function MasterDetail({
  master,
  detail,
  autoSaveId,
  defaultSize = 34,
  minSize = 20,
  maxSize = 55,
}: {
  master: ReactNode
  detail: ReactNode
  autoSaveId: string
  defaultSize?: number
  minSize?: number
  maxSize?: number
}) {
  const isDesktop = useIsDesktop()

  if (!isDesktop) {
    return (
      <div className="flex flex-col gap-4">
        {master}
        {detail}
      </div>
    )
  }

  return (
    <ResizablePanelGroup
      direction="horizontal"
      autoSaveId={autoSaveId}
      className="min-h-[calc(100svh-13rem)] items-stretch"
    >
      <ResizablePanel
        defaultSize={defaultSize}
        minSize={minSize}
        maxSize={maxSize}
        className="min-w-0 pr-3"
      >
        {master}
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel minSize={40} className="min-w-0 pl-3">
        {detail}
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
