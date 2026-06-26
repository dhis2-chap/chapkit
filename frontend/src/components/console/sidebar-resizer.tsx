// Drag/keyboard handle to resize the primary navigation sidebar. The width is
// owned by the Shell (persisted); this only reports new widths.
import { useCallback } from 'react'

import { useSidebar } from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'

export const SIDEBAR_MIN_WIDTH = 200
export const SIDEBAR_MAX_WIDTH = 420
export const SIDEBAR_DEFAULT_WIDTH = 256
const STEP = 16

/** Clamp a candidate sidebar width to the allowed range. */
function clampWidth(width: number): number {
  return Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, width))
}

/** Vertical drag handle on the sidebar's right edge; resizes the expanded desktop sidebar. */
export function SidebarResizer({
  width,
  onWidth,
}: {
  width: number
  onWidth: (width: number) => void
}) {
  const { state, isMobile } = useSidebar()

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      // The sidebar's left edge is the viewport edge, so the cursor x is the width.
      const move = (moveEvent: PointerEvent) => onWidth(clampWidth(moveEvent.clientX))
      const up = () => {
        document.removeEventListener('pointermove', move)
        document.removeEventListener('pointerup', up)
        document.body.style.removeProperty('cursor')
        document.body.style.removeProperty('user-select')
      }
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      document.addEventListener('pointermove', move)
      document.addEventListener('pointerup', up)
    },
    [onWidth],
  )

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft') {
      event.preventDefault()
      onWidth(clampWidth(width - STEP))
    } else if (event.key === 'ArrowRight') {
      event.preventDefault()
      onWidth(clampWidth(width + STEP))
    }
  }

  // Resizing only applies to the expanded desktop sidebar (mobile is a sheet,
  // collapsed is the fixed icon rail).
  if (isMobile || state !== 'expanded') return null

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize sidebar"
      aria-valuenow={width}
      aria-valuemin={SIDEBAR_MIN_WIDTH}
      aria-valuemax={SIDEBAR_MAX_WIDTH}
      tabIndex={0}
      data-slot="sidebar-resizer"
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      className={cn(
        'relative z-20 hidden w-1 shrink-0 cursor-col-resize md:block',
        'after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2',
        'hover:after:bg-sidebar-border focus-visible:after:bg-sidebar-border focus-visible:outline-hidden',
      )}
    />
  )
}
