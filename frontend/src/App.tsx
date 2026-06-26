import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { BookOpen, ExternalLink, TerminalSquare } from 'lucide-react'
import { Outlet, Route, Routes } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Toaster } from '@/components/ui/sonner'
import { Separator } from '@/components/ui/separator'
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppSidebar } from '@/components/console/app-sidebar'
import {
  SidebarResizer,
  SIDEBAR_DEFAULT_WIDTH,
} from '@/components/console/sidebar-resizer'
import { ThemeProvider, ThemeToggle } from '@/components/console/theme'
import { OverviewPage } from '@/pages/OverviewPage'
import { ConfigsPage } from '@/pages/ConfigsPage'
import { ArtifactsPage } from '@/pages/ArtifactsPage'
import { JobsPage } from '@/pages/JobsPage'
import { TrainPage } from '@/pages/TrainPage'
import { PredictPage } from '@/pages/PredictPage'
import { EndpointsPage } from '@/pages/EndpointsPage'
import { SystemPage } from '@/pages/SystemPage'

function TopNav() {
  const { data: info } = useQuery({ queryKey: ['info'], queryFn: api.info })
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background px-3">
      <SidebarTrigger className="md:hidden" />
      <div className="flex items-center gap-2">
        <TerminalSquare className="size-5 text-primary" />
        <span className="text-base font-semibold tracking-tight">Chapkit</span>
      </div>
      {info ? (
        <>
          <Separator orientation="vertical" className="h-6" />
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="truncate font-medium">{info.display_name}</span>
            <span className="shrink-0 text-xs text-muted-foreground">
              v{info.version}
            </span>
          </div>
        </>
      ) : null}
      <div className="ml-auto flex items-center gap-1">
        <Button variant="ghost" size="sm" asChild>
          <a href="docs" target="_blank" rel="noreferrer">
            <BookOpen className="size-4" />
            API Docs
            <ExternalLink className="size-3.5" />
          </a>
        </Button>
        <ThemeToggle />
      </div>
    </header>
  )
}

// Full-width top nav above the sidebar; the sidebar and content fill the
// remaining viewport height below it. SidebarProvider is the column root so the
// trigger in the top nav keeps its context, and the fixed sidebar is offset
// below the 3.5rem header.
const SIDEBAR_WIDTH_STORAGE_KEY = 'console:sidebar-width'

function Shell() {
  // The sidebar width is user-resizable and persisted across sessions.
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (typeof window === 'undefined') return SIDEBAR_DEFAULT_WIDTH
    const saved = Number(window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY))
    return Number.isFinite(saved) && saved > 0 ? saved : SIDEBAR_DEFAULT_WIDTH
  })
  useEffect(() => {
    window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidth))
  }, [sidebarWidth])

  return (
    <SidebarProvider
      className="!h-svh flex-col [&_[data-slot=sidebar-container]]:!top-14 [&_[data-slot=sidebar-container]]:!h-[calc(100svh-3.5rem)] [&_[data-slot=sidebar-gap]]:!h-[calc(100svh-3.5rem)]"
      style={{ '--sidebar-width': `${sidebarWidth}px` } as CSSProperties}
    >
      <TopNav />
      <div className="flex min-h-0 w-full flex-1">
        <AppSidebar />
        <SidebarResizer width={sidebarWidth} onWidth={setSidebarWidth} />
        <SidebarInset className="min-h-0 overflow-hidden">
          <Outlet />
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <TooltipProvider delayDuration={300}>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<OverviewPage />} />
            <Route path="configs" element={<ConfigsPage />} />
            <Route path="configs/:configId" element={<ConfigsPage />} />
            <Route path="artifacts" element={<ArtifactsPage />} />
            <Route path="artifacts/:artifactId" element={<ArtifactsPage />} />
            <Route path="jobs" element={<JobsPage />} />
            <Route path="jobs/:jobId" element={<JobsPage />} />
            <Route path="train" element={<TrainPage />} />
            <Route path="predict" element={<PredictPage />} />
            <Route path="api" element={<EndpointsPage />} />
            <Route path="system" element={<SystemPage />} />
          </Route>
        </Routes>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    </ThemeProvider>
  )
}
