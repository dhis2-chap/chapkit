import { TerminalSquare } from 'lucide-react'
import { Outlet, Route, Routes } from 'react-router-dom'
import { Toaster } from '@/components/ui/sonner'
import { Separator } from '@/components/ui/separator'
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppSidebar } from '@/components/console/app-sidebar'
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
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background px-3">
      <SidebarTrigger />
      <Separator orientation="vertical" className="mx-1 h-5" />
      <div className="flex items-center gap-2">
        <div className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <TerminalSquare className="size-4" />
        </div>
        <span className="text-base font-semibold tracking-tight">Chapkit</span>
      </div>
      <div className="ml-auto flex items-center gap-1">
        <ThemeToggle />
      </div>
    </header>
  )
}

function Shell() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <TopNav />
        <div className="flex flex-1 flex-col overflow-hidden">
          <Outlet />
        </div>
      </SidebarInset>
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
            <Route path="artifacts" element={<ArtifactsPage />} />
            <Route path="jobs" element={<JobsPage />} />
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
