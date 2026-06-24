// Primary navigation sidebar for the console.
import {
  Boxes,
  BookOpen,
  FileCode2,
  LayoutDashboard,
  LineChart,
  ListTree,
  Play,
  Settings2,
  SlidersHorizontal,
  TerminalSquare,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'
import { api } from '@/lib/api'

interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  end?: boolean
}

const NAV: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/configs', label: 'Configs', icon: SlidersHorizontal },
  { to: '/artifacts', label: 'Artifacts', icon: ListTree },
  { to: '/jobs', label: 'Jobs', icon: Boxes },
  { to: '/train', label: 'Train', icon: Play },
  { to: '/predict', label: 'Predict', icon: LineChart },
]

const EXPLORE: NavItem[] = [
  { to: '/api', label: 'Endpoints', icon: FileCode2 },
  { to: '/system', label: 'System', icon: Settings2 },
]

function NavGroup({ label, items }: { label: string; items: NavItem[] }) {
  const { pathname } = useLocation()
  return (
    <SidebarGroup>
      <SidebarGroupLabel>{label}</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => {
            const active = item.end
              ? pathname === item.to
              : pathname === item.to || pathname.startsWith(`${item.to}/`)
            return (
              <SidebarMenuItem key={item.to}>
                <SidebarMenuButton asChild isActive={active} tooltip={item.label}>
                  <Link to={item.to}>
                    <item.icon className="size-4" />
                    <span>{item.label}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            )
          })}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}

export function AppSidebar() {
  const { data: info } = useQuery({ queryKey: ['info'], queryFn: api.info })

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-1.5">
          <div className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <TerminalSquare className="size-4" />
          </div>
          <div className="grid leading-tight">
            <span className="truncate text-sm font-semibold">
              {info?.display_name ?? 'Chapkit'}
            </span>
            <span className="truncate text-xs text-muted-foreground">
              {info ? `v${info.version}` : 'Console'}
            </span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <NavGroup label="Service" items={NAV} />
        <NavGroup label="Explore" items={EXPLORE} />
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Swagger API docs">
              <a href="docs" target="_blank" rel="noreferrer">
                <BookOpen className="size-4" />
                <span>API Docs</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
