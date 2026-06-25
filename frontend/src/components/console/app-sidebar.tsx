// Primary navigation sidebar for the console.
import {
  Boxes,
  FileCode2,
  LayoutDashboard,
  LineChart,
  ListTree,
  Play,
  Settings2,
  SlidersHorizontal,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
} from '@/components/ui/sidebar'

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
  return (
    <Sidebar collapsible="icon">
      <SidebarContent className="pt-2">
        <NavGroup label="Service" items={NAV} />
        <NavGroup label="Explore" items={EXPLORE} />
      </SidebarContent>

      <SidebarFooter>
        {/* Collapse control sits at the seam with the content (right edge), the
            direction it collapses — matching the SidebarRail convention. */}
        <SidebarTrigger className="self-end text-sidebar-foreground/70 group-data-[collapsible=icon]:self-center" />
      </SidebarFooter>
    </Sidebar>
  )
}
