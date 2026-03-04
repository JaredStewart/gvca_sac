import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Tags,
  BarChart3,
  Compass,
  Table,
  ListChecks,
} from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/data', icon: Table, label: 'Data Table' },
  { to: '/tagging', icon: Tags, label: 'Tagging' },
  { to: '/explore', icon: Compass, label: 'Explore' },
  { to: '/visualizations', icon: BarChart3, label: 'Visualizations' },
  { to: '/batches', icon: ListChecks, label: 'Batches' },
]

export default function Sidebar() {
  return (
    <aside className="w-64 border-r bg-background">
      <nav className="space-y-1 p-4">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )
            }
          >
            <Icon className="h-4 w-4 mr-3" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
