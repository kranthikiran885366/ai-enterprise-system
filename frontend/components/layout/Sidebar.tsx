'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  TrendingUp,
  DollarSign,
  Users,
  Megaphone,
  Headphones,
  FileText,
  Server,
  ShieldAlert,
  Settings,
} from 'lucide-react';

const menuItems = [
  {
    label: 'Dashboard',
    icon: BarChart3,
    href: '/dashboard',
  },
  {
    label: 'Sales',
    icon: TrendingUp,
    href: '/dashboard/sales',
  },
  {
    label: 'Finance',
    icon: DollarSign,
    href: '/dashboard/finance',
  },
  {
    label: 'HR',
    icon: Users,
    href: '/dashboard/hr',
  },
  {
    label: 'Marketing',
    icon: Megaphone,
    href: '/dashboard/marketing',
  },
  {
    label: 'Support',
    icon: Headphones,
    href: '/dashboard/support',
  },
  {
    label: 'Legal',
    icon: FileText,
    href: '/dashboard/legal',
  },
  {
    label: 'IT',
    icon: Server,
    href: '/dashboard/it',
  },
  {
    label: 'Admin',
    icon: ShieldAlert,
    href: '/dashboard/admin',
  },
  {
    label: 'Settings',
    icon: Settings,
    href: '/dashboard/settings',
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-900 text-white h-screen overflow-y-auto sticky top-0">
      {/* Logo */}
      <div className="p-6 border-b border-slate-700">
        <h1 className="text-2xl font-bold text-white">Enterprise AI</h1>
        <p className="text-sm text-slate-400 mt-1">Intelligence Platform</p>
      </div>

      {/* Navigation */}
      <nav className="p-6 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-slate-700 bg-slate-900">
        <p className="text-xs text-slate-500">
          AI Enterprise System
        </p>
        <p className="text-xs text-slate-500 mt-1">
          v1.0.0
        </p>
      </div>
    </aside>
  );
}
