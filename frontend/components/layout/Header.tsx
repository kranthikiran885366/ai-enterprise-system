'use client';

import { Bell, User, LogOut, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/dashboard': { title: 'Dashboard', subtitle: 'Welcome back to your intelligence hub' },
  '/dashboard/ai': { title: 'AI Assistant', subtitle: 'Multi-agent enterprise AI system' },
  '/dashboard/sales': { title: 'Sales Intelligence', subtitle: 'Pipeline management and AI-powered lead scoring' },
  '/dashboard/finance': { title: 'Financial Intelligence', subtitle: 'Expense analysis, fraud detection, and budget forecasting' },
  '/dashboard/hr': { title: 'HR Intelligence', subtitle: 'Workforce management and AI recruitment' },
  '/dashboard/marketing': { title: 'Marketing Intelligence', subtitle: 'Campaign performance and lead generation' },
  '/dashboard/support': { title: 'Customer Support', subtitle: 'AI-powered ticket management and customer success' },
  '/dashboard/legal': { title: 'Legal & Compliance', subtitle: 'Contract lifecycle management and compliance tracking' },
  '/dashboard/it': { title: 'IT Infrastructure', subtitle: 'Real-time monitoring and incident management' },
  '/dashboard/admin': { title: 'Administration', subtitle: 'User management, RBAC, and system configuration' },
  '/dashboard/settings': { title: 'Settings', subtitle: 'Manage your account and preferences' },
};

export default function Header() {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const { user, logout } = useAuthStore();

  const page = pageTitles[pathname] || { title: 'Dashboard', subtitle: 'Enterprise AI Platform' };

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const notifications = [
    { text: 'TKT-4821: Critical SLA breach — TechCorp Inc.', time: '18m ago', type: 'critical' },
    { text: 'AI Engine CPU at 92% on prod-ai-01', time: '20m ago', type: 'warning' },
    { text: 'New candidate matched for Senior Engineer role', time: '1h ago', type: 'info' },
    { text: 'GDPR compliance action required by Jun 2024', time: '2h ago', type: 'warning' },
  ];

  return (
    <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-40">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">{page.title}</h2>
        <p className="text-sm text-slate-500 mt-0.5">{page.subtitle}</p>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative">
          <button
            onClick={() => { setShowNotifications(!showNotifications); setShowUserMenu(false); }}
            className="relative p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <Bell size={20} />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>

          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden z-50">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-900">Notifications</p>
                <span className="text-xs bg-red-100 text-red-700 font-bold px-2 py-0.5 rounded-full">4 new</span>
              </div>
              {notifications.map((n, i) => (
                <div key={i} className={`px-4 py-3 border-b border-slate-50 hover:bg-slate-50 cursor-pointer ${i === 0 ? 'bg-red-50/50' : ''}`}>
                  <div className="flex items-start gap-2">
                    <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${n.type === 'critical' ? 'bg-red-500' : n.type === 'warning' ? 'bg-amber-500' : 'bg-blue-500'}`} />
                    <div>
                      <p className="text-xs text-slate-800">{n.text}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{n.time}</p>
                    </div>
                  </div>
                </div>
              ))}
              <div className="px-4 py-2.5 text-center">
                <button className="text-xs text-blue-600 hover:underline">View all notifications</button>
              </div>
            </div>
          )}
        </div>

        <div className="relative">
          <button
            onClick={() => { setShowUserMenu(!showUserMenu); setShowNotifications(false); }}
            className="flex items-center gap-2.5 px-3 py-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <User size={16} className="text-white" />
            </div>
            <div className="text-left">
              <p className="text-sm font-semibold text-slate-900">{user?.username || 'Admin'}</p>
              <p className="text-xs text-slate-500">{user?.is_admin ? 'Administrator' : 'User'}</p>
            </div>
            <ChevronDown size={14} className="text-slate-400" />
          </button>

          {showUserMenu && (
            <div className="absolute right-0 mt-2 w-52 bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden z-50">
              <div className="px-4 py-3 border-b border-slate-100">
                <p className="text-sm font-semibold text-slate-900">{user?.username || 'admin'}</p>
                <p className="text-xs text-slate-500">{user?.email || 'admin@enterprise.ai'}</p>
              </div>
              <button
                onClick={() => { setShowUserMenu(false); router.push('/dashboard/settings'); }}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
              >
                <User size={14} />
                Profile & Settings
              </button>
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors border-t border-slate-100"
              >
                <LogOut size={14} />
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
