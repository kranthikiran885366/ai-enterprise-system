'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { useState } from 'react';
import { Users, ShieldAlert, Settings, Activity, Key, Plus } from 'lucide-react';

const users = [
  { id: '1', username: 'admin', email: 'admin@enterprise.ai', role: 'admin', status: 'active', lastLogin: '2h ago', department: 'IT' },
  { id: '2', username: 'jsmith', email: 'j.smith@enterprise.ai', role: 'manager', status: 'active', lastLogin: '1h ago', department: 'Sales' },
  { id: '3', username: 'mwilson', email: 'm.wilson@enterprise.ai', role: 'analyst', status: 'active', lastLogin: '4h ago', department: 'Finance' },
  { id: '4', username: 'ktaylor', email: 'k.taylor@enterprise.ai', role: 'user', status: 'active', lastLogin: '1d ago', department: 'HR' },
  { id: '5', username: 'rbrown', email: 'r.brown@enterprise.ai', role: 'manager', status: 'inactive', lastLogin: '7d ago', department: 'Marketing' },
  { id: '6', username: 'ldavis', email: 'l.davis@enterprise.ai', role: 'user', status: 'active', lastLogin: '3h ago', department: 'Support' },
];

const auditLogs = [
  { action: 'User login', user: 'admin', resource: '/dashboard', ip: '192.168.1.1', time: '2h ago', status: 'success' },
  { action: 'Config updated', user: 'admin', resource: '/api/v1/admin/config', ip: '192.168.1.1', time: '3h ago', status: 'success' },
  { action: 'User created', user: 'admin', resource: '/api/v1/admin/users', ip: '192.168.1.1', time: '5h ago', status: 'success' },
  { action: 'Failed login', user: 'unknown', resource: '/auth/login', ip: '203.0.113.42', time: '6h ago', status: 'failed' },
  { action: 'API key generated', user: 'jsmith', resource: '/api/v1/admin/keys', ip: '192.168.1.45', time: '1d ago', status: 'success' },
];

const roleColors: Record<string, string> = {
  admin: 'bg-red-100 text-red-700',
  manager: 'bg-blue-100 text-blue-700',
  analyst: 'bg-purple-100 text-purple-700',
  user: 'bg-slate-100 text-slate-700',
};

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<'users' | 'audit' | 'config'>('users');

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Administration</h1>
        <p className="text-slate-600 mt-2">User management, RBAC, audit logs, and system configuration</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard title="Total Users" value="156" subtitle="Across all roles" icon={Users} trend="up" trendValue="4 new this month" bgColor="bg-blue-50" />
        <MetricCard title="Active Sessions" value="42" subtitle="Right now" icon={Activity} trend="neutral" trendValue="Peak: 89 yesterday" bgColor="bg-emerald-50" />
        <MetricCard title="Security Alerts" value="3" subtitle="Pending review" icon={ShieldAlert} trend="down" trendValue="2 false positives" bgColor="bg-red-50" />
        <MetricCard title="API Keys" value="28" subtitle="Active integrations" icon={Key} trend="up" trendValue="3 expiring soon" bgColor="bg-amber-50" />
        <MetricCard title="Roles & Permissions" value="8" subtitle="Defined roles" icon={Settings} trend="neutral" trendValue="RBAC enabled" bgColor="bg-purple-50" />
        <MetricCard title="Failed Logins" value="12" subtitle="Last 24 hours" icon={ShieldAlert} trend="neutral" trendValue="1 IP blocked" bgColor="bg-rose-50" />
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex gap-2 mb-6 border-b border-slate-200">
          {(['users', 'audit', 'config'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-3 px-4 text-sm font-medium border-b-2 -mb-px transition-colors ${
                activeTab === tab ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              {tab === 'audit' ? 'Audit Log' : tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
          {activeTab === 'users' && (
            <button className="ml-auto mb-3 px-4 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-1">
              <Plus size={14} /> Add User
            </button>
          )}
        </div>

        {activeTab === 'users' && (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">User</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Role</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Department</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Last Login</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Status</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {users.map((u) => (
                  <tr key={u.id} className="hover:bg-slate-50 transition-colors">
                    <td className="py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                          {u.username.charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-900">{u.username}</p>
                          <p className="text-xs text-slate-500">{u.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4">
                      <span className={`text-xs font-bold px-2 py-1 rounded ${roleColors[u.role]}`}>{u.role}</span>
                    </td>
                    <td className="py-4 text-sm text-slate-600">{u.department}</td>
                    <td className="py-4 text-sm text-slate-500">{u.lastLogin}</td>
                    <td className="py-4">
                      <span className={`text-xs font-bold px-2 py-1 rounded ${u.status === 'active' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                        {u.status}
                      </span>
                    </td>
                    <td className="py-4">
                      <button className="text-xs text-blue-600 hover:underline mr-3">Edit</button>
                      <button className="text-xs text-red-500 hover:underline">Disable</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'audit' && (
          <div className="space-y-2">
            {auditLogs.map((log, i) => (
              <div key={i} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
                <div className="flex items-center gap-4">
                  <span className={`text-xs font-bold px-2 py-1 rounded ${log.status === 'success' ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'}`}>
                    {log.status}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-slate-900">{log.action}</p>
                    <p className="text-xs text-slate-500">{log.resource} • IP: {log.ip}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-slate-700">{log.user}</p>
                  <p className="text-xs text-slate-400">{log.time}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'config' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { section: 'Authentication', settings: [
                { key: 'JWT Expiration', value: '30 minutes', editable: true },
                { key: 'MFA Enabled', value: 'Yes', editable: true },
                { key: 'Password Min Length', value: '12 characters', editable: true },
                { key: 'Session Timeout', value: '8 hours', editable: true },
              ]},
              { section: 'Rate Limiting', settings: [
                { key: 'API Rate Limit', value: '100 req/min', editable: true },
                { key: 'Auth Rate Limit', value: '5 req/min', editable: true },
                { key: 'Burst Allowance', value: '20 requests', editable: true },
                { key: 'Block Duration', value: '15 minutes', editable: true },
              ]},
            ].map((section) => (
              <div key={section.section}>
                <h3 className="text-sm font-semibold text-slate-700 mb-3">{section.section}</h3>
                <div className="space-y-3">
                  {section.settings.map((s) => (
                    <div key={s.key} className="flex items-center justify-between py-2.5 border-b border-slate-100 last:border-0">
                      <span className="text-sm text-slate-600">{s.key}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-900">{s.value}</span>
                        {s.editable && <button className="text-xs text-blue-600 hover:underline">Edit</button>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
