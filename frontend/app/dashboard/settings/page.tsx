'use client';

import { useState } from 'react';
import { User, Bell, Shield, Key, Globe, Database, Save } from 'lucide-react';

type SettingTab = 'profile' | 'notifications' | 'security' | 'api' | 'integrations';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingTab>('profile');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const tabs: { id: SettingTab; label: string; icon: React.ComponentType<{ size: number }> }[] = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'api', label: 'API Keys', icon: Key },
    { id: 'integrations', label: 'Integrations', icon: Globe },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Settings</h1>
        <p className="text-slate-600 mt-2">Manage your account, preferences, and integrations</p>
      </div>

      <div className="flex gap-8">
        <aside className="w-56 shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                    activeTab === tab.id ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <Icon size={18} />
                  <span className="text-sm font-medium">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </aside>

        <div className="flex-1 bg-white rounded-lg border border-slate-200 p-8">
          {activeTab === 'profile' && (
            <div className="space-y-6 max-w-xl">
              <h2 className="text-lg font-semibold text-slate-900 mb-6">Profile Information</h2>
              <div className="flex items-center gap-6 mb-8">
                <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center text-white text-3xl font-bold">A</div>
                <button className="px-4 py-2 border border-slate-300 text-sm rounded-lg hover:bg-slate-50 transition-colors text-slate-700">Change Avatar</button>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">First Name</label>
                  <input defaultValue="Admin" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Last Name</label>
                  <input defaultValue="User" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
                <input defaultValue="admin@enterprise.ai" type="email" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Job Title</label>
                <input defaultValue="System Administrator" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Department</label>
                <select defaultValue="IT" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {['IT', 'Sales', 'Finance', 'HR', 'Marketing', 'Support', 'Legal'].map((d) => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Timezone</label>
                <select className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option>America/New_York (UTC-5)</option>
                  <option>America/Los_Angeles (UTC-8)</option>
                  <option>Europe/London (UTC+0)</option>
                  <option>Asia/Tokyo (UTC+9)</option>
                </select>
              </div>
            </div>
          )}

          {activeTab === 'notifications' && (
            <div className="space-y-6 max-w-xl">
              <h2 className="text-lg font-semibold text-slate-900 mb-6">Notification Preferences</h2>
              {[
                { category: 'Security & Alerts', items: [
                  { label: 'Security alerts', desc: 'Failed login attempts, suspicious activity', defaultOn: true },
                  { label: 'System incidents', desc: 'Infrastructure alerts and outages', defaultOn: true },
                  { label: 'Compliance warnings', desc: 'Regulatory compliance updates', defaultOn: true },
                ]},
                { category: 'Business', items: [
                  { label: 'Sales pipeline updates', desc: 'New leads, deal stage changes', defaultOn: true },
                  { label: 'Finance alerts', desc: 'Budget exceeded, fraud flags', defaultOn: true },
                  { label: 'HR events', desc: 'New hires, departures, requests', defaultOn: false },
                ]},
                { category: 'AI & Automation', items: [
                  { label: 'AI workflow completions', desc: 'When automated tasks finish', defaultOn: true },
                  { label: 'AI insights ready', desc: 'New analysis and recommendations', defaultOn: false },
                ]},
              ].map((cat) => (
                <div key={cat.category}>
                  <h3 className="text-sm font-semibold text-slate-500 uppercase mb-3">{cat.category}</h3>
                  <div className="space-y-4">
                    {cat.items.map((item) => (
                      <div key={item.label} className="flex items-center justify-between py-2">
                        <div>
                          <p className="text-sm font-medium text-slate-900">{item.label}</p>
                          <p className="text-xs text-slate-500 mt-0.5">{item.desc}</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" defaultChecked={item.defaultOn} className="sr-only peer" />
                          <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'security' && (
            <div className="space-y-8 max-w-xl">
              <h2 className="text-lg font-semibold text-slate-900 mb-6">Security Settings</h2>
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-4">Change Password</h3>
                <div className="space-y-3">
                  <input type="password" placeholder="Current password" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
                  <input type="password" placeholder="New password" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
                  <input type="password" placeholder="Confirm new password" className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
                  <button className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">Update Password</button>
                </div>
              </div>
              <div className="border-t border-slate-100 pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700">Two-Factor Authentication</h3>
                    <p className="text-xs text-slate-500 mt-1">Add an extra layer of security to your account</p>
                  </div>
                  <span className="text-xs bg-emerald-100 text-emerald-700 font-bold px-3 py-1 rounded-full">Enabled</span>
                </div>
              </div>
              <div className="border-t border-slate-100 pt-6">
                <h3 className="text-sm font-semibold text-slate-700 mb-3">Active Sessions</h3>
                {[
                  { device: 'Chrome on MacOS', ip: '192.168.1.1', location: 'New York, US', current: true },
                  { device: 'Safari on iPhone', ip: '10.0.0.45', location: 'New York, US', current: false },
                ].map((session, i) => (
                  <div key={i} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
                    <div>
                      <p className="text-sm font-medium text-slate-900">{session.device} {session.current && <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded ml-1">Current</span>}</p>
                      <p className="text-xs text-slate-500">{session.ip} • {session.location}</p>
                    </div>
                    {!session.current && <button className="text-xs text-red-500 hover:underline">Revoke</button>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'api' && (
            <div className="space-y-6 max-w-2xl">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-slate-900">API Keys</h2>
                <button className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">Generate Key</button>
              </div>
              <div className="space-y-3">
                {[
                  { name: 'Production API Key', key: 'ent_prod_••••••••••••8f2a', created: 'Jan 15, 2024', lastUsed: '2h ago', scopes: ['read', 'write'] },
                  { name: 'Analytics Integration', key: 'ent_int_••••••••••••3d9c', created: 'Mar 1, 2024', lastUsed: '1d ago', scopes: ['read'] },
                  { name: 'Webhook Secret', key: 'ent_wh_••••••••••••7b1e', created: 'Feb 20, 2024', lastUsed: 'Never', scopes: ['webhook'] },
                ].map((key) => (
                  <div key={key.name} className="p-4 border border-slate-200 rounded-lg">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{key.name}</p>
                        <p className="text-xs text-slate-500 mt-0.5">Created {key.created} • Last used: {key.lastUsed}</p>
                      </div>
                      <button className="text-xs text-red-500 hover:underline">Revoke</button>
                    </div>
                    <div className="flex items-center gap-3">
                      <code className="flex-1 text-xs bg-slate-50 border border-slate-200 px-3 py-2 rounded font-mono text-slate-700">{key.key}</code>
                      <button className="text-xs text-blue-600 hover:underline">Copy</button>
                    </div>
                    <div className="flex gap-2 mt-2">
                      {key.scopes.map((s) => <span key={s} className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded">{s}</span>)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'integrations' && (
            <div className="space-y-4 max-w-2xl">
              <h2 className="text-lg font-semibold text-slate-900 mb-6">Connected Integrations</h2>
              {[
                { name: 'Salesforce CRM', status: 'connected', icon: '💼', desc: 'Sync leads and opportunities bidirectionally' },
                { name: 'Slack', status: 'connected', icon: '💬', desc: 'Real-time alerts and workflow notifications' },
                { name: 'HubSpot', status: 'disconnected', icon: '🎯', desc: 'Marketing automation and contact sync' },
                { name: 'Stripe', status: 'connected', icon: '💳', desc: 'Payment processing and invoice management' },
                { name: 'SendGrid', status: 'connected', icon: '📧', desc: 'Transactional email and marketing campaigns' },
                { name: 'Jira', status: 'disconnected', icon: '🔧', desc: 'IT issue tracking and sprint management' },
              ].map((integration) => (
                <div key={integration.name} className="flex items-center justify-between p-4 border border-slate-200 rounded-lg">
                  <div className="flex items-center gap-4">
                    <span className="text-2xl">{integration.icon}</span>
                    <div>
                      <p className="text-sm font-semibold text-slate-900">{integration.name}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{integration.desc}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${integration.status === 'connected' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                      {integration.status}
                    </span>
                    <button className="text-xs text-blue-600 hover:underline">
                      {integration.status === 'connected' ? 'Configure' : 'Connect'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="mt-8 pt-6 border-t border-slate-100 flex items-center gap-3">
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Save size={16} />
              Save Changes
            </button>
            {saved && <span className="text-sm text-emerald-600 font-medium">✓ Saved successfully</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
