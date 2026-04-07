'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { useState } from 'react';
import { Headphones, Clock, CheckCircle, AlertCircle, Star, MessageSquare } from 'lucide-react';

const tickets = [
  { id: 'TKT-4821', subject: 'API integration failing on production', priority: 'critical', status: 'open', customer: 'TechCorp Inc.', agent: 'Sarah M.', created: '2h ago', sla: '1h overdue' },
  { id: 'TKT-4820', subject: 'Cannot export reports to PDF', priority: 'high', status: 'in_progress', customer: 'DataFlow Ltd.', agent: 'James R.', created: '4h ago', sla: '2h remaining' },
  { id: 'TKT-4819', subject: 'Billing discrepancy for March invoice', priority: 'medium', status: 'in_progress', customer: 'GlobalOps Co.', agent: 'Maria K.', created: '6h ago', sla: '8h remaining' },
  { id: 'TKT-4818', subject: 'Dashboard loading slowly', priority: 'low', status: 'open', customer: 'StartupXYZ', agent: 'Unassigned', created: '1d ago', sla: '2d remaining' },
  { id: 'TKT-4817', subject: 'Password reset email not received', priority: 'medium', status: 'resolved', customer: 'MegaCorp', agent: 'Tom H.', created: '2d ago', sla: 'Resolved' },
  { id: 'TKT-4816', subject: 'Custom field not saving in CRM', priority: 'high', status: 'resolved', customer: 'SalesForce Pro', agent: 'Sarah M.', created: '2d ago', sla: 'Resolved' },
];

const priorityColors: Record<string, string> = {
  critical: 'bg-red-100 text-red-700',
  high: 'bg-orange-100 text-orange-700',
  medium: 'bg-amber-100 text-amber-700',
  low: 'bg-slate-100 text-slate-700',
};

const statusColors: Record<string, string> = {
  open: 'bg-blue-100 text-blue-700',
  in_progress: 'bg-amber-100 text-amber-700',
  resolved: 'bg-emerald-100 text-emerald-700',
};

export default function SupportPage() {
  const [activeTab, setActiveTab] = useState<'all' | 'open' | 'in_progress' | 'resolved'>('all');

  const filtered = activeTab === 'all' ? tickets : tickets.filter((t) => t.status === activeTab);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Customer Support</h1>
        <p className="text-slate-600 mt-2">AI-powered ticket management and customer success platform</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard title="Open Tickets" value="43" subtitle="Requires attention" icon={AlertCircle} trend="down" trendValue="12 resolved today" bgColor="bg-red-50" />
        <MetricCard title="Avg Resolution" value="4.2h" subtitle="This week" icon={Clock} trend="down" trendValue="-0.8h vs last week" bgColor="bg-blue-50" />
        <MetricCard title="CSAT Score" value="4.7/5" subtitle="Last 30 days" icon={Star} trend="up" trendValue="+0.2 this month" bgColor="bg-amber-50" />
        <MetricCard title="First Response" value="12m" subtitle="Avg time" icon={MessageSquare} trend="up" trendValue="SLA: 95.8% met" bgColor="bg-emerald-50" />
        <MetricCard title="AI Resolved" value="31%" subtitle="Auto-resolved by AI" icon={CheckCircle} trend="up" trendValue="+5% this month" bgColor="bg-purple-50" />
        <MetricCard title="Escalations" value="7" subtitle="This week" icon={Headphones} trend="down" trendValue="3 pending review" bgColor="bg-rose-50" />
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Support Queue</h2>
          <button className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
            Create Ticket
          </button>
        </div>

        <div className="flex gap-2 mb-6 border-b border-slate-200">
          {(['all', 'open', 'in_progress', 'resolved'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-3 px-4 text-sm font-medium border-b-2 -mb-px transition-colors ${
                activeTab === tab ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              {tab === 'in_progress' ? 'In Progress' : tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Ticket</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Priority</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Status</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Customer</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Agent</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">SLA</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {filtered.map((ticket) => (
                <tr key={ticket.id} className="hover:bg-slate-50 transition-colors cursor-pointer">
                  <td className="py-4">
                    <p className="text-sm font-mono text-blue-600">{ticket.id}</p>
                    <p className="text-sm text-slate-900 font-medium mt-0.5">{ticket.subject}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{ticket.created}</p>
                  </td>
                  <td className="py-4">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${priorityColors[ticket.priority]}`}>
                      {ticket.priority}
                    </span>
                  </td>
                  <td className="py-4">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${statusColors[ticket.status]}`}>
                      {ticket.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="py-4 text-sm text-slate-700">{ticket.customer}</td>
                  <td className="py-4 text-sm text-slate-700">{ticket.agent}</td>
                  <td className="py-4">
                    <span className={`text-xs font-medium ${ticket.sla.includes('overdue') ? 'text-red-600' : 'text-slate-600'}`}>
                      {ticket.sla}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Agent Performance</h2>
          <div className="space-y-4">
            {[
              { name: 'Sarah M.', resolved: 48, csat: 4.9, avgTime: '3.2h' },
              { name: 'James R.', resolved: 41, csat: 4.7, avgTime: '4.1h' },
              { name: 'Maria K.', resolved: 37, csat: 4.8, avgTime: '3.8h' },
              { name: 'Tom H.', resolved: 33, csat: 4.6, avgTime: '4.5h' },
            ].map((agent) => (
              <div key={agent.name} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {agent.name.charAt(0)}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-900">{agent.name}</p>
                    <p className="text-xs text-slate-500">{agent.resolved} resolved this week</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold text-amber-500">★ {agent.csat}</p>
                  <p className="text-xs text-slate-500">{agent.avgTime} avg</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">AI Insights</h2>
          <div className="space-y-4">
            {[
              { insight: 'API-related tickets increased 28% this week. Root cause: rate limit changes in v2.4.1', severity: 'high' },
              { insight: '15 customers asking about PDF export — suggest creating knowledge base article', severity: 'medium' },
              { insight: 'Predict 12 escalations before end of week based on current SLA patterns', severity: 'high' },
              { insight: 'TechCorp Inc. sentiment score dropped to 2.1/5 — proactive outreach recommended', severity: 'critical' },
            ].map((item, i) => (
              <div key={i} className={`p-4 rounded-lg border ${
                item.severity === 'critical' ? 'bg-red-50 border-red-200' :
                item.severity === 'high' ? 'bg-orange-50 border-orange-200' : 'bg-blue-50 border-blue-200'
              }`}>
                <p className="text-sm text-slate-800">{item.insight}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
