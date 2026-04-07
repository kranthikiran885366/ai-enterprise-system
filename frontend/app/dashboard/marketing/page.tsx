'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { Megaphone, TrendingUp, Users, MousePointer, Mail, Target } from 'lucide-react';

const campaigns = [
  { name: 'Q1 Product Launch', status: 'active', channel: 'Multi-channel', budget: '$45,000', spent: '$32,400', leads: 847, conversion: '3.2%' },
  { name: 'Email Re-engagement', status: 'active', channel: 'Email', budget: '$8,000', spent: '$5,200', leads: 312, conversion: '4.8%' },
  { name: 'LinkedIn B2B', status: 'paused', channel: 'Social', budget: '$22,000', spent: '$18,700', leads: 156, conversion: '2.1%' },
  { name: 'Google Ads - SaaS', status: 'active', channel: 'PPC', budget: '$30,000', spent: '$27,100', leads: 634, conversion: '5.6%' },
  { name: 'Content Marketing', status: 'active', channel: 'SEO', budget: '$12,000', spent: '$9,800', leads: 420, conversion: '2.9%' },
];

const channels = [
  { name: 'Organic Search', leads: 1243, percentage: 34, color: 'bg-blue-500' },
  { name: 'Paid Ads', leads: 892, percentage: 24, color: 'bg-emerald-500' },
  { name: 'Email', leads: 634, percentage: 17, color: 'bg-amber-500' },
  { name: 'Social Media', leads: 521, percentage: 14, color: 'bg-purple-500' },
  { name: 'Referral', leads: 289, percentage: 8, color: 'bg-rose-500' },
  { name: 'Direct', leads: 112, percentage: 3, color: 'bg-slate-400' },
];

export default function MarketingPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Marketing Intelligence</h1>
        <p className="text-slate-600 mt-2">Campaign performance, lead generation, and audience analytics</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard title="Total Leads" value="3,691" subtitle="This quarter" icon={Users} trend="up" trendValue="+18% vs last quarter" bgColor="bg-blue-50" />
        <MetricCard title="Conversion Rate" value="3.8%" subtitle="Lead to opportunity" icon={Target} trend="up" trendValue="+0.4% this month" bgColor="bg-emerald-50" />
        <MetricCard title="Campaign Budget" value="$117K" subtitle="Total allocated" icon={Megaphone} trend="neutral" trendValue="$93.2K spent" bgColor="bg-amber-50" />
        <MetricCard title="Email Open Rate" value="28.4%" subtitle="Avg across campaigns" icon={Mail} trend="up" trendValue="+2.1% this month" bgColor="bg-purple-50" />
        <MetricCard title="Click-through Rate" value="4.7%" subtitle="Paid campaigns" icon={MousePointer} trend="down" trendValue="-0.3% this week" bgColor="bg-rose-50" />
        <MetricCard title="Revenue Attributed" value="$1.2M" subtitle="Marketing sourced" icon={TrendingUp} trend="up" trendValue="+24% YoY" bgColor="bg-blue-50" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-slate-900">Active Campaigns</h2>
            <button className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
              New Campaign
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Campaign</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Channel</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Leads</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Conv.</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Budget</th>
                  <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {campaigns.map((c) => (
                  <tr key={c.name} className="hover:bg-slate-50 transition-colors">
                    <td className="py-4 text-sm font-medium text-slate-900">{c.name}</td>
                    <td className="py-4 text-sm text-slate-600">{c.channel}</td>
                    <td className="py-4 text-sm font-semibold text-slate-900">{c.leads.toLocaleString()}</td>
                    <td className="py-4 text-sm font-semibold text-blue-600">{c.conversion}</td>
                    <td className="py-4 text-sm text-slate-600">
                      <div>{c.spent} / {c.budget}</div>
                    </td>
                    <td className="py-4">
                      <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                        c.status === 'active' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                      }`}>{c.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Lead Sources</h2>
          <div className="space-y-4">
            {channels.map((ch) => (
              <div key={ch.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-700 font-medium">{ch.name}</span>
                  <span className="text-slate-500">{ch.leads.toLocaleString()}</span>
                </div>
                <div className="bg-slate-100 h-2.5 rounded-full overflow-hidden">
                  <div className={`${ch.color} h-full rounded-full`} style={{ width: `${ch.percentage}%` }} />
                </div>
                <p className="text-xs text-slate-500 mt-1">{ch.percentage}% of total</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Email Performance</h2>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'Sent', value: '124,500', sub: 'This month' },
              { label: 'Delivered', value: '121,320', sub: '97.4% rate' },
              { label: 'Opened', value: '34,454', sub: '28.4% rate' },
              { label: 'Clicked', value: '5,721', sub: '4.7% rate' },
              { label: 'Bounced', value: '3,180', sub: '2.6% rate' },
              { label: 'Unsubscribed', value: '248', sub: '0.2% rate' },
            ].map((item) => (
              <div key={item.label} className="bg-slate-50 rounded-lg p-4">
                <p className="text-xs text-slate-500">{item.label}</p>
                <p className="text-xl font-bold text-slate-900 mt-1">{item.value}</p>
                <p className="text-xs text-slate-400 mt-1">{item.sub}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Top Performing Content</h2>
          <div className="space-y-3">
            {[
              { title: 'AI Automation Whitepaper', type: 'Whitepaper', views: 4821, leads: 312 },
              { title: 'Enterprise ROI Calculator', type: 'Tool', views: 3654, leads: 287 },
              { title: '2024 SaaS Benchmark Report', type: 'Report', views: 2980, leads: 198 },
              { title: 'How to Scale with AI', type: 'Blog', views: 8120, leads: 156 },
              { title: 'Product Demo Webinar', type: 'Webinar', views: 1240, leads: 142 },
            ].map((item) => (
              <div key={item.title} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
                <div>
                  <p className="text-sm font-medium text-slate-900">{item.title}</p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">{item.type}</span>
                    <span className="text-xs text-slate-500">{item.views.toLocaleString()} views</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold text-emerald-600">{item.leads}</p>
                  <p className="text-xs text-slate-500">leads</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
