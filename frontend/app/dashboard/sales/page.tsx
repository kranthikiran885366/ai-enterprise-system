'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { TrendingUp, Target, DollarSign, BarChart3 } from 'lucide-react';

export default function SalesDashboard() {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Sales Intelligence</h1>
        <p className="text-slate-600 mt-2">Real-time sales pipeline analysis with AI forecasting</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Pipeline Value"
          value="$8.2M"
          icon={DollarSign}
          trend="up"
          trendValue="+$1.2M this month"
        />
        <MetricCard
          title="Active Leads"
          value="1,247"
          icon={Target}
          trend="up"
          trendValue="+145 new leads"
        />
        <MetricCard
          title="Avg Deal Size"
          value="$65K"
          icon={BarChart3}
          trend="up"
          trendValue="+8% vs last month"
        />
        <MetricCard
          title="Win Rate"
          value="42%"
          icon={TrendingUp}
          trend="up"
          trendValue="+5% improvement"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Lead Scoring Distribution */}
        <div className="lg:col-span-2 bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Lead Quality Distribution</h2>
          
          <div className="space-y-4">
            {[
              { label: 'Hot Leads (80-100)', count: 127, percentage: 10 },
              { label: 'Warm Leads (60-79)', count: 356, percentage: 29 },
              { label: 'Cold Leads (40-59)', count: 451, percentage: 36 },
              { label: 'Low Quality (<40)', count: 313, percentage: 25 },
            ].map((item, idx) => (
              <div key={idx}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-700">{item.label}</span>
                  <span className="text-sm font-bold text-slate-900">{item.count} leads</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full ${
                      idx === 0
                        ? 'bg-emerald-500'
                        : idx === 1
                        ? 'bg-blue-500'
                        : idx === 2
                        ? 'bg-amber-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${item.percentage}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Churn Risk */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Churn Risk Analysis</h2>
          <div className="space-y-4">
            {[
              { company: 'TechCorp Inc.', risk: 'low', value: '8%', clv: '$2.1M' },
              { company: 'DataFlow LLC', risk: 'medium', value: '42%', clv: '$850K' },
              { company: 'CloudNine Co.', risk: 'high', value: '78%', clv: '$1.2M' },
            ].map((item, idx) => (
              <div key={idx} className="border-b border-slate-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-medium text-slate-900">{item.company}</p>
                    <p className="text-xs text-slate-500 mt-1">CLV: {item.clv}</p>
                  </div>
                  <span
                    className={`text-xs font-bold px-2 py-1 rounded ${
                      item.risk === 'low'
                        ? 'bg-emerald-100 text-emerald-700'
                        : item.risk === 'medium'
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-red-100 text-red-700'
                    }`}
                  >
                    {item.value}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Deal Pipeline */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-6">Deal Pipeline</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { stage: 'Prospecting', deals: 245, value: '$1.2M', forecast: '$800K' },
            { stage: 'Qualification', deals: 156, value: '$3.1M', forecast: '$2.2M' },
            { stage: 'Proposal', deals: 89, value: '$2.4M', forecast: '$1.8M' },
            { stage: 'Negotiation', deals: 42, value: '$1.5M', forecast: '$1.2M' },
          ].map((stage) => (
            <div key={stage.stage} className="border border-slate-200 rounded-lg p-4">
              <h3 className="font-semibold text-slate-900">{stage.stage}</h3>
              <p className="text-2xl font-bold text-blue-600 mt-2">{stage.deals}</p>
              <p className="text-xs text-slate-600 mt-1">deals</p>
              <div className="mt-4 space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-600">Value:</span>
                  <span className="font-semibold text-slate-900">{stage.value}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-600">Forecast:</span>
                  <span className="font-semibold text-slate-900">{stage.forecast}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
