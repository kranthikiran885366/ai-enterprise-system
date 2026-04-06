'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { DollarSign, TrendingUp, AlertCircle, PieChart } from 'lucide-react';

export default function FinanceDashboard() {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Financial Intelligence</h1>
        <p className="text-slate-600 mt-2">Expense analysis, fraud detection, and budget forecasting</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Expenses"
          value="$487K"
          subtitle="This month"
          icon={DollarSign}
          trend="up"
          trendValue="+12% vs last month"
        />
        <MetricCard
          title="Budget Utilization"
          value="73%"
          subtitle="Current quarter"
          icon={PieChart}
          trend="up"
          trendValue="+5% this week"
        />
        <MetricCard
          title="Fraud Alerts"
          value="8"
          subtitle="Pending review"
          icon={AlertCircle}
          trend="down"
          trendValue="3 resolved today"
        />
        <MetricCard
          title="Cash Flow"
          value="$2.1M"
          subtitle="Projected (30 days)"
          icon={TrendingUp}
          trend="up"
          trendValue="+$200K improvement"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Budget vs Actual */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Budget vs Actual</h2>
          <div className="space-y-5">
            {[
              { category: 'Payroll', budget: 250000, actual: 245000, percentage: 98 },
              { category: 'Operations', budget: 80000, actual: 72500, percentage: 91 },
              { category: 'Marketing', budget: 50000, actual: 58000, percentage: 116 },
              { category: 'Technology', budget: 40000, actual: 35200, percentage: 88 },
              { category: 'Travel', budget: 30000, actual: 34800, percentage: 116 },
            ].map((item) => (
              <div key={item.category}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-700">{item.category}</span>
                  <span className={`text-sm font-bold ${item.percentage > 100 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {item.percentage}%
                  </span>
                </div>
                <div className="flex gap-2">
                  <div className="flex-1 bg-blue-100 h-3 rounded-full overflow-hidden">
                    <div
                      className="bg-blue-600 h-full"
                      style={{ width: '100%' }}
                    ></div>
                  </div>
                  <div className="flex-1 bg-slate-200 h-3 rounded-full overflow-hidden">
                    <div
                      className={item.percentage > 100 ? 'bg-red-500' : 'bg-emerald-500'}
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Budget: ${(item.budget / 1000).toFixed(0)}K</span>
                  <span>Actual: ${(item.actual / 1000).toFixed(0)}K</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Fraud Risk Analysis */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Flagged Expenses</h2>
          <div className="space-y-3">
            {[
              {
                vendor: 'TechSupplies Inc',
                amount: '$8,500',
                reason: 'Unusually high amount',
                risk: 'high',
              },
              {
                vendor: 'CloudServices Co',
                amount: '$3,200',
                reason: 'Duplicate transaction',
                risk: 'medium',
              },
              {
                vendor: 'Office Furniture',
                amount: '$12,400',
                reason: 'Exceeds approval limit',
                risk: 'high',
              },
              {
                vendor: 'Travel Booking',
                amount: '$2,100',
                reason: 'Off-policy vendor',
                risk: 'medium',
              },
              {
                vendor: 'Supplies Store',
                amount: '$450',
                reason: 'Weekend submission',
                risk: 'low',
              },
            ].map((expense, idx) => (
              <div key={idx} className="flex items-start justify-between p-3 border border-slate-100 rounded-lg">
                <div className="flex-1">
                  <p className="font-medium text-slate-900">{expense.vendor}</p>
                  <p className="text-xs text-slate-600 mt-1">{expense.reason}</p>
                </div>
                <div className="text-right">
                  <p className="font-bold text-slate-900">{expense.amount}</p>
                  <span
                    className={`inline-block text-xs font-bold px-2 py-1 mt-1 rounded ${
                      expense.risk === 'high'
                        ? 'bg-red-100 text-red-700'
                        : expense.risk === 'medium'
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-emerald-100 text-emerald-700'
                    }`}
                  >
                    {expense.risk}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cash Flow Forecast */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-6">30-Day Cash Flow Forecast</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { label: 'Current Balance', value: '$2.4M', icon: '💰', color: 'blue' },
            { label: 'Expected Inflows', value: '$850K', icon: '📈', color: 'emerald' },
            { label: 'Expected Outflows', value: '$620K', icon: '📉', color: 'slate' },
            { label: 'Projected Balance', value: '$2.63M', icon: '🎯', color: 'blue' },
            { label: 'Confidence Score', value: '87%', icon: '✓', color: 'emerald' },
            { label: 'Risk Level', value: 'Low', icon: '⚠️', color: 'emerald' },
          ].map((item, idx) => (
            <div key={idx} className="border border-slate-200 rounded-lg p-4 bg-slate-50">
              <p className="text-sm text-slate-600">{item.label}</p>
              <p className="text-2xl font-bold text-slate-900 mt-2">{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
