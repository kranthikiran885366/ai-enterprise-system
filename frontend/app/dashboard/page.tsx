'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import {
  TrendingUp,
  DollarSign,
  Users,
  AlertCircle,
  Activity,
  Target,
} from 'lucide-react';

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg p-8 text-white">
        <h1 className="text-3xl font-bold">Welcome to Enterprise Intelligence</h1>
        <p className="text-blue-100 mt-2">
          Real-time insights across all business operations powered by AI
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard
          title="Active Leads"
          value="1,247"
          subtitle="Sales Pipeline"
          icon={Target}
          trend="up"
          trendValue="12% from last month"
          bgColor="bg-blue-50"
        />
        <MetricCard
          title="Total Revenue"
          value="$2.5M"
          subtitle="This quarter"
          icon={DollarSign}
          trend="up"
          trendValue="8% growth"
          bgColor="bg-emerald-50"
        />
        <MetricCard
          title="Team Members"
          value="156"
          subtitle="Across all departments"
          icon={Users}
          trend="up"
          trendValue="4 new hires"
          bgColor="bg-amber-50"
        />
        <MetricCard
          title="Fraud Risk Alerts"
          value="8"
          subtitle="Flagged for review"
          icon={AlertCircle}
          trend="down"
          trendValue="3 resolved today"
          bgColor="bg-red-50"
        />
        <MetricCard
          title="System Uptime"
          value="99.8%"
          subtitle="Last 30 days"
          icon={Activity}
          trend="neutral"
          trendValue="0 incidents"
          bgColor="bg-emerald-50"
        />
        <MetricCard
          title="Support Tickets"
          value="43"
          subtitle="Open requests"
          icon={TrendingUp}
          trend="down"
          trendValue="12 resolved today"
          bgColor="bg-blue-50"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity */}
        <div className="lg:col-span-2 bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-slate-900">Recent Activity</h2>
            <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
              View All
            </button>
          </div>
          <div className="space-y-4">
            {[
              {
                title: 'Lead score updated',
                description: 'TechCorp Inc. reached 92/100 score',
                time: '2 hours ago',
                type: 'sales',
              },
              {
                title: 'Expense flagged',
                description: 'Unusual purchase detected - requires review',
                time: '3 hours ago',
                type: 'finance',
              },
              {
                title: 'Candidate matched',
                description: 'New candidate profile matches Senior Engineer role',
                time: '5 hours ago',
                type: 'hr',
              },
              {
                title: 'Campaign launched',
                description: 'Q1 product launch campaign activated',
                time: '1 day ago',
                type: 'marketing',
              },
            ].map((activity, idx) => (
              <div
                key={idx}
                className="flex items-start gap-4 pb-4 border-b border-slate-100 last:border-0 last:pb-0"
              >
                <div className="w-2 h-2 rounded-full bg-blue-600 mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-900">{activity.title}</p>
                  <p className="text-sm text-slate-600">{activity.description}</p>
                  <p className="text-xs text-slate-500 mt-1">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="space-y-4">
          {/* Sales Performance */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-900 mb-4">Sales Performance</h3>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-slate-600">Pipeline Forecast</span>
                  <span className="text-sm font-bold text-slate-900">$4.2M</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '68%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-slate-600">Close Probability</span>
                  <span className="text-sm font-bold text-slate-900">42%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div className="bg-amber-600 h-2 rounded-full" style={{ width: '42%' }}></div>
                </div>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-900 mb-4">System Status</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">API Gateway</span>
                <span className="inline-block w-2 h-2 bg-emerald-500 rounded-full"></span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">Database</span>
                <span className="inline-block w-2 h-2 bg-emerald-500 rounded-full"></span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">AI Services</span>
                <span className="inline-block w-2 h-2 bg-emerald-500 rounded-full"></span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">Monitoring</span>
                <span className="inline-block w-2 h-2 bg-emerald-500 rounded-full"></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
