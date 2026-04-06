'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { Users, Briefcase, TrendingUp, Calendar } from 'lucide-react';

export default function HRDashboard() {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">HR Intelligence</h1>
        <p className="text-slate-600 mt-2">Employee management, recruitment, and talent analytics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Employees"
          value="156"
          subtitle="Active workforce"
          icon={Users}
          trend="up"
          trendValue="4 new hires"
        />
        <MetricCard
          title="Open Positions"
          value="12"
          subtitle="Recruiting"
          icon={Briefcase}
          trend="down"
          trendValue="2 roles filled"
        />
        <MetricCard
          title="Avg Tenure"
          value="4.2 yrs"
          subtitle="Employee retention"
          icon={Calendar}
          trend="up"
          trendValue="+0.3 years"
        />
        <MetricCard
          title="Engagement Score"
          value="82%"
          subtitle="Employee satisfaction"
          icon={TrendingUp}
          trend="up"
          trendValue="+5 points"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recruitment Pipeline */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Recruitment Pipeline</h2>
          <div className="space-y-4">
            {[
              {
                position: 'Senior Engineer',
                stage: 'Offer Review',
                candidates: 2,
                timeOpen: '45 days',
                aiScore: 92,
              },
              {
                position: 'Product Manager',
                stage: 'Interview',
                candidates: 5,
                timeOpen: '32 days',
                aiScore: 78,
              },
              {
                position: 'Data Scientist',
                stage: 'Screening',
                candidates: 18,
                timeOpen: '12 days',
                aiScore: 85,
              },
              {
                position: 'Marketing Manager',
                stage: 'Initial Contact',
                candidates: 34,
                timeOpen: '5 days',
                aiScore: 72,
              },
            ].map((job) => (
              <div key={job.position} className="border-b border-slate-100 pb-4 last:border-0">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-medium text-slate-900">{job.position}</p>
                    <div className="flex gap-4 text-xs text-slate-600 mt-1">
                      <span>{job.stage}</span>
                      <span>{job.candidates} candidates</span>
                      <span>{job.timeOpen} open</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-bold">
                      {job.aiScore}% Match
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Department Breakdown */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Employees by Department</h2>
          <div className="space-y-4">
            {[
              { dept: 'Engineering', count: 45, headcount: 50, utilization: 90 },
              { dept: 'Sales', count: 32, headcount: 35, utilization: 91 },
              { dept: 'Finance', count: 18, headcount: 20, utilization: 90 },
              { dept: 'Marketing', count: 24, headcount: 25, utilization: 96 },
              { dept: 'Operations', count: 22, headcount: 25, utilization: 88 },
              { dept: 'HR', count: 8, headcount: 8, utilization: 100 },
            ].map((dept) => (
              <div key={dept.dept}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-700">{dept.dept}</span>
                  <span className="text-sm font-bold text-slate-900">{dept.count}/{dept.headcount}</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-blue-600 h-2"
                    style={{ width: `${dept.utilization}%` }}
                  ></div>
                </div>
                <p className="text-xs text-slate-500 mt-1">{dept.utilization}% capacity</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Upcoming Events & Reviews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Upcoming Reviews</h2>
          <div className="space-y-3">
            {[
              { employee: 'John Smith', date: 'Jan 20', type: 'Annual', status: 'Scheduled' },
              { employee: 'Sarah Johnson', date: 'Jan 22', type: 'Annual', status: 'Scheduled' },
              { employee: 'Mike Chen', date: 'Jan 25', type: 'Q1 Check-in', status: 'Scheduled' },
              { employee: 'Emma Davis', date: 'Jan 28', type: 'Annual', status: 'Scheduled' },
            ].map((review, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div>
                  <p className="font-medium text-slate-900">{review.employee}</p>
                  <p className="text-xs text-slate-600">{review.type}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-slate-900">{review.date}</p>
                  <p className="text-xs text-emerald-600">{review.status}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Attendance Summary</h2>
          <div className="space-y-4">
            {[
              { metric: 'Average Daily Attendance', value: '94%' },
              { metric: 'On-Time Arrival Rate', value: '97%' },
              { metric: 'Leave Balance Used', value: '62%' },
              { metric: 'Sick Days (Month)', value: '2.3 days avg' },
            ].map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <span className="text-sm text-slate-700">{item.metric}</span>
                <span className="font-bold text-slate-900">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
