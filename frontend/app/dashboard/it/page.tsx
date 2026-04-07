'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { Server, Activity, AlertTriangle, CheckCircle, Cpu, HardDrive } from 'lucide-react';

const servers = [
  { name: 'prod-api-01', type: 'API Server', cpu: 68, memory: 72, disk: 45, status: 'healthy', region: 'us-east-1' },
  { name: 'prod-api-02', type: 'API Server', cpu: 71, memory: 68, disk: 43, status: 'healthy', region: 'us-east-1' },
  { name: 'prod-db-01', type: 'Database', cpu: 42, memory: 85, disk: 67, status: 'warning', region: 'us-east-1' },
  { name: 'prod-cache-01', type: 'Redis Cache', cpu: 28, memory: 54, disk: 22, status: 'healthy', region: 'us-west-2' },
  { name: 'prod-ai-01', type: 'AI Engine', cpu: 92, memory: 88, disk: 51, status: 'critical', region: 'us-east-1' },
  { name: 'staging-api-01', type: 'API Server', cpu: 15, memory: 32, disk: 28, status: 'healthy', region: 'eu-west-1' },
];

const incidents = [
  { id: 'INC-0421', title: 'AI Engine CPU Spike - prod-ai-01', severity: 'critical', status: 'investigating', started: '18m ago', assignee: 'DevOps Team' },
  { id: 'INC-0420', title: 'Database memory usage above 85%', severity: 'warning', status: 'monitoring', started: '2h ago', assignee: 'Alex T.' },
  { id: 'INC-0419', title: 'SSL certificate expiring in 14 days', severity: 'warning', status: 'scheduled', started: '1d ago', assignee: 'Sarah K.' },
  { id: 'INC-0418', title: 'CDN cache miss rate increased to 34%', severity: 'info', status: 'resolved', started: '2d ago', assignee: 'Mark R.' },
];

const severityColors: Record<string, string> = {
  critical: 'bg-red-100 text-red-700',
  warning: 'bg-amber-100 text-amber-700',
  info: 'bg-blue-100 text-blue-700',
};

const statusColors: Record<string, string> = {
  investigating: 'bg-red-100 text-red-700',
  monitoring: 'bg-amber-100 text-amber-700',
  scheduled: 'bg-blue-100 text-blue-700',
  resolved: 'bg-emerald-100 text-emerald-700',
};

function UsageBar({ value, warning = 75, critical = 90 }: { value: number; warning?: number; critical?: number }) {
  const color = value >= critical ? 'bg-red-500' : value >= warning ? 'bg-amber-500' : 'bg-emerald-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-slate-100 h-2 rounded-full overflow-hidden">
        <div className={`${color} h-full rounded-full`} style={{ width: `${value}%` }} />
      </div>
      <span className={`text-xs font-semibold w-8 text-right ${value >= critical ? 'text-red-600' : value >= warning ? 'text-amber-600' : 'text-emerald-600'}`}>{value}%</span>
    </div>
  );
}

export default function ITPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">IT Infrastructure</h1>
        <p className="text-slate-600 mt-2">Real-time infrastructure monitoring, incident management, and capacity planning</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard title="System Uptime" value="99.8%" subtitle="Last 30 days" icon={Activity} trend="neutral" trendValue="0 P1 incidents" bgColor="bg-emerald-50" />
        <MetricCard title="Active Incidents" value="2" subtitle="Under investigation" icon={AlertTriangle} trend="down" trendValue="1 critical, 1 warning" bgColor="bg-red-50" />
        <MetricCard title="Servers Monitored" value="48" subtitle="Across 3 regions" icon={Server} trend="neutral" trendValue="46 healthy, 2 degraded" bgColor="bg-blue-50" />
        <MetricCard title="Avg CPU Usage" value="52%" subtitle="Production fleet" icon={Cpu} trend="up" trendValue="+8% this week" bgColor="bg-amber-50" />
        <MetricCard title="Storage Used" value="67%" subtitle="Total capacity" icon={HardDrive} trend="up" trendValue="Need 2TB in 30 days" bgColor="bg-purple-50" />
        <MetricCard title="Deployments" value="24" subtitle="This month" icon={CheckCircle} trend="up" trendValue="100% success rate" bgColor="bg-emerald-50" />
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Server Health</h2>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-xs text-slate-500"><span className="w-2 h-2 bg-emerald-500 rounded-full"></span>Healthy</span>
            <span className="flex items-center gap-1 text-xs text-slate-500"><span className="w-2 h-2 bg-amber-500 rounded-full"></span>Warning</span>
            <span className="flex items-center gap-1 text-xs text-slate-500"><span className="w-2 h-2 bg-red-500 rounded-full"></span>Critical</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Server</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Type</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3 w-32">CPU</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3 w-32">Memory</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3 w-32">Disk</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Region</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {servers.map((s) => (
                <tr key={s.name} className="hover:bg-slate-50 transition-colors">
                  <td className="py-4 font-mono text-sm text-slate-900">{s.name}</td>
                  <td className="py-4 text-sm text-slate-600">{s.type}</td>
                  <td className="py-4 pr-4"><UsageBar value={s.cpu} /></td>
                  <td className="py-4 pr-4"><UsageBar value={s.memory} /></td>
                  <td className="py-4 pr-4"><UsageBar value={s.disk} warning={70} critical={85} /></td>
                  <td className="py-4 text-sm text-slate-600">{s.region}</td>
                  <td className="py-4">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${
                      s.status === 'healthy' ? 'bg-emerald-100 text-emerald-700' :
                      s.status === 'warning' ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                    }`}>{s.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-slate-900">Incident Log</h2>
            <button className="px-4 py-2 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition-colors">Report Incident</button>
          </div>
          <div className="space-y-3">
            {incidents.map((inc) => (
              <div key={inc.id} className="p-4 border border-slate-100 rounded-lg hover:bg-slate-50 transition-colors">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div>
                    <p className="text-xs font-mono text-slate-400">{inc.id}</p>
                    <p className="text-sm font-medium text-slate-900 mt-0.5">{inc.title}</p>
                  </div>
                  <span className={`text-xs font-bold px-2 py-1 rounded shrink-0 ${statusColors[inc.status]}`}>{inc.status}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className={`text-xs font-bold px-2 py-1 rounded ${severityColors[inc.severity]}`}>{inc.severity}</span>
                  <span className="text-xs text-slate-500">{inc.started}</span>
                  <span className="text-xs text-slate-500">→ {inc.assignee}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Service Overview</h2>
          <div className="space-y-3">
            {[
              { service: 'API Gateway', uptime: '99.99%', latency: '42ms', requests: '2.4M/day' },
              { service: 'Auth Service', uptime: '100%', latency: '18ms', requests: '450K/day' },
              { service: 'AI Engine', uptime: '98.2%', latency: '1.2s', requests: '120K/day' },
              { service: 'Database Cluster', uptime: '99.95%', latency: '8ms', requests: '8.1M/day' },
              { service: 'Cache (Redis)', uptime: '100%', latency: '0.4ms', requests: '15M/day' },
              { service: 'Message Queue', uptime: '99.9%', latency: '12ms', requests: '890K/day' },
            ].map((s) => (
              <div key={s.service} className="flex items-center justify-between py-2.5 border-b border-slate-100 last:border-0">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${parseFloat(s.uptime) >= 99.9 ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                  <span className="text-sm font-medium text-slate-900">{s.service}</span>
                </div>
                <div className="flex items-center gap-6 text-xs text-slate-500">
                  <span>↑ {s.uptime}</span>
                  <span>{s.latency}</span>
                  <span>{s.requests}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
