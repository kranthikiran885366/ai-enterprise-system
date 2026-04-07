'use client';

import MetricCard from '@/components/dashboard/MetricCard';
import { FileText, Shield, AlertTriangle, CheckCircle, Clock, Scale } from 'lucide-react';

const contracts = [
  { id: 'CTR-2841', name: 'SaaS Master Agreement - TechCorp', type: 'MSA', status: 'active', value: '$480,000', expiry: 'Dec 2025', risk: 'low' },
  { id: 'CTR-2840', name: 'Data Processing Agreement - EU GDPR', type: 'DPA', status: 'review', value: 'N/A', expiry: 'Jun 2024', risk: 'high' },
  { id: 'CTR-2839', name: 'Enterprise License - GlobalOps', type: 'License', status: 'active', value: '$220,000', expiry: 'Mar 2025', risk: 'low' },
  { id: 'CTR-2838', name: 'Vendor Agreement - CloudProvider', type: 'Vendor', status: 'expiring', value: '$95,000', expiry: 'Apr 2024', risk: 'medium' },
  { id: 'CTR-2837', name: 'NDA - Acquisition Target Alpha', type: 'NDA', status: 'active', value: 'N/A', expiry: 'Oct 2024', risk: 'low' },
];

const complianceItems = [
  { framework: 'SOC 2 Type II', status: 'compliant', score: 98, nextAudit: 'Sep 2024' },
  { framework: 'GDPR', status: 'action_required', score: 84, nextAudit: 'Ongoing' },
  { framework: 'ISO 27001', status: 'compliant', score: 96, nextAudit: 'Jan 2025' },
  { framework: 'CCPA', status: 'compliant', score: 92, nextAudit: 'Jun 2024' },
  { framework: 'HIPAA', status: 'not_applicable', score: 0, nextAudit: 'N/A' },
];

const riskColors: Record<string, string> = {
  low: 'bg-emerald-100 text-emerald-700',
  medium: 'bg-amber-100 text-amber-700',
  high: 'bg-red-100 text-red-700',
};

const statusColors: Record<string, string> = {
  active: 'bg-emerald-100 text-emerald-700',
  review: 'bg-blue-100 text-blue-700',
  expiring: 'bg-amber-100 text-amber-700',
  expired: 'bg-red-100 text-red-700',
};

export default function LegalPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Legal & Compliance</h1>
        <p className="text-slate-600 mt-2">Contract lifecycle management, compliance tracking, and risk analysis</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard title="Active Contracts" value="127" subtitle="Total portfolio" icon={FileText} trend="up" trendValue="12 new this quarter" bgColor="bg-blue-50" />
        <MetricCard title="Compliance Score" value="93%" subtitle="Avg across frameworks" icon={Shield} trend="up" trendValue="+2% this month" bgColor="bg-emerald-50" />
        <MetricCard title="Expiring Soon" value="8" subtitle="Next 60 days" icon={Clock} trend="neutral" trendValue="3 require renewal" bgColor="bg-amber-50" />
        <MetricCard title="Risk Flags" value="5" subtitle="Needs attention" icon={AlertTriangle} trend="down" trendValue="2 resolved this week" bgColor="bg-red-50" />
        <MetricCard title="AI Reviews" value="34" subtitle="Completed this month" icon={CheckCircle} trend="up" trendValue="Avg 98% accuracy" bgColor="bg-purple-50" />
        <MetricCard title="Contract Value" value="$4.2M" subtitle="Total active portfolio" icon={Scale} trend="up" trendValue="+$800K this quarter" bgColor="bg-blue-50" />
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Contract Registry</h2>
          <button className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
            New Contract
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Contract</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Type</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Value</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Expiry</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Risk</th>
                <th className="text-left text-xs font-semibold text-slate-500 uppercase pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {contracts.map((c) => (
                <tr key={c.id} className="hover:bg-slate-50 transition-colors cursor-pointer">
                  <td className="py-4">
                    <p className="text-xs font-mono text-slate-400">{c.id}</p>
                    <p className="text-sm font-medium text-slate-900 mt-0.5">{c.name}</p>
                  </td>
                  <td className="py-4 text-sm text-slate-600">{c.type}</td>
                  <td className="py-4 text-sm font-semibold text-slate-900">{c.value}</td>
                  <td className="py-4 text-sm text-slate-600">{c.expiry}</td>
                  <td className="py-4">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${riskColors[c.risk]}`}>{c.risk}</span>
                  </td>
                  <td className="py-4">
                    <span className={`text-xs font-bold px-2 py-1 rounded ${statusColors[c.status]}`}>{c.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">Compliance Dashboard</h2>
          <div className="space-y-4">
            {complianceItems.filter((c) => c.status !== 'not_applicable').map((item) => (
              <div key={item.framework} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
                <div>
                  <p className="text-sm font-medium text-slate-900">{item.framework}</p>
                  <p className="text-xs text-slate-500 mt-0.5">Next audit: {item.nextAudit}</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <p className="text-lg font-bold text-slate-900">{item.score}%</p>
                    <div className="w-24 bg-slate-100 h-1.5 rounded-full mt-1">
                      <div
                        className={`h-full rounded-full ${item.score >= 90 ? 'bg-emerald-500' : 'bg-amber-500'}`}
                        style={{ width: `${item.score}%` }}
                      />
                    </div>
                  </div>
                  <span className={`text-xs font-bold px-2 py-1 rounded ${
                    item.status === 'compliant' ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                  }`}>{item.status === 'compliant' ? 'Compliant' : 'Action Required'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">AI Contract Analysis</h2>
          <div className="space-y-4">
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <p className="text-sm font-semibold text-amber-900 mb-1">GDPR Article 28 Gap Detected</p>
              <p className="text-xs text-amber-700">Data Processing Agreement with EU customers missing sub-processor clauses. Update required before June 2024.</p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-sm font-semibold text-red-900 mb-1">Vendor Contract Expiring</p>
              <p className="text-xs text-red-700">CloudProvider agreement expires in 23 days. Auto-renewal clause absent — manual renewal required immediately.</p>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm font-semibold text-blue-900 mb-1">Liability Cap Opportunity</p>
              <p className="text-xs text-blue-700">3 enterprise contracts have liability caps below industry standard. Recommend renegotiation at next renewal.</p>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
              <p className="text-sm font-semibold text-emerald-900 mb-1">Contract Portfolio Health</p>
              <p className="text-xs text-emerald-700">94.3% of contracts are fully compliant with current regulatory requirements. Portfolio risk score: LOW.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
