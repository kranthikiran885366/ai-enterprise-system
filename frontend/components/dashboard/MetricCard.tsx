import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  bgColor?: string;
}

export default function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  bgColor = 'bg-blue-50',
}: MetricCardProps) {
  const trendColor = {
    up: 'text-emerald-600',
    down: 'text-red-600',
    neutral: 'text-slate-600',
  }[trend || 'neutral'];

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-600">{title}</p>
          <p className="text-3xl font-bold text-slate-900 mt-2">{value}</p>
          {subtitle && (
            <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
          )}
          {trendValue && (
            <p className={`text-sm font-medium mt-2 ${trendColor}`}>
              {trend === 'up' && '↑'} {trend === 'down' && '↓'} {trendValue}
            </p>
          )}
        </div>
        {Icon && (
          <div className={`${bgColor} p-3 rounded-lg`}>
            <Icon size={24} className="text-blue-600" />
          </div>
        )}
      </div>
    </div>
  );
}
