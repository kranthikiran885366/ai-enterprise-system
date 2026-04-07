'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Zap, RefreshCw, Loader2, ChevronRight } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const workflows = [
  { id: 'lead-scoring', name: 'Lead Scoring', desc: 'AI-powered lead analysis and scoring', icon: '🎯', status: 'ready' },
  { id: 'expense-audit', name: 'Expense Audit', desc: 'Automated fraud detection and review', icon: '💰', status: 'ready' },
  { id: 'contract-review', name: 'Contract Review', desc: 'Legal document analysis and risk flagging', icon: '📄', status: 'ready' },
  { id: 'candidate-match', name: 'Candidate Matching', desc: 'AI resume screening and ranking', icon: '👥', status: 'ready' },
  { id: 'market-report', name: 'Market Report', desc: 'Competitive analysis and insights', icon: '📊', status: 'running' },
  { id: 'incident-triage', name: 'Incident Triage', desc: 'Automated IT incident classification', icon: '🔧', status: 'ready' },
];

const suggestions = [
  'Analyze our Q1 sales pipeline and identify top opportunities',
  'Review the latest expense reports for fraud risk',
  'Summarize open support tickets by priority',
  'Generate a hiring plan based on current team capacity',
  'Check compliance status across all frameworks',
  'Forecast cash flow for the next 90 days',
];

function AgentTyping() {
  return (
    <div className="flex items-end gap-3 max-w-2xl">
      <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center shrink-0">
        <Bot size={16} className="text-white" />
      </div>
      <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-none px-4 py-3">
        <div className="flex gap-1">
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}

const systemResponses: Record<string, string> = {
  default: "I'm analyzing your request using the enterprise AI system. I have access to data from all departments including Sales, Finance, HR, Marketing, Support, Legal, and IT. How can I help you make better decisions today?",
};

function simulateResponse(message: string): string {
  const lower = message.toLowerCase();
  if (lower.includes('sales') || lower.includes('lead') || lower.includes('pipeline')) {
    return "📊 **Sales Pipeline Analysis**\n\nBased on current CRM data:\n\n• **Total active leads:** 1,247 (↑12% MoM)\n• **Pipeline value:** $8.2M across 4 stages\n• **Top opportunity:** TechCorp Inc — $480K deal, 82% close probability\n• **At-risk deals:** 14 deals showing decreased engagement\n\n**AI Recommendation:** Focus efforts on the 42 deals in Negotiation stage ($1.5M). Conversion velocity has increased 23% this week. Schedule follow-ups with GlobalOps and DataFlow Ltd. within 48 hours.";
  }
  if (lower.includes('expense') || lower.includes('fraud') || lower.includes('finance')) {
    return "💰 **Expense Audit Report**\n\nFinance AI analysis complete:\n\n• **Flagged transactions:** 8 requiring review\n• **Highest risk:** $8,500 purchase at TechSupplies Inc (unusual vendor, 3x avg amount)\n• **Policy violations:** 3 travel expenses exceed per-diem by >20%\n• **Duplicate detected:** Invoice INV-2024-0418 appears submitted twice\n\n**Action Required:** Approve/reject 5 flagged items before month-end close. Estimated recovery if fraud confirmed: $12,400.";
  }
  if (lower.includes('support') || lower.includes('ticket')) {
    return "🎧 **Support Queue Analysis**\n\nCurrent status:\n\n• **Open tickets:** 43 (2 critical SLA breaches)\n• **Avg resolution time:** 4.2h (↓0.8h vs last week)\n• **CSAT score:** 4.7/5.0\n• **AI auto-resolved:** 31% of this week's volume\n\n**Urgent:** TKT-4821 (API integration failure) is 1h overdue for TechCorp Inc — their sentiment score has dropped to 2.1/5. Recommend immediate senior engineer escalation.";
  }
  if (lower.includes('hr') || lower.includes('hiring') || lower.includes('employee') || lower.includes('team')) {
    return "👥 **HR Intelligence Report**\n\nWorkforce analysis:\n\n• **Total employees:** 156 across 8 departments\n• **Open positions:** 12 (4 critical — Engineering)\n• **Attrition risk:** 8 employees flagged by predictive model (65%+ risk)\n• **Top candidate match:** Senior Engineer role — 3 candidates scored >90%\n\n**Recommendation:** Prioritize retention conversation with 3 Senior Engineers showing disengagement signals. Time-to-hire for technical roles is 47 days — consider parallel pipeline building.";
  }
  if (lower.includes('compli') || lower.includes('legal') || lower.includes('gdpr')) {
    return "⚖️ **Compliance Status Report**\n\n• **SOC 2 Type II:** 98% compliant ✅\n• **GDPR:** 84% — **ACTION REQUIRED** ⚠️\n  → Missing sub-processor clauses in 3 EU DPAs\n  → Deadline: June 2024\n• **ISO 27001:** 96% compliant ✅\n• **CCPA:** 92% compliant ✅\n\n**Critical Action:** Update GDPR Data Processing Agreements before June 2024. AI contract review identified 3 agreements requiring sub-processor addendums. Estimated legal effort: 8 hours.";
  }
  if (lower.includes('forecast') || lower.includes('cash flow') || lower.includes('revenue')) {
    return "📈 **90-Day Financial Forecast**\n\nAI financial model projection:\n\n• **Month 1:** $2.1M revenue (↑8% YoY), $487K expenses\n• **Month 2:** $2.3M projected (seasonality adjustment applied)\n• **Month 3:** $2.4M projected (Q2 contract renewals: $890K)\n\n**Cash flow position:** Strong. 4.2 months runway at current burn. Marketing budget is 16% over target — recommend reallocation from LinkedIn Ads to Google PPC (5.6% vs 2.1% conversion).";
  }
  return `I've processed your request through the multi-agent system:\n\n**PlannerAgent** → analyzed the query and routed to relevant data sources\n**ExecutorAgent** → retrieved data from Sales, Finance, HR, and Operations\n**AnalyzerAgent** → applied ML models and pattern recognition\n\nHere's what I found: Your query touches ${Math.floor(Math.random() * 3) + 2} departments with ${Math.floor(Math.random() * 15) + 5} relevant data points. Based on current enterprise data, I recommend scheduling a cross-functional review. Would you like me to drill deeper into any specific area?`;
}

export default function AIPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      role: 'assistant',
      content: "Hello! I'm your Enterprise AI Assistant powered by a multi-agent system. I have real-time access to Sales, Finance, HR, Marketing, Support, Legal, and IT data. Ask me anything or choose a quick action below.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeView, setActiveView] = useState<'chat' | 'workflows'>('chat');
  const [runningWorkflow, setRunningWorkflow] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const sendMessage = async (text?: string) => {
    const content = text || input.trim();
    if (!content) return;
    setInput('');

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    await new Promise((r) => setTimeout(r, 1500 + Math.random() * 1000));

    const reply = simulateResponse(content);
    setIsTyping(false);
    setMessages((prev) => [...prev, {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: reply,
      timestamp: new Date(),
    }]);
  };

  const runWorkflow = async (wf: typeof workflows[0]) => {
    setRunningWorkflow(wf.id);
    await new Promise((r) => setTimeout(r, 2500));
    setRunningWorkflow(null);
    setActiveView('chat');
    setMessages((prev) => [...prev, {
      id: Date.now().toString(),
      role: 'assistant',
      content: `✅ **${wf.name} workflow completed**\n\nThe automated workflow has finished processing. ${wf.desc}. Results have been recorded and are available in the respective department dashboard. Would you like a detailed summary?`,
      timestamp: new Date(),
    }]);
  };

  return (
    <div className="h-full -m-8">
      <div className="flex h-screen">
        <div className="w-72 bg-slate-900 text-white flex flex-col border-r border-slate-700">
          <div className="p-6 border-b border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <Bot size={20} className="text-white" />
              </div>
              <div>
                <h2 className="font-bold text-white">AI Assistant</h2>
                <p className="text-xs text-slate-400">Multi-Agent System</p>
              </div>
            </div>
          </div>

          <div className="p-4 border-b border-slate-700">
            <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
              <button onClick={() => setActiveView('chat')} className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${activeView === 'chat' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>Chat</button>
              <button onClick={() => setActiveView('workflows')} className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${activeView === 'workflows' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>Workflows</button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {activeView === 'chat' ? (
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase mb-3">Quick Actions</p>
                <div className="space-y-1">
                  {suggestions.map((s, i) => (
                    <button key={i} onClick={() => sendMessage(s)} className="w-full text-left text-xs text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2.5 rounded-lg transition-colors flex items-start gap-2">
                      <ChevronRight size={12} className="shrink-0 mt-0.5 text-slate-500" />
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase mb-3">Available Workflows</p>
                <div className="space-y-2">
                  {workflows.map((wf) => (
                    <div key={wf.id} className="bg-slate-800 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm">{wf.icon} {wf.name}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${wf.status === 'running' ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'}`}>{wf.status}</span>
                      </div>
                      <p className="text-xs text-slate-400 mb-2">{wf.desc}</p>
                      <button
                        onClick={() => runWorkflow(wf)}
                        disabled={runningWorkflow === wf.id || wf.status === 'running'}
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 bg-blue-600 text-white text-xs rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
                      >
                        {runningWorkflow === wf.id ? <><Loader2 size={12} className="animate-spin" /> Running...</> : <><Zap size={12} /> Run Workflow</>}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="p-4 border-t border-slate-700">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-xs text-slate-400">4 agents active</span>
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col bg-slate-50">
          <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-lg font-bold text-slate-900">Enterprise AI Chat</h1>
              <p className="text-xs text-slate-500">Planner → Executor → Analyzer → Response</p>
            </div>
            <button onClick={() => setMessages([{id: '0', role: 'assistant', content: "New conversation started. How can I help you?", timestamp: new Date()}])} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
              <RefreshCw size={14} /> New Chat
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex items-end gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-slate-600' : 'bg-blue-600'}`}>
                  {msg.role === 'user' ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
                </div>
                <div className={`max-w-2xl px-4 py-3 rounded-2xl text-sm whitespace-pre-line ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-white border border-slate-200 text-slate-800 rounded-bl-none'}`}>
                  {msg.content}
                </div>
              </div>
            ))}
            {isTyping && <AgentTyping />}
            <div ref={messagesEndRef} />
          </div>

          <div className="bg-white border-t border-slate-200 p-4">
            <form onSubmit={(e) => { e.preventDefault(); sendMessage(); }} className="flex gap-3">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about your enterprise data..."
                className="flex-1 px-4 py-3 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button type="submit" disabled={!input.trim() || isTyping} className="px-5 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center gap-2">
                <Send size={16} />
              </button>
            </form>
            <p className="text-xs text-slate-400 text-center mt-2">AI responses use enterprise data. Always verify critical decisions.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
