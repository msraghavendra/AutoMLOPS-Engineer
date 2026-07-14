'use client';

import { LayoutDashboard, Upload, Trophy, Activity, ScrollText, Cpu, ChevronRight } from 'lucide-react';
import type { Tab } from '@/app/page';

const NAV_ITEMS: { id: Tab; label: string; icon: React.ElementType; description: string }[] = [
  { id: 'overview', label: 'Dashboard', icon: LayoutDashboard, description: 'System overview' },
  { id: 'upload', label: 'Dataset Upload', icon: Upload, description: 'Import & analyze data' },
  { id: 'models', label: 'Model Registry', icon: Trophy, description: 'Leaderboard & deploy' },
  { id: 'monitoring', label: 'Monitoring', icon: Activity, description: 'Drift & predictions' },
  { id: 'logs', label: 'Audit Logs', icon: ScrollText, description: 'Event history' },
];

interface SidebarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export default function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <aside className="w-64 h-full flex flex-col shrink-0" style={{ background: 'rgba(10,15,30,0.95)', borderRight: '1px solid #1e2d47' }}>
      {/* Logo */}
      <div className="px-5 py-5 flex items-center gap-3" style={{ borderBottom: '1px solid #1e2d47' }}>
        <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', boxShadow: '0 0 20px rgba(59,130,246,0.3)' }}>
          <Cpu size={18} className="text-white" />
        </div>
        <div>
          <div className="font-bold text-sm text-gradient-blue">Ship It ML</div>
          <div className="text-xs" style={{ color: '#64748b' }}>AutoMLOps Engine</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <div className="text-xs font-medium uppercase tracking-wider px-3 mb-3" style={{ color: '#475569' }}>Navigation</div>
        {NAV_ITEMS.map(({ id, label, icon: Icon, description }) => (
          <button
            key={id}
            id={`nav-${id}`}
            onClick={() => onTabChange(id)}
            className={`sidebar-link w-full ${activeTab === id ? 'active' : ''}`}
          >
            <Icon size={16} className="shrink-0" />
            <div className="flex-1 text-left">
              <div className="leading-tight">{label}</div>
              <div className="text-xs mt-0.5" style={{ color: '#475569', fontWeight: 400 }}>{description}</div>
            </div>
            {activeTab === id && <ChevronRight size={14} className="shrink-0 opacity-60" />}
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4" style={{ borderTop: '1px solid #1e2d47' }}>
        <div className="text-xs" style={{ color: '#475569' }}>
          <div className="font-medium mb-1" style={{ color: '#64748b' }}>Simulated Deployment Layer</div>
          <div>In-process model serving. No container orchestration.</div>
        </div>
      </div>
    </aside>
  );
}
