'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import OverviewDashboard from '@/components/OverviewDashboard';
import DatasetUpload from '@/components/DatasetUpload';
import ModelLeaderboard from '@/components/ModelLeaderboard';
import MonitoringDrift from '@/components/MonitoringDrift';
import AuditLogs from '@/components/AuditLogs';
import UploadCustomModel from '@/components/UploadCustomModel';

export type Tab = 'overview' | 'upload' | 'models' | 'monitoring' | 'logs' | 'custom';

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <OverviewDashboard />;
      case 'upload': return <DatasetUpload />;
      case 'models': return <ModelLeaderboard />;
      case 'monitoring': return <MonitoringDrift />;
      case 'logs': return <AuditLogs />;
      case 'custom': return <UploadCustomModel />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-y-auto">
        <div className="p-6 max-w-7xl mx-auto fade-in">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}
