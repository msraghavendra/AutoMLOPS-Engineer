import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ship It ML — AutoMLOps Engineer",
  description: "Automated MLOps platform that trains, deploys, monitors, and retrains ML models from end to end.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-grid-pattern">{children}</body>
    </html>
  );
}
