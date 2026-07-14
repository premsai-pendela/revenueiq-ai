import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RevenueIQ — Executive Intelligence",
  description:
    "Governed retail analytics: leakage-free churn, customer segmentation, forecasting, and a grounding-verified LLM executive report.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
