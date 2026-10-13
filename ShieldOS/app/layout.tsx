import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ShieldOS · Real-Time Fraud Decisioning",
  description:
    "Interactive real-time fraud monitoring, explainable risk scoring, automated decision routing, and traceability audit.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
