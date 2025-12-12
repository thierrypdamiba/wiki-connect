import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wiki Connect - Connect Any Two Topics",
  description: "Connect any two topics using Qdrant vector search across 35M Wikipedia articles, with Linkup for real-time web grounding. Built with Google ADK.",
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "Wiki Connect",
    description: "Connect any two topics using Qdrant vector search across 35M Wikipedia articles",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Wiki Connect",
    description: "Connect any two topics using Qdrant vector search across 35M Wikipedia articles",
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
