import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trace Lens · AIGC evidence browser",
  description: "Inspect image lineage, transformations, and detector evidence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
