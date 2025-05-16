import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css"; // Import global styles

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Ultrasound Data Explorer",
  description: "Web application for visualizing processed ultrasound data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[#f0f0f0]`}>
        {children}
      </body>
    </html>
  );
} 