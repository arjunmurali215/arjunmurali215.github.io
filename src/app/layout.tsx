import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import 'katex/dist/katex.min.css';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'Arjun Murali | Portfolio',
  description: 'Robotics Engineer & Developer Portfolio',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body className={`${inter.variable} min-h-screen bg-background font-sans text-foreground antialiased selection:bg-primary/30`}>
        <Navbar />
        <main className="flex min-h-screen flex-col pt-16">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
