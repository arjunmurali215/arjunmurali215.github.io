'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { Menu, X, Github, Linkedin, Mail } from 'lucide-react';
import { useState } from 'react';
import { resumeData } from '@/data/resume';

const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Projects', path: '/projects' },
];

export function Navbar() {
    const pathname = usePathname();
    const [isOpen, setIsOpen] = useState(false);

    return (
        <nav className="fixed top-0 z-50 w-full border-b border-white/10 bg-black/50 backdrop-blur-xl supports-[backdrop-filter]:bg-black/20">
            <div className="container mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
                <Link href="/" className="text-xl font-bold tracking-tight text-white">
                    Arjun<span className="text-primary">Murali</span>
                </Link>

                {/* Desktop Nav */}
                <div className="hidden md:flex md:items-center md:gap-8">
                    {navItems.map((item) => (
                        <Link
                            key={item.path}
                            href={item.path}
                            className={clsx(
                                'relative text-sm uppercase tracking-widest font-medium transition-colors hover:text-primary',
                                pathname === item.path ? 'text-white' : 'text-gray-300'
                            )}
                        >
                            {item.name}
                            {pathname === item.path && (
                                <motion.div
                                    layoutId="navbar-indicator"
                                    className="absolute -bottom-[21px] left-0 right-0 h-0.5 bg-primary"
                                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                                />
                            )}
                        </Link>
                    ))}

                    <div className="ml-8 flex items-center gap-6 border-l border-white/10 pl-8">
                        {/* GitHub */}
                        {resumeData.socials.find(s => s.platform === 'GitHub') && (
                            <a
                                href={resumeData.socials.find(s => s.platform === 'GitHub')?.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-400 hover:text-primary transition-colors"
                                aria-label="GitHub"
                            >
                                <Github size={20} />
                            </a>
                        )}
                        {/* Mail */}
                        <a
                            href={`mailto:${resumeData.email}`}
                            className="text-gray-400 hover:text-primary transition-colors"
                            aria-label="Email"
                        >
                            <Mail size={20} />
                        </a>
                        {/* LinkedIn */}
                        {resumeData.socials.find(s => s.platform === 'LinkedIn') && (
                            <a
                                href={resumeData.socials.find(s => s.platform === 'LinkedIn')?.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-400 hover:text-primary transition-colors"
                                aria-label="LinkedIn"
                            >
                                <Linkedin size={20} />
                            </a>
                        )}
                    </div>
                </div>

                {/* Mobile Menu Button */}
                <button
                    className="rounded-md p-2 text-gray-300 hover:bg-white/10 hover:text-white md:hidden"
                    onClick={() => setIsOpen(!isOpen)}
                >
                    {isOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Mobile Nav */}
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="border-b border-white/10 bg-black/90 backdrop-blur-xl md:hidden"
                >
                    <div className="flex flex-col space-y-4 p-4">
                        {navItems.map((item) => (
                            <Link
                                key={item.path}
                                href={item.path}
                                className={clsx(
                                    'text-base font-medium transition-colors hover:text-primary',
                                    pathname === item.path ? 'text-white' : 'text-gray-300'
                                )}
                                onClick={() => setIsOpen(false)}
                            >
                                {item.name}
                            </Link>
                        ))}

                        <div className="flex items-center gap-6 border-t border-white/10 pt-4">
                            {/* GitHub */}
                            {resumeData.socials.find(s => s.platform === 'GitHub') && (
                                <a
                                    href={resumeData.socials.find(s => s.platform === 'GitHub')?.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-gray-400 hover:text-primary transition-colors"
                                    aria-label="GitHub"
                                >
                                    <Github size={20} />
                                </a>
                            )}
                            {/* Mail */}
                            <a
                                href={`mailto:${resumeData.email}`}
                                className="text-gray-400 hover:text-primary transition-colors"
                                aria-label="Email"
                            >
                                <Mail size={20} />
                            </a>
                            {/* LinkedIn */}
                            {resumeData.socials.find(s => s.platform === 'LinkedIn') && (
                                <a
                                    href={resumeData.socials.find(s => s.platform === 'LinkedIn')?.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-gray-400 hover:text-primary transition-colors"
                                    aria-label="LinkedIn"
                                >
                                    <Linkedin size={20} />
                                </a>
                            )}
                        </div>
                    </div>
                </motion.div>
            )}
        </nav>
    );
}
