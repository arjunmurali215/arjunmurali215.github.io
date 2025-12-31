'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';

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
                <div className="hidden md:flex md:gap-8">
                    {navItems.map((item) => (
                        <Link
                            key={item.path}
                            href={item.path}
                            className={clsx(
                                'relative text-sm font-medium transition-colors hover:text-white',
                                pathname === item.path ? 'text-white' : 'text-gray-400'
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
                </div>

                {/* Mobile Menu Button */}
                <button
                    className="rounded-md p-2 text-gray-400 hover:bg-white/10 hover:text-white md:hidden"
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
                                    'text-base font-medium transition-colors hover:text-white',
                                    pathname === item.path ? 'text-white' : 'text-gray-400'
                                )}
                                onClick={() => setIsOpen(false)}
                            >
                                {item.name}
                            </Link>
                        ))}
                    </div>
                </motion.div>
            )}
        </nav>
    );
}
