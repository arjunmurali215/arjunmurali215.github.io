import { Github, Linkedin, Mail } from 'lucide-react';
import Link from 'next/link';
import { resumeData } from '@/data/resume';

export function Footer() {
    return (
        <footer className="border-t border-white/10 bg-black/20 py-12">
            <div className="container mx-auto px-4 md:px-6">
                <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
                    <div className="text-center md:text-left">
                        <h3 className="text-lg font-semibold text-white">Arjun Murali</h3>
                        <p className="mt-2 text-sm text-gray-400">
                            Robotics Engineer | Developer | Innovator
                        </p>
                    </div>

                    <div className="flex gap-6">
                        <Link
                            href={resumeData.socials.find(s => s.platform === 'GitHub')?.url || 'https://github.com/arjunmurali215'}
                            target="_blank"
                            className="text-gray-400 transition-colors hover:text-white"
                        >
                            <Github size={24} />
                        </Link>
                        <Link
                            href={resumeData.socials.find(s => s.platform === 'LinkedIn')?.url || '#'}
                            target="_blank"
                            className="text-gray-400 transition-colors hover:text-white"
                        >
                            <Linkedin size={24} />
                        </Link>
                        <a
                            href={`mailto:${resumeData.email}`}
                            className="text-gray-400 transition-colors hover:text-white"
                        >
                            <Mail size={24} />
                        </a>
                        {/* Adding Github manually as user mentioned projects have repos, likely they have a GH profile even if not in parsed resume text explicitly, or I can omit if unknown. I'll omit for now or match user intent later. */}
                    </div>
                </div>
                <div className="mt-8 text-center text-xs text-gray-500">
                    © {new Date().getFullYear()} Arjun Murali. All rights reserved.
                </div>
            </div>
        </footer>
    );
}
