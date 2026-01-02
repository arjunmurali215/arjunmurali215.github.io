'use client';

import Link from 'next/link';
import { ProjectData } from '@/lib/projects';
import { ArrowUpRight } from 'lucide-react';
import { useEffect, useRef } from 'react';

export function ProjectCard({ project }: { project: ProjectData }) {
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.defaultMuted = true;
            videoRef.current.muted = true;
            videoRef.current.play().catch(e => console.log("Autoplay blocked", e));
        }
    }, [project.coverImage]);

    return (
        <Link
            href={`/projects/${project.slug}`}
            className="group flex flex-col gap-4"
        >
            {/* Image Container - Dominant Visual */}
            <div className="relative aspect-[16/9] w-full overflow-hidden bg-white/5 ring-1 ring-white/10 transition-all group-hover:ring-4 group-hover:ring-primary">
                {project.coverImage ? (
                    <div className="relative h-full w-full overflow-hidden">
                        {project.coverImage.endsWith('.mp4') || project.coverImage.endsWith('.webm') ? (
                            <video
                                ref={videoRef}
                                src={project.coverImage}
                                autoPlay
                                loop
                                muted
                                playsInline
                                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                            />
                        ) : (
                            <img
                                src={project.coverImage}
                                alt={project.title}
                                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                            />
                        )}
                    </div>
                ) : (
                    <div className="flex h-full w-full items-center justify-center text-gray-700 font-mono text-xs">
                        NO_IMAGE
                    </div>
                )}
            </div>

            {/* Content - Minimal */}
            <div className="flex flex-col gap-1">
                {/* Date and Tech Line */}
                <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-gray-500">
                    <span>{project.date || 'NODATE'}</span>
                </div>

                {/* Title */}
                <h3 className="text-lg font-bold text-white group-hover:text-primary transition-colors">
                    {project.title}
                </h3>

                {/* Excerpt */}
                <p className="line-clamp-2 text-sm text-gray-400">
                    {project.excerpt}
                </p>
            </div>
        </Link>
    );
}
