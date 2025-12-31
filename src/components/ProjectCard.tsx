import Link from 'next/link';
import { ProjectData } from '@/lib/projects';
import { ArrowUpRight } from 'lucide-react';

export function ProjectCard({ project }: { project: ProjectData }) {
    return (
        <Link
            href={`/projects/${project.slug}`}
            className="group relative flex h-full flex-col overflow-hidden rounded-xl border border-white/10 bg-white/5 transition-all hover:-translate-y-1 hover:border-white/20 hover:bg-white/10"
        >
            <div className="flex flex-1 flex-col p-6">
                <h3 className="mb-2 text-xl font-bold text-white group-hover:text-primary">
                    {project.title}
                </h3>
                <p className="mb-4 line-clamp-3 flex-1 text-sm text-gray-400">
                    {project.excerpt}
                </p>
                <div className="flex items-center text-sm font-medium text-primary">
                    View Project <ArrowUpRight size={16} className="ml-1" />
                </div>
            </div>
        </Link>
    );
}
