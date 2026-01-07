import { getAllProjectSlugs, getProjectData } from '@/lib/projects';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { GithubRepoCard } from '@/components/GithubRepoCard';

export async function generateStaticParams() {
    const slugs = getAllProjectSlugs();
    return slugs.map((slug) => ({ slug }));
}

export default async function ProjectPage({ params }: { params: { slug: string } }) {
    const project = await getProjectData(params.slug);

    return (
        <article className="container mx-auto min-h-screen px-4 py-16 sm:px-6 lg:px-8">
            <Link href="/projects" className="mb-8 inline-flex items-center text-sm text-gray-400 hover:text-white">
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Projects
            </Link>

            <header className="mb-12">
                <h1 className="mb-4 text-4xl font-extrabold text-white sm:text-5xl">{project.title}</h1>

                <div className="mb-6 flex flex-col gap-6">
                    {project.date && (
                        <span className="text-gray-400 font-medium">
                            {new Date(project.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                        </span>
                    )}

                    {project.repo && (
                        <GithubRepoCard repoUrl={project.repo} />
                    )}
                </div>
            </header>

            {/* Markdown Content */}
            <div
                className="prose prose-lg prose-invert max-w-none custom-prose-content
        prose-headings:text-white prose-p:text-gray-300 prose-strong:text-white 
        prose-a:text-primary prose-a:no-underline hover:prose-a:underline
        prose-img:border prose-img:border-white/10"
                dangerouslySetInnerHTML={{ __html: project.contentHtml }}
            />

            <div className="mt-16 border-t border-white/10 pt-8">
                <Link href="/projects" className="inline-flex items-center text-sm text-gray-400 hover:text-white transition-colors">
                    <ArrowLeft className="mr-2 h-4 w-4" /> Back to Projects
                </Link>
            </div>
        </article>
    );
}
