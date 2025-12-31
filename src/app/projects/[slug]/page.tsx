import { getAllProjectSlugs, getProjectData } from '@/lib/projects';
import Link from 'next/link';
import { ArrowLeft, Github } from 'lucide-react';

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
                {project.repo && (
                    <a
                        href={project.repo} // Assuming user might add repo to frontmatter later
                        target="_blank"
                        className="inline-flex items-center rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-white hover:bg-white/20"
                    >
                        <Github className="mr-2 h-4 w-4" /> View Source
                    </a>
                )}
            </header>

            {/* Markdown Content */}
            <div
                className="prose prose-lg prose-invert max-w-none 
        prose-headings:text-white prose-p:text-gray-300 prose-strong:text-white 
        prose-a:text-primary prose-a:no-underline hover:prose-a:underline
        prose-img:rounded-xl prose-img:border prose-img:border-white/10"
                dangerouslySetInnerHTML={{ __html: project.contentHtml }}
            />
        </article>
    );
}
