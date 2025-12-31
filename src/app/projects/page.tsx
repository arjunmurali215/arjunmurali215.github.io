import { getAllProjects } from '@/lib/projects';
import { ProjectCard } from '@/components/ProjectCard';

export default async function ProjectsPage() {
    const projects = await getAllProjects();

    return (
        <div className="container mx-auto min-h-screen px-4 py-16 sm:px-6 lg:px-8">
            <h1 className="mb-4 text-4xl font-bold text-white">Projects</h1>
            <p className="mb-12 text-xl text-gray-400">
                A collection of my work in robotics, computer vision, and software engineering.
            </p>

            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {projects.map((project) => (
                    <ProjectCard key={project.slug} project={project} />
                ))}
            </div>
        </div>
    );
}
