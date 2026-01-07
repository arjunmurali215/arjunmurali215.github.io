import { resumeData } from '@/data/resume';
import { getAllProjects } from '@/lib/projects';
import { ProjectCard } from '@/components/ProjectCard';
import Link from 'next/link';
import { ArrowRight, Download, GraduationCap, Trophy } from 'lucide-react';

// Define the order of featured projects here
const FEATURED_PROJECT_SLUGS = [
  'dexlite-grasp-synthesis',
  'f1tenth',
  'dual-arm-dataset',
];

export default async function Home() {
  const allProjects = await getAllProjects();

  // Filter and sort projects based on the manual list
  const featuredProjects = FEATURED_PROJECT_SLUGS
    .map(slug => allProjects.find(p => p.slug === slug))
    .filter((p): p is NonNullable<typeof p> => p !== undefined);

  return (
    <div className="flex flex-col gap-24 pb-20 pt-24">
      {/* Hero Section - Minimal */}
      <section className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl">
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-6xl">
            {resumeData.name}
          </h1>
          <p className="mt-4 text-xl text-gray-400 font-light">
            Student Researcher. Focus on Manipulators and Computer Vision.
          </p>
          <p className="mt-2 text-gray-500">
            Currently at {resumeData.education[0].school}.
          </p>

          <div className="mt-8 flex items-center gap-8">
            <a
              href="/Resume.pdf"
              target="_blank"
              className="inline-flex items-center rounded-full bg-primary px-6 py-3 text-sm font-medium text-black transition-all hover:bg-primary/90 hover:scale-105"
            >
              Download Resume <Download className="ml-2 h-4 w-4" />
            </a>
            <Link
              href="/projects"
              className="inline-flex items-center text-sm font-medium text-primary hover:text-white transition-colors"
            >
              View Projects <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Featured Projects - Chronological */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8 flex items-end justify-between border-b border-white/10 pb-4">
          <h2 className="text-xl font-bold text-white uppercase tracking-wider">Selected Works</h2>
          <Link href="/projects" className="group flex items-center gap-2 text-sm font-mono text-primary transition-colors hover:text-white">
            View All <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
        </div>
        <div className="grid gap-x-8 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
          {featuredProjects.map((project) => (
            <ProjectCard key={project.slug} project={project} />
          ))}
        </div>
      </section>

      {/* Skills Section - Table Style */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8 border-b border-white/10 pb-4">
          <h2 className="text-xl font-bold text-white uppercase tracking-wider">Technical Proficiency</h2>
        </div>
        <div className="grid gap-8 md:grid-cols-2">
          {resumeData.skills.map((category) => (
            <div key={category.category}>
              <h3 className="mb-2 text-sm font-mono text-primary uppercase">{category.category}</h3>
              <div className="text-gray-300 leading-relaxed">
                {category.skills.join(" / ")}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Experience / Leadership - Minimal List */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid gap-16 lg:grid-cols-2">
          <div>
            <div className="mb-8 border-b border-white/10 pb-4">
              <h2 className="text-xl font-bold text-white uppercase tracking-wider">Education</h2>
            </div>
            <div className="space-y-8">
              {resumeData.education.map((edu, idx) => (
                <div key={idx}>
                  <div className="flex justify-between items-baseline mb-1">
                    <h3 className="text-lg font-bold text-white">{edu.school}</h3>
                    <span className="text-sm font-mono text-gray-500">{edu.year}</span>
                  </div>
                  <p className="text-primary whitespace-pre-line">{edu.degree}</p>
                  <p className="text-sm text-gray-500">{edu.location}</p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="mb-8 border-b border-white/10 pb-4">
              <h2 className="text-xl font-bold text-white uppercase tracking-wider">Leadership</h2>
            </div>
            <div className="space-y-8">
              {resumeData.leadership.map((role, idx) => (
                <div key={idx}>
                  <div className="flex justify-between items-baseline mb-1">
                    <h3 className="text-lg font-bold text-white">{role.role}</h3>
                    <span className="text-sm font-mono text-gray-500">{role.period}</span>
                  </div>
                  <p className="text-primary mb-2">{role.organization}</p>
                  <ul className="list-none space-y-1">
                    {role.description.map((desc, i) => (
                      <li key={i} className="text-sm text-gray-400 before:content-['-'] before:mr-2 before:text-gray-600">
                        {desc}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
