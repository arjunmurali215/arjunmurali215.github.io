import { resumeData } from '@/data/resume';
import { getAllProjects } from '@/lib/projects';
import { ProjectCard } from '@/components/ProjectCard';
import Link from 'next/link';
import { ArrowRight, Download, GraduationCap, Trophy } from 'lucide-react';

export default async function Home() {
  const allProjects = await getAllProjects();
  const featuredProjects = allProjects.slice(0, 3); // Show first 3

  return (
    <div className="flex flex-col gap-24 pb-20">
      {/* Hero Section */}
      <section className="relative flex min-h-[80vh] flex-col justify-center px-4 pt-16 sm:px-6 lg:px-8">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />
        <div className="container mx-auto max-w-5xl">
          <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-7xl">
            Hi, I'm <span className="text-primary">{resumeData.name}</span>
          </h1>
          <p className="mt-6 max-w-2xl text-xl text-gray-400">
            Robotics Engineer with a passion for Manipulators and Computer Vision.
            Currently studying at {resumeData.education[0].school}.
          </p>
          <div className="mt-10 flex gap-4">
            <Link
              href="/projects"
              className="inline-flex items-center justify-center rounded-lg bg-primary px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-primary/90"
            >
              View Projects <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
            <a
              href="/Resume.pdf"
              target="_blank"
              className="inline-flex items-center justify-center rounded-lg border border-white/10 bg-white/5 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-white/10"
            >
              Download Resume <Download className="ml-2 h-4 w-4" />
            </a>
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="mb-12 text-3xl font-bold text-white">Technical Skills</h2>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          {resumeData.skills.map((category) => (
            <div key={category.category} className="rounded-xl border border-white/10 bg-white/5 p-6">
              <h3 className="mb-4 text-lg font-semibold text-primary">{category.category}</h3>
              <ul className="space-y-2">
                {category.skills.map((skill) => (
                  <li key={skill} className="flex items-center text-gray-300">
                    <span className="mr-2 h-1.5 w-1.5 rounded-full bg-primary/50" />
                    {skill}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Experience / Leadership */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid gap-12 lg:grid-cols-2">
          {/* Education */}
          <div>
            <h2 className="mb-8 flex items-center text-3xl font-bold text-white">
              <GraduationCap className="mr-3 text-primary" /> Education
            </h2>
            <div className="space-y-8">
              {resumeData.education.map((edu, idx) => (
                <div key={idx} className="relative border-l-2 border-white/10 pl-6">
                  <span className="absolute -left-[9px] top-0 h-4 w-4 rounded-full bg-primary" />
                  <h3 className="text-xl font-bold text-white">{edu.school}</h3>
                  <p className="text-lg text-primary">{edu.degree}</p>
                  <p className="mt-1 text-sm text-gray-400">{edu.year} | {edu.location}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Leadership */}
          <div>
            <h2 className="mb-8 flex items-center text-3xl font-bold text-white">
              <Trophy className="mr-3 text-primary" /> Leadership
            </h2>
            <div className="space-y-8">
              {resumeData.leadership.map((role, idx) => (
                <div key={idx} className="relative border-l-2 border-white/10 pl-6">
                  <span className="absolute -left-[9px] top-0 h-4 w-4 rounded-full bg-primary" />
                  <h3 className="text-xl font-bold text-white">{role.role}</h3>
                  <p className="text-lg text-primary">{role.organization}</p>
                  <p className="mt-1 text-sm text-gray-400">{role.period}</p>
                  <ul className="mt-4 list-disc space-y-2 pl-4 text-gray-300">
                    {role.description.map((desc, i) => (
                      <li key={i}>{desc}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Featured Projects */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12 flex items-center justify-between">
          <h2 className="text-3xl font-bold text-white">Featured Projects</h2>
          <Link href="/projects" className="text-primary hover:underline">
            View all projects
          </Link>
        </div>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {featuredProjects.map((project) => (
            <ProjectCard key={project.slug} project={project} />
          ))}
        </div>
      </section>
    </div>
  );
}
