import Link from 'next/link';
import { Star, GitFork, Github, Circle } from 'lucide-react';

interface GithubRepoCardProps {
    repoUrl: string;
}

interface RepoData {
    name: string;
    description: string;
    stargazers_count: number;
    forks_count: number;
    language: string;
    html_url: string;
    owner: {
        login: string;
    };
}

async function getRepoData(repoUrl: string): Promise<RepoData | null> {
    try {
        const urlParts = repoUrl.split('github.com/');
        if (urlParts.length !== 2) return null;

        const [owner, repo] = urlParts[1].split('/');
        if (!owner || !repo) return null;

        const res = await fetch(`https://api.github.com/repos/${owner}/${repo}`, {
            next: { revalidate: 3600 },
            headers: {
                'User-Agent': 'Portfolio-Website'
            }
        });

        if (!res.ok) return null;

        return res.json();
    } catch (error) {
        console.error('Error fetching repo data:', error);
        return null;
    }
}

export async function GithubRepoCard({ repoUrl }: GithubRepoCardProps) {
    const data = await getRepoData(repoUrl);

    if (!data) {
        // Fallback to simple link if API fails
        return (
            <a
                href={repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-white hover:bg-white/20 transition-colors"
            >
                <Github className="mr-2 h-4 w-4" /> View Source
            </a>
        );
    }

    return (
        <a
            href={data.html_url}
            target="_blank"
            rel="noopener noreferrer"
            className="group block w-full max-w-md overflow-hidden rounded-xl border border-white/10 bg-white/5 p-4 transition-all hover:bg-white/10 hover:border-white/20"
        >
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Github className="h-4 w-4" />
                    <span>{data.owner.login}</span>
                </div>
                <span className="text-xs text-gray-500 font-mono">Public</span>
            </div>

            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-primary transition-colors">
                {data.name}
            </h3>

            <p className="text-sm text-gray-400 mb-4 line-clamp-2 min-h-[2.5em]">
                {data.description || "No description available."}
            </p>

            <div className="flex items-center gap-4 text-sm text-gray-400">
                {data.language && (
                    <div className="flex items-center gap-1.5">
                        <Circle className="h-3 w-3 fill-current text-primary" />
                        <span>{data.language}</span>
                    </div>
                )}

                <div className="flex items-center gap-1.5">
                    <Star className="h-3.5 w-3.5" />
                    <span>{data.stargazers_count}</span>
                </div>

                <div className="flex items-center gap-1.5">
                    <GitFork className="h-3.5 w-3.5" />
                    <span>{data.forks_count}</span>
                </div>
            </div>
        </a>
    );
}
