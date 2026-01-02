
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';

const projectsDirectory = path.resolve(process.cwd(), 'projects');

export interface ProjectData {
    slug: string;
    title: string;
    excerpt: string;
    contentHtml: string;
    coverImage?: string;
    date?: string;
    repo?: string;
    tags?: string[];
    [key: string]: any;
}

export function getAllProjectSlugs() {
    if (!fs.existsSync(projectsDirectory)) return [];
    const entries = fs.readdirSync(projectsDirectory, { withFileTypes: true });
    return entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name);
}

// Custom plugin to rewrite image URLs and transform video files
function rehypeRewriteUrls(options: { slug: string }) {
    return (tree: any) => {
        visit(tree, 'element', (node: any) => {
            if (node.tagName === 'img' && node.properties && node.properties.src) {
                let src = node.properties.src as string;

                // Rewrite relative paths
                if (src.startsWith('./') || (!src.startsWith('/') && !src.startsWith('http'))) {
                    // Remove ./ prefix if present
                    if (src.startsWith('./')) {
                        src = src.substring(2);
                    }
                    // Prepend project slug path
                    src = `/projects/${options.slug}/${src}`;
                    node.properties.src = src;
                }

                if (src.endsWith('.mp4') || src.endsWith('.webm')) {
                    node.tagName = 'video';
                    node.properties.autoplay = true;
                    node.properties.loop = true;
                    node.properties.muted = true;
                    node.properties.playsInline = true;
                    // node.properties.controls = true; // Removed per user request for "default" play
                    node.properties.width = '100%';
                    node.properties.className = ['w-full', 'rounded-xl', 'border', 'border-white/10'];
                    delete node.properties.alt;
                }
            }
        });
    };
}

export async function getProjectData(slug: string): Promise<ProjectData> {
    const fullPath = path.join(projectsDirectory, slug, 'doc.md');

    if (!fs.existsSync(fullPath)) {
        throw new Error(`Project file not found: ${fullPath}`);
    }

    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const matterResult = matter(fileContents);
    const contentWithoutTitle = matterResult.content.replace(/^#\s+.+$/m, '');

    const processedContent = await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeRaw)
        .use(rehypeSlug)
        .use(rehypeKatex)
        .use(rehypeRewriteUrls, { slug })
        .use(rehypeStringify)
        .process(contentWithoutTitle);

    const contentHtml = processedContent.toString();

    let title = matterResult.data.title;
    let excerpt = matterResult.data.excerpt;

    if (!title) {
        const h1Match = matterResult.content.match(/^#\s+(.+)$/m);
        title = h1Match ? h1Match[1] : slug.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    if (!excerpt) {
        const lines = matterResult.content.split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('![')) {
                excerpt = trimmed;
                break;
            }
        }
        if (!excerpt) excerpt = "No description available.";
    }

    // Cover Image Logic
    let coverImage = matterResult.data.coverImage;
    if (!coverImage) {
        const imgMatch = matterResult.content.match(/!\[.*?\]\((.*?)\)/);
        if (imgMatch) {
            coverImage = imgMatch[1];
        }
    }

    // Apply path fixing to ANY cover image (whether from frontmatter or extracted)
    if (coverImage) {
        if (coverImage.startsWith('./')) {
            coverImage = coverImage.replace('./', `/projects/${slug}/`);
        } else if (!coverImage.startsWith('http') && !coverImage.startsWith('/')) {
            coverImage = `/projects/${slug}/${coverImage}`;
        }
    }


    return {
        ...matterResult.data,
        slug,
        contentHtml,
        title,
        excerpt,
        coverImage,
    };
}

export async function getAllProjects(): Promise<ProjectData[]> {
    const slugs = getAllProjectSlugs();
    const projects = await Promise.all(slugs.map(slug => getProjectData(slug)));
    // Sort projects by date
    return projects.sort((a, b) => {
        if (a.date && b.date) {
            return a.date < b.date ? 1 : -1;
        }
        return 0;
    });
}
