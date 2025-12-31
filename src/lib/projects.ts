
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
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

// Custom plugin to rewrite image URLs
function rehypeRewriteUrls(options: { slug: string }) {
    return (tree: any) => {
        visit(tree, 'element', (node: any) => {
            if (node.tagName === 'img' && node.properties && node.properties.src) {
                let src = node.properties.src as string;
                // Check for relative paths (./assets or assets/)
                if (src.startsWith('./') || (!src.startsWith('/') && !src.startsWith('http'))) {
                    // Remove ./ prefix if present
                    if (src.startsWith('./')) {
                        src = src.substring(2);
                    }
                    // Prepend project slug path
                    node.properties.src = `/projects/${options.slug}/${src}`;
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

    // Use gray-matter to parse the post metadata section
    const matterResult = matter(fileContents);

    // Remove the first H1 from the content if it exists, as it is rendered separately in the page template
    const contentWithoutTitle = matterResult.content.replace(/^#\s+.+$/m, '');

    // Use unified pipeline to convert markdown into HTML string
    const processedContent = await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeRaw)
        .use(rehypeKatex)
        .use(rehypeRewriteUrls, { slug })
        .use(rehypeStringify)
        .process(contentWithoutTitle);

    const contentHtml = processedContent.toString();

    // Inference logic if frontmatter is missing
    let title = matterResult.data.title;
    let excerpt = matterResult.data.excerpt;

    if (!title) {
        // Try to find the first H1
        const h1Match = matterResult.content.match(/^#\s+(.+)$/m);
        if (h1Match) {
            title = h1Match[1];
        } else {
            title = slug.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    }

    if (!excerpt) {
        // Try to find the first paragraph that is not a header
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

    // Look for first image for cover if not specified
    let coverImage = matterResult.data.coverImage;
    if (!coverImage) {
        const imgMatch = matterResult.content.match(/!\[.*?\]\((.*?)\)/);
        if (imgMatch) {
            coverImage = imgMatch[1];
            // Ensure path logic if it is relative
            if (coverImage.startsWith('./')) {
                coverImage = coverImage.replace('./', `/projects/${slug}/`);
            } else if (!coverImage.startsWith('http') && !coverImage.startsWith('/')) {
                coverImage = `/projects/${slug}/${coverImage}`;
            }
        }
    }


    return {
        slug,
        contentHtml,
        title,
        excerpt,
        coverImage,
        ...matterResult.data,
    };
}

export async function getAllProjects(): Promise<ProjectData[]> {
    const slugs = getAllProjectSlugs();
    const projects = await Promise.all(slugs.map(slug => getProjectData(slug)));
    return projects;
}
