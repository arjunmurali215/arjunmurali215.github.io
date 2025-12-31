
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECTS_DIR = path.resolve(__dirname, '../projects');
const PUBLIC_PROJECTS_DIR = path.resolve(__dirname, '../public/projects');

function syncProjects() {
    if (!fs.existsSync(PROJECTS_DIR)) {
        console.warn('Projects directory not found:', PROJECTS_DIR);
        return;
    }

    if (fs.existsSync(PUBLIC_PROJECTS_DIR)) {
        fs.rmSync(PUBLIC_PROJECTS_DIR, { recursive: true, force: true });
    }

    fs.mkdirSync(PUBLIC_PROJECTS_DIR, { recursive: true });

    const entries = fs.readdirSync(PROJECTS_DIR, { withFileTypes: true });

    for (const entry of entries) {
        if (entry.isDirectory()) {
            const srcDir = path.join(PROJECTS_DIR, entry.name);
            const destDir = path.join(PUBLIC_PROJECTS_DIR, entry.name);

            // Copy the entire directory
            fs.cpSync(srcDir, destDir, { recursive: true });
            console.log(`Synced project: ${entry.name}`);
        }
    }
    console.log('Project sync complete.');
}

syncProjects();
