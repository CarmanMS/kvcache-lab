import { mkdir, copyFile, lstat, readdir, realpath, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isPublicAsset } from './assets.mjs';

const project = fileURLToPath(new URL('../', import.meta.url));

async function sourceFiles(root) {
  const index = await lstat(path.join(root, 'index.html'));
  const source = await lstat(path.join(root, 'src'));
  if (!index.isFile() || !source.isDirectory()) {
    throw new Error('index.html and src must be regular files/directories, not symbolic links.');
  }
  const files = ['index.html'];
  for (const document of ['LICENSE', 'NOTICE.md']) {
    if (!(await lstat(path.join(root, document))).isFile()) {
      throw new Error(`Distribution notice must be a regular file: ${document}`);
    }
    files.push(document);
  }
  async function visit(relativePath) {
    const entries = await readdir(path.join(root, relativePath), { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.')) continue;
      const relative = path.posix.join(relativePath, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`Symbolic links are not build inputs: ${relative}`);
      }
      if (entry.isDirectory()) await visit(relative);
      else if (entry.isFile() && isPublicAsset(relative)) files.push(relative);
    }
  }
  await visit('src');
  return files;
}

export async function build(root = project) {
  const canonicalRoot = await realpath(root);
  const files = await sourceFiles(canonicalRoot);
  const output = path.join(canonicalRoot, 'dist');
  // Removing a previous output also removes stale files and replaces output symlinks safely.
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  for (const relative of files) {
    const destination = path.join(output, relative);
    await mkdir(path.dirname(destination), { recursive: true });
    await copyFile(path.join(canonicalRoot, relative), destination);
  }
  return output;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await build();
    console.log('Built dist/ — static, no runtime dependencies.');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
