import { readdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
async function check(folder) {
  for (const entry of await readdir(path.join(root, folder), { withFileTypes: true })) {
    const relative = path.join(folder, entry.name);
    if (entry.isDirectory()) await check(relative);
    else if (/\.(?:js|mjs)$/.test(entry.name)) {
      const result = spawnSync(process.execPath, ['--check', relative], {
        cwd: root,
        stdio: 'inherit',
      });
      if (result.error) throw result.error;
      if (result.status !== 0) throw new Error(`Syntax check failed: ${relative}`);
    }
  }
}
for (const folder of ['src', 'scripts', 'tests']) await check(folder);
console.log('JavaScript syntax checks passed.');
