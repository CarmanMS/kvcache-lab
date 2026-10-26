import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import {
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from '../scripts/build.mjs';

const run = promisify(execFile);
const project = fileURLToPath(new URL('../', import.meta.url));

async function fixture(t) {
  await mkdir(path.join(project, 'tmp'), { recursive: true });
  const base = await mkdtemp(path.join(project, 'tmp/build-test-'));
  const root = path.join(base, 'project');
  t.after(() => rm(base, { recursive: true, force: true }));
  await mkdir(path.join(root, 'src/nested'), { recursive: true });
  await writeFile(path.join(root, 'index.html'), '<h1>Fixture</h1>');
  await writeFile(path.join(root, 'LICENSE'), 'Fixture MIT license');
  await writeFile(path.join(root, 'NOTICE.md'), 'Fixture source notice');
  await writeFile(path.join(root, 'src/app.js'), 'export const fixture = true;');
  await writeFile(path.join(root, 'src/nested/math.js'), 'export const result = 2;');
  await writeFile(path.join(root, 'src/styles.css'), 'body { color: black; }');
  return { base, root };
}

async function contents(root) {
  const entries = await readdir(root, { recursive: true });
  const files = {};
  for (const entry of entries.sort()) {
    const file = path.join(root, entry);
    if ((await lstat(file)).isFile()) {
      files[entry.split(path.sep).join('/')] = await readFile(file, 'utf8');
    }
  }
  return files;
}

test('build contains public browser resources and distribution notices with reproducible bytes', async (t) => {
  const { root } = await fixture(t);
  await mkdir(path.join(root, 'src/.private'), { recursive: true });
  await writeFile(path.join(root, 'src/.private/secret.js'), 'synthetic-secret');
  await writeFile(path.join(root, 'src/.secret.js'), 'synthetic-secret');
  await writeFile(path.join(root, 'src/notes.md'), 'not a browser asset');
  await writeFile(path.join(root, 'package.json'), '{"private":true}');
  const output = await build(root);
  const first = await contents(output);
  assert.deepEqual(Object.keys(first).sort(), [
    'LICENSE',
    'NOTICE.md',
    'index.html',
    'src/app.js',
    'src/nested/math.js',
    'src/styles.css',
  ]);
  assert.equal(first.LICENSE, 'Fixture MIT license');
  assert.equal(first['NOTICE.md'], 'Fixture source notice');
  assert.equal(first['src/app.js'], await readFile(path.join(root, 'src/app.js'), 'utf8'));
  await build(root);
  assert.deepEqual(await contents(output), first);
});

test('rebuilding removes deleted sources and any stale output files', async (t) => {
  const { root } = await fixture(t);
  await writeFile(path.join(root, 'src/removed.js'), 'obsolete');
  const output = await build(root);
  await writeFile(path.join(output, 'stale.txt'), 'obsolete');
  await rm(path.join(root, 'src/removed.js'));
  await build(root);
  await assert.rejects(readFile(path.join(output, 'src/removed.js')), { code: 'ENOENT' });
  await assert.rejects(readFile(path.join(output, 'stale.txt')), { code: 'ENOENT' });
  assert.equal(await readFile(path.join(output, 'index.html'), 'utf8'), '<h1>Fixture</h1>');
});

test('refuses symbolic links in sources before replacing an existing build', async (t) => {
  const { base, root } = await fixture(t);
  const output = await build(root);
  const original = await contents(output);
  const external = path.join(base, 'external');
  await mkdir(external);
  await writeFile(path.join(external, 'secret.js'), 'synthetic-secret');
  await symlink(
    external,
    path.join(root, 'src/link'),
    process.platform === 'win32' ? 'junction' : 'dir',
  );
  await assert.rejects(build(root), /Symbolic links are not build inputs/);
  assert.deepEqual(await contents(output), original);
});

test('refuses a linked src directory', async (t) => {
  const { base, root } = await fixture(t);
  const external = path.join(base, 'external');
  await mkdir(external);
  await writeFile(path.join(external, 'secret.js'), 'synthetic-secret');
  await rm(path.join(root, 'src'), { recursive: true });
  await symlink(
    external,
    path.join(root, 'src'),
    process.platform === 'win32' ? 'junction' : 'dir',
  );
  await assert.rejects(build(root), /not symbolic links/);
});

test('replaces a linked output without deleting the link target', async (t) => {
  const { base, root } = await fixture(t);
  const external = path.join(base, 'external');
  await mkdir(external);
  await writeFile(path.join(external, 'keep.txt'), 'preserve');
  await symlink(
    external,
    path.join(root, 'dist'),
    process.platform === 'win32' ? 'junction' : 'dir',
  );
  const output = await build(root);
  assert.equal((await lstat(output)).isSymbolicLink(), false);
  assert.equal(await readFile(path.join(external, 'keep.txt'), 'utf8'), 'preserve');
  assert.equal(await readFile(path.join(output, 'index.html'), 'utf8'), '<h1>Fixture</h1>');
});

test('CLI build works when invoked from another working directory', async (t) => {
  const { base, root } = await fixture(t);
  await mkdir(path.join(root, 'scripts'));
  for (const name of ['build.mjs', 'assets.mjs']) {
    await copyFile(path.join(project, 'scripts', name), path.join(root, 'scripts', name));
  }
  const result = await run(process.execPath, [path.join(root, 'scripts/build.mjs')], { cwd: base });
  assert.match(result.stdout, /Built dist/);
  assert.equal(await readFile(path.join(root, 'dist/index.html'), 'utf8'), '<h1>Fixture</h1>');
});

test('missing distribution notice fails before replacing the existing output', async (t) => {
  const { root } = await fixture(t);
  const output = await build(root);
  const original = await contents(output);
  await rm(path.join(root, 'NOTICE.md'));
  await assert.rejects(build(root), { code: 'ENOENT' });
  assert.deepEqual(await contents(output), original);
});
