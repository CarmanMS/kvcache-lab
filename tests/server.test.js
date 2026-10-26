import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createStaticServer } from '../scripts/server.mjs';

const temporaryRoot = fileURLToPath(new URL('../tmp/', import.meta.url));

async function fixture(t) {
  await mkdir(temporaryRoot, { recursive: true });
  const base = await mkdtemp(path.join(temporaryRoot, 'server-test-'));
  const root = path.join(base, 'public');
  let server;
  t.after(async () => {
    if (server?.listening) {
      server.closeAllConnections();
      await new Promise((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    }
    await rm(base, { recursive: true, force: true });
  });
  await mkdir(path.join(root, 'src/nested'), { recursive: true });
  await writeFile(path.join(root, 'index.html'), '<h1>Fixture</h1>');
  await writeFile(path.join(root, 'src/app.js'), 'export const fixture = true;');
  await writeFile(path.join(root, 'src/styles.css'), 'body { color: black; }');
  await writeFile(path.join(root, 'src/nested/math.js'), 'export const result = 2;');
  await writeFile(path.join(root, 'package.json'), '{"private": true}');
  await writeFile(path.join(root, 'src/.secret.js'), 'synthetic-secret');
  server = await createStaticServer(root);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  const request = (pathname, { method = 'GET', host = `127.0.0.1:${port}` } = {}) =>
    new Promise((resolve, reject) => {
      const req = http.request(
        {
          hostname: '127.0.0.1',
          port,
          path: pathname,
          method,
          setHost: false,
          headers: { Host: host },
        },
        (res) => {
          const chunks = [];
          res.on('data', (chunk) => chunks.push(chunk));
          res.on('error', reject);
          res.on('end', () =>
            resolve({
              status: res.statusCode,
              headers: res.headers,
              body: Buffer.concat(chunks).toString('utf8'),
            }),
          );
        },
      );
      req.setTimeout(5000, () => req.destroy(new Error('HTTP fixture timed out')));
      req.on('error', reject);
      req.end();
    });
  return { base, root, request };
}

test('serves browser resources with MIME types and HEAD metadata, without a HEAD body', async (t) => {
  const { request } = await fixture(t);
  const index = await request('/');
  assert.equal(index.status, 200);
  assert.equal(index.body, '<h1>Fixture</h1>');
  assert.equal(index.headers['content-type'], 'text/html; charset=utf-8');
  assert.equal(index.headers['x-content-type-options'], 'nosniff');
  assert.equal(index.headers['cache-control'], 'no-cache');
  const head = await request('/', { method: 'HEAD' });
  assert.equal(head.status, 200);
  assert.equal(head.body, '');
  assert.equal(head.headers['content-length'], index.headers['content-length']);
  const script = await request('/src/nested/math.js?revision=1');
  assert.equal(script.status, 200);
  assert.equal(script.headers['content-type'], 'text/javascript; charset=utf-8');
  assert.equal(script.body, 'export const result = 2;');
  assert.equal(
    (await request('/src/styles.css')).headers['content-type'],
    'text/css; charset=utf-8',
  );
});

test('rejects non-resource paths, hidden paths and traversal on either platform', async (t) => {
  const { request } = await fixture(t);
  for (const pathname of [
    '/package.json',
    '/scripts/serve.mjs',
    '/tmp/probe.js',
    '/.git/config',
    '/src/.secret.js',
    '/src/.private/secret.js',
    '/src/app.js.map',
    '/src/readme.md',
    '/../index.html',
    '/%2e%2e/index.html',
    '/src/%2e%2e/index.html',
    '/src%5c..%5cindex.html',
    '/src/app.js:private',
    '/src//app.js',
  ]) {
    assert.equal((await request(pathname)).status, 403, pathname);
  }
  assert.equal((await request('/src/missing.js')).status, 404);
  assert.equal((await request('/')).status, 200);
});

test('malformed URLs return 400 without breaking subsequent requests', async (t) => {
  const { request } = await fixture(t);
  for (const pathname of [
    '/%E0%A4%A',
    '/%00',
    '//example.invalid/index.html',
    'http://example.invalid/',
  ]) {
    assert.equal((await request(pathname)).status, 400, pathname);
  }
  assert.equal((await request('/')).status, 200);
});

test('accepts only GET and HEAD', async (t) => {
  const { request } = await fixture(t);
  for (const method of ['POST', 'PUT', 'DELETE', 'OPTIONS']) {
    const response = await request('/', { method });
    assert.equal(response.status, 405, method);
    assert.equal(response.headers.allow, 'GET, HEAD');
  }
});

test('requires an exact loopback Host and rejects DNS rebinding hostnames', async (t) => {
  const { request } = await fixture(t);
  for (const host of ['localhost', 'localhost:5174', '127.0.0.1', '[::1]:5174']) {
    assert.equal((await request('/', { host })).status, 200, host);
  }
  for (const host of [
    '',
    'review-attacker.invalid',
    'localhost.evil.invalid',
    '127.0.0.1.evil.invalid',
    'localhost@evil.invalid',
    'localhost:65536',
    'localhost:0',
  ]) {
    assert.equal((await request('/', { host })).status, 403, host);
  }
});

test('resolved links cannot expose external, hidden or non-public files', async (t) => {
  const { base, root, request } = await fixture(t);
  const type = process.platform === 'win32' ? 'junction' : 'dir';
  for (const [name, target] of [
    ['outside', path.join(base, 'external')],
    ['hidden', path.join(root, '.private')],
    ['internal', path.join(root, 'tmp')],
  ]) {
    await mkdir(target, { recursive: true });
    await writeFile(path.join(target, 'leak.js'), 'synthetic-secret');
    await symlink(target, path.join(root, 'src', name), type);
    const response = await request(`/src/${name}/leak.js`);
    assert.equal(response.status, 403, name);
    assert.ok(!response.body.includes('synthetic-secret'));
  }
  await symlink(path.join(root, 'src/nested'), path.join(root, 'src/alias'), type);
  assert.equal((await request('/src/alias/math.js')).status, 200);
});
