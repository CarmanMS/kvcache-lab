import http from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { contentTypes, isPublicAsset } from './assets.mjs';

function isLocalHost(host) {
  if (typeof host !== 'string') return false;
  const match = /^(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?$/iu.exec(host);
  return Boolean(match && (!match[1] || (Number(match[1]) > 0 && Number(match[1]) <= 65535)));
}

export async function createStaticServer(root) {
  const canonicalRoot = await realpath(root);
  return http.createServer(async (req, res) => {
    const reply = (status, message, headers = {}) => {
      res.writeHead(status, {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
        ...headers,
      });
      res.end(message);
    };

    if (!isLocalHost(req.headers.host)) {
      reply(403, 'Forbidden');
      return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      reply(405, 'Method not allowed', { Allow: 'GET, HEAD' });
      return;
    }

    let pathname;
    try {
      // Inspect the original path before URL parsing can normalize dot segments.
      if (!req.url.startsWith('/') || req.url.startsWith('//')) throw new URIError();
      pathname = decodeURIComponent(req.url.split('?')[0]);
      if (pathname.includes('\0')) throw new URIError();
    } catch {
      reply(400, 'Bad request');
      return;
    }

    const relativePath = pathname === '/' ? 'index.html' : pathname.slice(1);
    if (!isPublicAsset(relativePath)) {
      reply(403, 'Forbidden');
      return;
    }

    try {
      const file = await realpath(path.join(canonicalRoot, relativePath));
      const actualRelativePath = path.relative(canonicalRoot, file).split(path.sep).join('/');
      // Recheck the resolved target: links cannot escape or expose private files inside the root.
      if (!isPublicAsset(actualRelativePath)) {
        reply(403, 'Forbidden');
        return;
      }
      if (!(await stat(file)).isFile()) {
        reply(404, 'Not found');
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, {
        'Content-Type':
          relativePath === 'index.html'
            ? 'text/html; charset=utf-8'
            : contentTypes[path.posix.extname(relativePath)],
        'Content-Length': body.length,
        'Cache-Control': 'no-cache',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(req.method === 'HEAD' ? undefined : body);
    } catch (error) {
      const missing = ['ENOENT', 'ENOTDIR', 'EISDIR'].includes(error.code);
      reply(missing ? 404 : 500, missing ? 'Not found' : 'Internal server error');
    }
  });
}
