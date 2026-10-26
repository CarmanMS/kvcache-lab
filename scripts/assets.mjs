import path from 'node:path';

export const contentTypes = Object.freeze({
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
});

// Both the build and local server expose the same browser resources.
export function isPublicAsset(relativePath) {
  if (relativePath === 'index.html') return true;
  if (!relativePath.startsWith('src/') || /[\\:\0]/u.test(relativePath)) return false;
  if (relativePath.split('/').some((part) => !part || part.startsWith('.'))) return false;
  return Object.hasOwn(contentTypes, path.posix.extname(relativePath));
}
