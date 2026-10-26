import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createStaticServer } from './server.mjs';

const project = fileURLToPath(new URL('../', import.meta.url));
const root = process.argv.includes('--dist') ? path.join(project, 'dist') : project;
const portArg = process.argv.indexOf('--port');
const portValue = portArg >= 0 ? process.argv[portArg + 1] : process.env.PORT || '5174';

try {
  if (!/^[0-9]+$/u.test(portValue ?? '') || Number(portValue) > 65535) {
    throw new Error('Port must be an integer between 0 and 65535.');
  }
  const server = await createStaticServer(root);
  server.on('error', (error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
  server.listen(Number(portValue), '127.0.0.1', () => {
    console.log(`KVCache Lab → http://127.0.0.1:${server.address().port}`);
  });
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
