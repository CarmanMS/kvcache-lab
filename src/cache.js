import { kvHeadFor, validateConfig, validateTP, validateTokens } from './model.js';

export const QWEN3_32B_SOURCE =
  'https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json';

export const QWEN3_32B = Object.freeze({
  layers: 64,
  qHeads: 64,
  kvHeads: 8,
  headDim: 128,
  hidden: 5120,
});
export function bytesPerToken(config, bytes = 2) {
  validateConfig(config);
  if (!Number.isFinite(bytes) || bytes <= 0) throw new Error('Bytes per element must be positive');
  return 2 * config.layers * config.kvHeads * config.headDim * bytes;
}
export function rankLayout(config, tp = 1, gpusPerHost = 2) {
  validateTP(config, tp);
  if (!Number.isSafeInteger(gpusPerHost) || gpusPerHost < 1) {
    throw new Error('GPUs per host must be a positive integer');
  }
  const perRank = config.qHeads / tp;
  const ranks = Array.from({ length: tp }, (_, rank) => {
    const qHeads = Array.from({ length: perRank }, (_, h) => rank * perRank + h);
    return {
      rank,
      host: Math.floor(rank / gpusPerHost),
      gpu: rank % gpusPerHost,
      qHeads,
      kvHeads: [...new Set(qHeads.map((h) => kvHeadFor(h, config)))],
    };
  });
  return ranks.map((rank) => ({
    ...rank,
    replicas: rank.kvHeads.filter((h) =>
      ranks.some((other) => other.rank !== rank.rank && other.kvHeads.includes(h)),
    ),
  }));
}
export function prefixLength(a, b) {
  let n = 0;
  while (n < Math.min(a.length, b.length) && a[n] === b[n]) n++;
  return n;
}

/** Reuse complete pages before allocation; allocated slots never move. */
export class CachePool {
  constructor(pageSize = 2) {
    if (!Number.isSafeInteger(pageSize) || pageSize < 1) throw new Error('Invalid page size');
    this.pageSize = pageSize;
    this.pages = [];
    this.requests = Object.create(null);
    this.highWater = 8;
    this.evicted = [];
  }
  checkpoint() {
    return structuredClone({
      pageSize: this.pageSize,
      pages: this.pages,
      requests: this.requests,
      highWater: this.highWater,
      evicted: this.evicted,
    });
  }
  restore(checkpoint) {
    const copy = structuredClone(checkpoint);
    Object.assign(this, copy, { requests: Object.assign(Object.create(null), copy.requests) });
  }
  allocate(owner, identity) {
    const order = [3, 0, 6, 2, 7, 1, 5, 4];
    let id = order.find((n) => !this.pages.some((p) => p.id === n));
    if (id == null) {
      id = 8;
      while (this.pages.some((p) => p.id === id)) id++;
    }
    const page = { id, identity, tokens: [], owners: [owner], prefixKey: null };
    this.pages.push(page);
    this.highWater = Math.max(this.highWater, id + 1);
    return page;
  }
  extend(owner, tokens, identity = 'base') {
    if (typeof owner !== 'string' || !owner.length) throw new Error('Invalid request ID');
    if (typeof identity !== 'string' || !identity.length) throw new Error('Invalid model identity');
    validateTokens(tokens);
    const req = (this.requests[owner] ??= { tokens: [], slots: [], active: true, identity });
    if (!req.active) throw new Error('Ended requests must be reset before extending');
    if (req.identity !== identity || prefixLength(req.tokens, tokens) !== req.tokens.length) {
      throw new Error('An existing request must retain its model identity and prefix');
    }
    for (let t = req.tokens.length; t < tokens.length; t++) {
      const offset = t % this.pageSize;
      // Only reuse a page when its entire prefix is known before allocation.
      // Filling an existing partial page must not relocate its earlier tokens.
      if (offset === 0 && t + this.pageSize <= tokens.length) {
        const prefixKey = JSON.stringify(tokens.slice(0, t + this.pageSize));
        const hit = this.pages.find((p) => p.identity === identity && p.prefixKey === prefixKey);
        if (hit) {
          if (!hit.owners.includes(owner)) hit.owners.push(owner);
          req.tokens.push(...tokens.slice(t, t + this.pageSize));
          for (let k = 0; k < this.pageSize; k++) req.slots.push(hit.id * this.pageSize + k);
          t += this.pageSize - 1;
          continue;
        }
      }
      const page =
        offset === 0
          ? this.allocate(owner, identity)
          : this.pages.find((p) => p.id === Math.floor(req.slots[t - 1] / this.pageSize));
      page.tokens.push(tokens[t]);
      req.tokens.push(tokens[t]);
      req.slots.push(page.id * this.pageSize + offset);
      if (offset === this.pageSize - 1) {
        page.prefixKey = JSON.stringify(req.tokens);
      }
    }
  }
  finish(owner) {
    if (this.requests[owner]) this.requests[owner].active = false;
  }
  evictUnused() {
    this.evicted = this.pages.filter((p) => !p.owners.some((o) => this.requests[o].active));
    const ids = new Set(this.evicted.map((p) => p.id));
    this.pages = this.pages.filter((p) => !ids.has(p.id));
    for (const req of Object.values(this.requests))
      req.slots = req.slots.map((slot) =>
        slot != null && ids.has(Math.floor(slot / this.pageSize)) ? null : slot,
      );
    return this.evicted.length;
  }
  snapshot() {
    const a = this.requests.A,
      b = this.requests.B;
    const mappings = { A: [...(a?.slots || [])], B: [...(b?.slots || [])] };
    const common = a && b && a.identity === b.identity ? prefixLength(a.tokens, b.tokens) : 0;
    const sharedTokens = mappings.A.filter(
      (slot, t) => slot != null && slot === mappings.B[t],
    ).length;
    const pages = this.pages
      .map((p) => ({
        ...p,
        activeOwners: p.owners.filter((o) => this.requests[o].active),
        refs: p.owners.filter((o) => this.requests[o].active).length,
      }))
      .sort((x, y) => x.id - y.id);
    return structuredClone({
      pages,
      mappings,
      common,
      sharedTokens,
      evicted: this.evicted,
      pageSize: this.pageSize,
      capacity: this.highWater * this.pageSize,
      reservedSlots: pages.length * this.pageSize,
      usedSlots: pages.reduce((sum, p) => sum + p.tokens.length, 0),
    });
  }
}
export function memorySnapshot({
  a = [],
  b = [],
  aActive = true,
  bActive = true,
  sameIdentity = true,
  pageSize = 2,
  evict = false,
}) {
  const pool = new CachePool(pageSize);
  pool.extend('A', a);
  pool.extend('B', b, sameIdentity ? 'base' : 'other');
  if (!aActive) pool.finish('A');
  if (!bActive) pool.finish('B');
  if (evict) pool.evictUnused();
  return pool.snapshot();
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) throw new Error('Bytes must be finite and nonnegative');
  if (bytes === 0) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const index = Math.max(0, Math.min(4, Math.floor(Math.log(bytes) / Math.log(1024))));
  return `${Number((bytes / 1024 ** index).toFixed(index > 0 || bytes < 1 ? 2 : 0))} ${units[index]}`;
}
