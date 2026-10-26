import test from 'node:test';
import assert from 'node:assert/strict';
import { configFor } from '../src/model.js';
import {
  CachePool,
  memorySnapshot,
  rankLayout,
  QWEN3_32B,
  bytesPerToken,
  formatBytes,
} from '../src/cache.js';

test('Qwen3-32B capacity uses actual head_dim, independently of hidden size', () => {
  assert.equal(bytesPerToken(QWEN3_32B), 262144);
  assert.equal(bytesPerToken(QWEN3_32B) * 32768, 8 * 1024 ** 3);
  const rank = rankLayout(QWEN3_32B, 4)[0];
  const perRank = 2 * 64 * rank.kvHeads.length * 128 * 2;
  assert.equal(perRank * 32768, 2 * 1024 ** 3);
});
test('TP partitions Q heads and explicitly replicates KV when needed', () => {
  const ranks = rankLayout(configFor('gqa'), 4, 2);
  assert.deepEqual(
    ranks.flatMap((r) => r.qHeads),
    [0, 1, 2, 3],
  );
  assert.deepEqual(
    ranks.map((r) => r.kvHeads),
    [[0], [0], [1], [1]],
  );
  assert.deepEqual(
    ranks.map((r) => r.host),
    [0, 0, 1, 1],
  );
  assert.ok(ranks.every((r) => r.replicas.length === 1));
});
test('compatible full prefix pages share slots; partial tail does not', () => {
  const p = memorySnapshot({ a: ['a', 'b', 'c', 'x'], b: ['a', 'b', 'c', 'y'] });
  assert.equal(p.common, 3);
  assert.equal(p.sharedTokens, 2);
  assert.deepEqual(p.mappings.A.slice(0, 2), p.mappings.B.slice(0, 2));
  assert.notEqual(p.mappings.A[2], p.mappings.B[2]);
  assert.equal(p.reservedSlots, 6);
  assert.equal(p.pages.filter((page) => page.refs === 2).length, 1);
});
test('same suffix and different model identities do not share cache', () => {
  const a = ['a', 'b', 'c', 'd'];
  assert.equal(memorySnapshot({ a, b: ['x', 'b', 'c', 'd'] }).sharedTokens, 0);
  assert.equal(memorySnapshot({ a, b: a, sameIdentity: false }).sharedTokens, 0);
});
test('finish retains pages; eviction only frees pages without active references', () => {
  const input = { a: ['a', 'b', 'c', 'd'], b: ['a', 'b', 'x', 'y'], aActive: false };
  const retained = memorySnapshot(input);
  assert.equal(retained.pages.length, 3);
  assert.equal(retained.pages.filter((p) => p.refs === 0).length, 1);
  const evicted = memorySnapshot({ ...input, evict: true });
  assert.equal(evicted.pages.length, 2);
  assert.equal(evicted.mappings.A[2], null);
  assert.equal(evicted.mappings.A[0], evicted.mappings.B[0]);
  assert.ok(evicted.pages.every((p) => p.refs > 0));
});
test('physical slots are unique except for valid prefix sharing and survive deterministic replay', () => {
  const args = { a: ['0', '1', '2', '3', '4'], b: ['0', '1', 'x', 'y', 'z'] };
  const p = memorySnapshot(args);
  const unique = new Set([...p.mappings.A, ...p.mappings.B]);
  assert.equal(unique.size, 8);
  assert.deepEqual(memorySnapshot(args), p);
  assert.equal(memorySnapshot({}).pages.length, 0);
});
test('one request growing never moves another request’s live slots', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b']);
  pool.extend('B', ['x', 'y']);
  const slots = [...pool.snapshot().mappings.B];
  pool.extend('A', ['a', 'b', 'c', 'd']);
  assert.deepEqual(pool.snapshot().mappings.B, slots);
});
test('eviction prevents later hits and checkpoint restores exact allocation', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b']);
  pool.finish('A');
  const checkpoint = pool.checkpoint();
  pool.evictUnused();
  pool.extend('B', ['a', 'b']);
  assert.equal(pool.snapshot().sharedTokens, 0);
  assert.deepEqual(pool.snapshot().mappings.A, [null, null]);
  pool.restore(checkpoint);
  assert.equal(pool.snapshot().pages.length, 1);
  pool.extend('B', ['a', 'b']);
  assert.equal(pool.snapshot().sharedTokens, 2);
});

test('a known complete prefix page is reused before allocating any new page', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b', 'c', 'd']);
  const before = pool.snapshot();
  pool.allocate = () => assert.fail('A full-page hit must not allocate a temporary page');
  pool.extend('B', ['a', 'b', 'c', 'd']);
  const after = pool.snapshot();
  assert.deepEqual(after.mappings.A, before.mappings.A);
  assert.deepEqual(after.mappings.B, before.mappings.A);
  assert.equal(after.pages.length, 2);
  assert.equal(after.sharedTokens, 4);
  assert.ok(after.pages.every((page) => page.refs === 2));
});

test('filling an allocated partial page never relocates its live slots', () => {
  for (const pageSize of [2, 4]) {
    const tokens = Array.from({ length: pageSize }, (_, i) => `t${i}`);
    const pool = new CachePool(pageSize);
    pool.extend('A', tokens);
    pool.extend('B', tokens.slice(0, 1));
    const slots = pool.snapshot().mappings.B;
    for (let n = 2; n <= pageSize; n++) {
      pool.extend('B', tokens.slice(0, n));
      assert.deepEqual(pool.snapshot().mappings.B.slice(0, slots.length), slots);
    }
    const final = pool.snapshot();
    assert.equal(final.sharedTokens, 0);
    assert.equal(final.pages.length, 2);
    assert.notEqual(final.mappings.A[0], final.mappings.B[0]);
  }
});

test('later full pages may be reused while an earlier partial page keeps its address', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b', 'c', 'd']);
  pool.extend('B', ['a']);
  const firstSlot = pool.snapshot().mappings.B[0];
  pool.extend('B', ['a', 'b', 'c', 'd']);
  const result = pool.snapshot();
  assert.equal(result.mappings.B[0], firstSlot);
  assert.notEqual(result.mappings.A[0], result.mappings.B[0]);
  assert.deepEqual(result.mappings.A.slice(2), result.mappings.B.slice(2));
  assert.equal(result.sharedTokens, 2);
  assert.equal(result.usedSlots, 6);
  assert.equal(result.reservedSlots, 6);
});

test('4-token pages share only full prefixes and account for partial-tail waste', () => {
  const a = Array.from({ length: 10 }, (_, i) => `t${i}`);
  const b = [...a.slice(0, 8), 'x', 'y'];
  const result = memorySnapshot({ a, b, pageSize: 4 });
  assert.equal(result.sharedTokens, 8);
  assert.equal(result.usedSlots, 12);
  assert.equal(result.reservedSlots, 16);
  assert.deepEqual(result.mappings.A.slice(0, 8), result.mappings.B.slice(0, 8));
  assert.notEqual(result.mappings.A[8], result.mappings.B[8]);
});

test('shared pages survive either owner finishing and become reclaimable after both finish', () => {
  const pool = new CachePool(4);
  const tokens = ['a', 'b', 'c', 'd'];
  pool.extend('A', tokens);
  pool.extend('B', tokens);
  const liveMappings = pool.snapshot().mappings;
  pool.finish('A');
  assert.equal(pool.snapshot().pages[0].refs, 1);
  assert.equal(pool.evictUnused(), 0);
  assert.deepEqual(pool.snapshot().mappings, liveMappings);
  pool.finish('B');
  assert.equal(pool.snapshot().pages[0].refs, 0);
  assert.equal(pool.evictUnused(), 1);
  const freed = pool.snapshot();
  assert.deepEqual(freed.mappings.A, [null, null, null, null]);
  assert.deepEqual(freed.mappings.B, [null, null, null, null]);
  assert.equal(freed.usedSlots, 0);
  assert.equal(pool.evictUnused(), 0);
  pool.extend('C', tokens);
  assert.equal(pool.snapshot().sharedTokens, 0);
  assert.deepEqual(pool.snapshot().pages[0].owners, ['C']);
  assert.deepEqual(pool.snapshot().mappings.A, [null, null, null, null]);
});

test('snapshots are isolated from later allocation, ownership, eviction, and caller edits', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a']);
  const partial = pool.snapshot(),
    savedPartial = structuredClone(partial);
  pool.extend('A', ['a', 'b']);
  assert.deepEqual(partial, savedPartial);
  const complete = pool.snapshot(),
    savedComplete = structuredClone(complete);
  pool.extend('B', ['a', 'b']);
  assert.deepEqual(complete, savedComplete);
  partial.pages[0].tokens[0] = 'changed';
  partial.pages[0].owners.length = 0;
  partial.mappings.A[0] = 999;
  assert.equal(pool.snapshot().pages[0].tokens[0], 'a');
  assert.equal(pool.snapshot().pages[0].refs, 2);
  pool.finish('A');
  pool.finish('B');
  pool.evictUnused();
  const evicted = pool.snapshot();
  evicted.evicted[0].tokens[0] = 'changed';
  assert.equal(pool.snapshot().evicted[0].tokens[0], 'a');
  assert.deepEqual(complete, savedComplete);
});

test('restoring a checkpoint isolates future state from edits to that checkpoint', () => {
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b']);
  const checkpoint = pool.checkpoint(),
    expected = pool.snapshot();
  pool.finish('A');
  pool.evictUnused();
  pool.restore(checkpoint);
  assert.deepEqual(pool.snapshot(), expected);
  checkpoint.pages[0].tokens[0] = 'changed';
  checkpoint.requests.A.tokens[0] = 'changed';
  assert.deepEqual(pool.snapshot(), expected);
  pool.extend('A', ['a', 'b', 'c']);
  assert.equal(pool.snapshot().mappings.A.length, 3);
});

test('all device menus cover Q heads and expose exactly the KV heads needed by each rank', () => {
  for (const variant of ['mha', 'gqa', 'mqa'])
    for (const qHeads of [4, 8]) {
      const config = configFor(variant, 4, qHeads);
      for (const tp of [1, 2, 4, 8].filter((n) => n <= qHeads)) {
        for (const gpusPerHost of [1, 2, 4, 8]) {
          const ranks = rankLayout(config, tp, gpusPerHost);
          assert.deepEqual(
            ranks.flatMap((rank) => rank.qHeads),
            Array.from({ length: qHeads }, (_, i) => i),
          );
          for (const rank of ranks) {
            const needed = [
              ...new Set(rank.qHeads.map((h) => Math.floor(h / (qHeads / config.kvHeads)))),
            ];
            assert.deepEqual(rank.kvHeads, needed);
            assert.equal(rank.host, Math.floor(rank.rank / gpusPerHost));
            assert.equal(rank.gpu, rank.rank % gpusPerHost);
            assert.deepEqual(
              rank.replicas,
              needed.filter((h) =>
                ranks.some((other) => other.rank !== rank.rank && other.kvHeads.includes(h)),
              ),
            );
          }
        }
      }
    }
});

test('FP8 estimates and TP8 replication distinguish unique KV from device copies', () => {
  assert.equal(bytesPerToken(QWEN3_32B, 1) * 32768, 4 * 1024 ** 3);
  const qwenRanks = rankLayout(QWEN3_32B, 8);
  assert.ok(qwenRanks.every((rank) => rank.kvHeads.length === 1 && !rank.replicas.length));
  const config = configFor('mqa', 4, 8);
  const ranks = rankLayout(config, 8);
  assert.ok(ranks.every((rank) => rank.kvHeads.length === 1 && rank.replicas.length === 1));
  const physicalBytes = ranks.reduce(
    (sum, rank) => sum + 2 * config.layers * rank.kvHeads.length * config.headDim * 2,
    0,
  );
  assert.equal(physicalBytes, 8 * bytesPerToken(config));
});

test('cache and capacity interfaces reject invalid inputs before changing pool state', () => {
  const config = configFor();
  for (const value of [0, -1, 0.5, NaN, Infinity, '2']) {
    assert.throws(() => new CachePool(value), /page size/);
    assert.throws(() => rankLayout(config, value), /TP/);
    assert.throws(() => rankLayout(config, 2, value), /GPUs per host/);
  }
  for (const value of [0, -1, NaN, Infinity, '2']) {
    assert.throws(() => bytesPerToken(config, value), /positive/);
  }
  for (const value of [-1, NaN, Infinity, '2']) {
    assert.throws(() => formatBytes(value), /nonnegative/);
  }
  assert.throws(() => bytesPerToken({ ...config, kvHeads: 3 }), /divide/);
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(0.5), '0.5 B');
  const pool = new CachePool(2);
  pool.extend('A', ['a', 'b']);
  const before = pool.snapshot();
  for (const [owner, tokens, identity] of [
    ['', ['a'], 'base'],
    [null, ['a'], 'base'],
    ['B', [null], 'base'],
    ['B', 'a', 'base'],
    ['B', ['a'], ''],
    ['A', ['changed'], 'base'],
    ['A', ['a'], 'base'],
    ['A', ['a', 'b'], 'other'],
  ]) {
    assert.throws(() => pool.extend(owner, tokens, identity));
    assert.deepEqual(pool.snapshot(), before);
  }
  pool.finish('A');
  assert.throws(() => pool.extend('A', ['a', 'b', 'c']), /Ended/);
  // Request names must not accidentally resolve through Object.prototype.
  pool.extend('__proto__', ['x']);
  pool.extend('constructor', ['y']);
  assert.equal(pool.snapshot().pages.length, 3);
});
