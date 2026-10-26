import test from 'node:test';
import assert from 'node:assert/strict';
import { add, maxDifference } from '../src/math.js';
import {
  configFor,
  createModel,
  forwardCached,
  forwardFull,
  kvHeadFor,
  tpPartialOutputs,
  tokenize,
} from '../src/model.js';
import { twoDimensionalOracle } from './fixtures/two-dimensional-oracle.js';

const words = ['请', '翻译', '：', '今天', '下雨', '了'];
for (const variant of ['mha', 'gqa', 'mqa']) {
  test(`${variant}: cached and independent full-prefix execution agree at every token`, () => {
    const config = configFor(variant);
    const cached = forwardCached(words, config);
    const full = forwardFull(words, config);
    assert.ok(maxDifference(cached.outputs, full) < 1e-12);
    for (const trace of cached.traces)
      for (const layer of trace.layers) {
        for (const head of layer.heads) {
          assert.ok(Math.abs(head.weights.reduce((a, b) => a + b, 0) - 1) < 1e-12);
          assert.equal(head.weights.length, trace.position + 1);
        }
      }
  });
}
test('appending future tokens leaves all old keys/values and outputs unchanged', () => {
  const before = forwardCached(words.slice(0, 3));
  const after = forwardCached(words);
  before.caches.forEach((cache, layer) => {
    assert.deepEqual(cache.keys, after.caches[layer].keys.slice(0, 3));
    assert.deepEqual(cache.values, after.caches[layer].values.slice(0, 3));
    assert.deepEqual(Object.keys(cache), ['keys', 'values']);
  });
  assert.deepEqual(before.outputs, after.outputs.slice(0, 3));
});
test('same token has different layer KV; changed context affects deeper-layer KV', () => {
  const a = forwardCached(words);
  const b = forwardCached(['改写', ...words.slice(1)]);
  assert.deepEqual(a.caches[0].keys[3], b.caches[0].keys[3]);
  assert.ok(maxDifference(a.caches[1].keys[3], b.caches[1].keys[3]) > 1e-4);
  assert.ok(maxDifference(a.caches[0].keys[3], a.caches[1].keys[3]) > 1e-4);
});
test('GQA maps groups of Q heads to KV heads', () => {
  assert.deepEqual(
    [0, 1, 2, 3].map((h) => kvHeadFor(h, configFor('gqa'))),
    [0, 0, 1, 1],
  );
});
test('all TP partial outputs sum to the unsharded attention output', () => {
  for (const variant of ['mha', 'gqa', 'mqa']) {
    const config = configFor(variant);
    const result = forwardCached(words, config);
    for (const tp of [1, 2, 4])
      for (const trace of result.traces.at(-1).layers) {
        const partials = tpPartialOutputs(trace, result.model.layers[trace.layer], config, tp);
        const sum = partials.reduce(add);
        assert.ok(maxDifference(sum, trace.attentionOutput) < 1e-12);
      }
  }
});
test('empty input is supported and invalid head grouping is rejected', () => {
  assert.deepEqual(forwardCached([]).outputs, []);
  assert.throws(() => forwardCached(words, { ...configFor(), kvHeads: 3 }));
});

test('all forward stages agree with an independent two-dimensional scalar oracle', () => {
  const config = { layers: 2, qHeads: 2, kvHeads: 1, headDim: 2, hidden: 2, seed: 17 };
  const tokens = ['alpha', 'beta', 'alpha'];
  const actual = forwardCached(tokens, config);
  const inputs = actual.traces.map((trace) => trace.layers[0].input);
  const expected = twoDimensionalOracle(inputs, actual.model.layers);
  for (const [t, trace] of actual.traces.entries()) {
    for (const [l, layer] of trace.layers.entries()) {
      for (const field of [
        'input',
        'normalized',
        'q',
        'k',
        'v',
        'attentionOutput',
        'residual',
        'ffnInput',
        'output',
      ]) {
        assert.ok(
          maxDifference(layer[field], expected.traces[t][l][field]) < 1e-12,
          `t${t} L${l} ${field}`,
        );
      }
      for (let h = 0; h < config.qHeads; h++) {
        assert.equal(layer.heads[h].kvHead, 0);
        for (const field of ['scores', 'weights', 'context']) {
          assert.ok(
            maxDifference(layer.heads[h][field], expected.traces[t][l].heads[h][field]) < 1e-12,
            `t${t} L${l} Q${h} ${field}`,
          );
        }
      }
    }
  }
  assert.ok(maxDifference(forwardFull(tokens, config), expected.outputs) < 1e-12);
});

test('a one-token causal head has unit attention and returns its own value', () => {
  for (const variant of ['mha', 'gqa', 'mqa']) {
    for (const layer of forwardCached(['only'], configFor(variant)).traces[0].layers) {
      for (const head of layer.heads) {
        assert.deepEqual(head.weights, [1]);
        assert.deepEqual(head.context, layer.v[head.kvHead]);
      }
    }
  }
});

test('actual model menus including 8 heads, TP8, and 20 tokens preserve outputs and shapes', () => {
  const tokens = Array.from({ length: 20 }, (_, i) => `token-${i}`);
  for (const variant of ['mha', 'gqa', 'mqa']) {
    for (const layers of [2, 3, 4])
      for (const qHeads of [4, 8]) {
        const config = configFor(variant, layers, qHeads);
        const cached = forwardCached(tokens, config);
        assert.equal(cached.outputs.length, tokens.length);
        assert.ok(cached.outputs.every((output) => output.length === config.hidden));
        assert.ok(maxDifference(cached.outputs, forwardFull(tokens, config)) < 1e-12);
        for (const cache of cached.caches) {
          assert.equal(cache.keys.length, tokens.length);
          assert.equal(cache.values.length, tokens.length);
          for (const tensor of [cache.keys, cache.values]) {
            assert.ok(
              tensor.every(
                (heads) =>
                  heads.length === config.kvHeads &&
                  heads.every(
                    (vector) => vector.length === config.headDim && vector.every(Number.isFinite),
                  ),
              ),
            );
          }
        }
        for (const tp of [1, 2, 4, 8].filter((n) => n <= qHeads)) {
          for (const layer of cached.traces.at(-1).layers) {
            const partials = tpPartialOutputs(layer, cached.model.layers[layer.layer], config, tp);
            assert.equal(partials.length, tp);
            assert.ok(maxDifference(partials.reduce(add), layer.attentionOutput) < 1e-12);
          }
        }
      }
  }
});

test('public model interfaces reject malformed dimensions, tokens, indices, and TP', () => {
  const config = configFor();
  assert.throws(() => configFor('unknown'), /variant/);
  for (const field of ['layers', 'qHeads', 'kvHeads', 'headDim', 'hidden']) {
    for (const value of [0, -1, 1.5, NaN, Infinity, '4', Number.MAX_SAFE_INTEGER + 1]) {
      assert.throws(() => createModel({ ...config, [field]: value }), /Invalid/);
    }
  }
  assert.throws(() => createModel({ ...config, headDim: 3 }), /even/);
  assert.throws(() => createModel(null), /configuration/);
  for (const seed of [undefined, null, 1.5, NaN, Infinity, '42']) {
    assert.throws(() => createModel({ ...config, seed }), /seed/);
  }
  for (const tokens of [null, 'text', [1], [null], Array(1)]) {
    assert.throws(() => forwardCached(tokens), /Tokens/);
    assert.throws(() => forwardFull(tokens), /Tokens/);
  }
  for (const head of [-1, 0.5, 4, NaN, '0']) {
    assert.throws(() => kvHeadFor(head, config), /index/);
  }
  const result = forwardCached(words, config);
  const trace = result.traces[0].layers[0],
    weights = result.model.layers[0];
  for (const tp of [0, -1, 0.5, NaN, Infinity, '2', null, 3, 8]) {
    assert.throws(() => tpPartialOutputs(trace, weights, config, tp), /TP/);
  }
  assert.throws(() => tpPartialOutputs({ heads: [] }, weights, config, 2), /head outputs/);
  assert.throws(() => tpPartialOutputs(trace, { wO: [] }, config, 2), /projection/);
});

test('teaching tokenizer handles Unicode and whitespace without silently truncating input', () => {
  assert.deepEqual(tokenize(' \n\t '), []);
  assert.deepEqual(tokenize('你🙂好'), ['你', '🙂', '好']);
  assert.deepEqual(tokenize(' hello\n world\t ! '), ['hello', 'world', '!']);
  const tokens = Array.from({ length: 20 }, (_, i) => `t${i}`);
  assert.deepEqual(tokenize(tokens.join(' ')), tokens);
  assert.equal(tokenize('汉'.repeat(170)).length, 170);
  assert.throws(() => tokenize(null), /string/);
});
