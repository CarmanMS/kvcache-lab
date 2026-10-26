import {
  add,
  dot,
  matvec,
  rmsNorm,
  rope,
  softmax,
  random,
  matrix,
  embedding,
  splitHeads,
} from './math.js';

export const DEFAULT_CONFIG = Object.freeze({
  layers: 3,
  qHeads: 4,
  kvHeads: 2,
  headDim: 4,
  hidden: 8,
  seed: 42,
});

export function configFor(variant = 'gqa', layers = 3, qHeads = 4) {
  if (!['mha', 'gqa', 'mqa'].includes(variant)) throw new Error('Invalid attention variant');
  const config = {
    ...DEFAULT_CONFIG,
    layers,
    qHeads,
    kvHeads: variant === 'mha' ? qHeads : variant === 'mqa' ? 1 : qHeads / 2,
  };
  validateConfig(config);
  return config;
}
export function validateConfig(c) {
  if (!c || typeof c !== 'object') throw new TypeError('Expected a model configuration');
  for (const field of ['layers', 'qHeads', 'kvHeads', 'headDim', 'hidden']) {
    if (!Number.isSafeInteger(c[field]) || c[field] < 1) throw new Error(`Invalid ${field}`);
  }
  if (c.qHeads % c.kvHeads !== 0 || c.headDim % 2 !== 0) {
    throw new Error('Head groups must divide evenly; RoPE requires an even head dimension');
  }
}
export function validateTP(config, tp) {
  validateConfig(config);
  if (!Number.isSafeInteger(tp) || tp < 1 || config.qHeads % tp !== 0) {
    throw new Error('TP must be a positive integer that evenly divides Q heads');
  }
}
export function validateTokens(tokens) {
  if (!Array.isArray(tokens) || Array.from(tokens).some((token) => typeof token !== 'string')) {
    throw new TypeError('Tokens must be an array of strings');
  }
}
export function createModel(config = DEFAULT_CONFIG) {
  validateConfig(config);
  if (!Number.isSafeInteger(config.seed)) throw new Error('Invalid seed');
  const { hidden: d, qHeads: q, kvHeads: kv, headDim: h } = config;
  return {
    config,
    layers: Array.from({ length: config.layers }, (_, i) => {
      const rng = random(config.seed + 997 * (i + 1));
      return {
        wQ: matrix(d, q * h, rng),
        wK: matrix(d, kv * h, rng),
        wV: matrix(d, kv * h, rng),
        wO: matrix(q * h, d, rng),
        wGate: matrix(d, d * 2, rng),
        wUp: matrix(d, d * 2, rng),
        wDown: matrix(d * 2, d, rng),
      };
    }),
  };
}
export function kvHeadFor(qHead, config) {
  validateConfig(config);
  if (!Number.isSafeInteger(qHead) || qHead < 0 || qHead >= config.qHeads) {
    throw new Error('Q head index is out of range');
  }
  return Math.floor(qHead / (config.qHeads / config.kvHeads));
}

function project(input, position, weights, config) {
  const normalized = rmsNorm(input);
  const q = splitHeads(matvec(normalized, weights.wQ), config.headDim).map((head) =>
    rope(rmsNorm(head), position),
  );
  const k = splitHeads(matvec(normalized, weights.wK), config.headDim).map((head) =>
    rope(rmsNorm(head), position),
  );
  const v = splitHeads(matvec(normalized, weights.wV), config.headDim);
  return { normalized, q, k, v };
}

function attention(q, allK, allV, config) {
  return q.map((query, h) => {
    const kvHead = kvHeadFor(h, config);
    const scores = allK.map((keys) => dot(query, keys[kvHead]) / Math.sqrt(config.headDim));
    const weights = softmax(scores);
    const context = Array.from({ length: config.headDim }, (_, d) =>
      weights.reduce((sum, weight, t) => sum + weight * allV[t][kvHead][d], 0),
    );
    return { q: query, kvHead, scores, weights, context };
  });
}

function finish(input, heads, weights) {
  const attentionOutput = matvec(
    heads.flatMap((head) => head.context),
    weights.wO,
  );
  const residual = add(input, attentionOutput);
  const ffnInput = rmsNorm(residual);
  const gate = matvec(ffnInput, weights.wGate);
  const up = matvec(ffnInput, weights.wUp);
  const activated = gate.map((x, i) => (x / (1 + Math.exp(-x))) * up[i]);
  const output = add(residual, matvec(activated, weights.wDown));
  return { attentionOutput, residual, ffnInput, output };
}

/** Incremental causal execution; caches contain only keys and values. */
export function forwardCached(tokens, config = DEFAULT_CONFIG) {
  validateTokens(tokens);
  const model = createModel(config);
  const caches = model.layers.map(() => ({ keys: [], values: [] }));
  const traces = tokens.map((token, position) => {
    let hidden = embedding(token, config.hidden);
    const layers = model.layers.map((weights, layer) => {
      const input = hidden;
      const projection = project(input, position, weights, config);
      caches[layer].keys.push(projection.k);
      caches[layer].values.push(projection.v);
      const heads = attention(projection.q, caches[layer].keys, caches[layer].values, config);
      const result = finish(input, heads, weights);
      hidden = result.output;
      return { layer, input, ...projection, heads, ...result };
    });
    return { token, position, layers, output: hidden };
  });
  return { model, caches, traces, outputs: traces.map((trace) => trace.output) };
}

/** Independent layer-major reference: project every token and apply a causal mask. */
export function forwardFull(tokens, config = DEFAULT_CONFIG) {
  validateTokens(tokens);
  const model = createModel(config);
  let hidden = tokens.map((token) => embedding(token, config.hidden));
  for (const layer of model.layers) {
    const projected = hidden.map((input, t) => project(input, t, layer, config));
    hidden = hidden.map((input, t) => {
      const visible = projected.slice(0, t + 1);
      const heads = attention(
        projected[t].q,
        visible.map((p) => p.k),
        visible.map((p) => p.v),
        config,
      );
      return finish(input, heads, layer).output;
    });
  }
  return hidden;
}

export function tpPartialOutputs(trace, layerWeights, config, tp) {
  validateTP(config, tp);
  const finiteVector = (values, length) =>
    Array.isArray(values) && values.length === length && Array.from(values).every(Number.isFinite);
  if (
    !Array.isArray(trace?.heads) ||
    trace.heads.length !== config.qHeads ||
    Array.from(trace.heads).some((head) => !finiteVector(head?.context, config.headDim))
  ) {
    throw new Error('Invalid attention head outputs');
  }
  if (
    !Array.isArray(layerWeights?.wO) ||
    layerWeights.wO.length !== config.qHeads * config.headDim ||
    Array.from(layerWeights.wO).some((row) => !finiteVector(row, config.hidden))
  ) {
    throw new Error('Invalid output projection weights');
  }
  const headsPerRank = config.qHeads / tp;
  const contexts = trace.heads.flatMap((head) => head.context);
  return Array.from({ length: tp }, (_, rank) => {
    const lo = rank * headsPerRank * config.headDim;
    const hi = lo + headsPerRank * config.headDim;
    return matvec(contexts.slice(lo, hi), layerWeights.wO.slice(lo, hi));
  });
}

export function tokenize(text) {
  if (typeof text !== 'string') throw new TypeError('Input text must be a string');
  const clean = text.trim();
  if (!clean) return [];
  return /\s/u.test(clean) ? clean.split(/\s+/u) : Array.from(clean);
}
