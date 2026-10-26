export const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const matvec = (x, weight) =>
  weight[0].map((_, j) => x.reduce((sum, value, i) => sum + value * weight[i][j], 0));
export const rmsNorm = (x) => {
  const scale = 1 / Math.sqrt(dot(x, x) / x.length + 1e-6);
  return x.map((value) => value * scale);
};
export function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((value) => value / sum);
}
export function rope(x, position) {
  const result = [...x];
  for (let i = 0; i < x.length; i += 2) {
    const angle = position / 10000 ** (i / x.length);
    result[i] = x[i] * Math.cos(angle) - x[i + 1] * Math.sin(angle);
    result[i + 1] = x[i] * Math.sin(angle) + x[i + 1] * Math.cos(angle);
  }
  return result;
}
export function hash(text) {
  let h = 2166136261;
  for (const char of text) h = Math.imul(h ^ char.codePointAt(0), 16777619);
  return h >>> 0;
}
export function random(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let x = Math.imul(state ^ (state >>> 15), 1 | state);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
export function matrix(rows, cols, rng) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rng() * 2 - 1) / Math.sqrt(rows)),
  );
}
export function embedding(token, width) {
  const rng = random(hash(token));
  return Array.from({ length: width }, () => rng() * 2 - 1);
}
export const splitHeads = (x, dim) =>
  Array.from({ length: x.length / dim }, (_, i) => x.slice(i * dim, (i + 1) * dim));
/** Compare equally shaped numeric tensors without discarding their dimensions. */
export function maxDifference(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) {
    throw new TypeError('Expected two numeric tensors');
  }
  let maximum = 0;
  function compare(left, right) {
    if (Array.isArray(left) || Array.isArray(right)) {
      if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
        throw new Error('Tensor shapes must match');
      }
      for (let i = 0; i < left.length; i++) compare(left[i], right[i]);
      return;
    }
    if (!Number.isFinite(left) || !Number.isFinite(right)) {
      throw new TypeError('Tensor values must be finite numbers');
    }
    maximum = Math.max(maximum, Math.abs(left - right));
  }
  compare(a, b);
  return maximum;
}
