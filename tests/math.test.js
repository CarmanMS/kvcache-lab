import test from 'node:test';
import assert from 'node:assert/strict';
import {
  add,
  dot,
  matvec,
  rmsNorm,
  softmax,
  rope,
  splitHeads,
  maxDifference,
} from '../src/math.js';

test('vector and matrix operations agree with hand-calculated values', () => {
  assert.equal(dot([1, -2, 3], [4, 5, -1]), -9);
  assert.deepEqual(add([1, -2, 3], [4, 5, -1]), [5, 3, 2]);
  assert.deepEqual(
    matvec(
      [2, 3],
      [
        [1, 4, -1],
        [2, 5, 6],
      ],
    ),
    [8, 23, 16],
  );
  assert.deepEqual(splitHeads([1, 2, 3, 4], 2), [
    [1, 2],
    [3, 4],
  ]);
  const expectedNorm = [3 / Math.sqrt(12.500001), 4 / Math.sqrt(12.500001)];
  assert.ok(maxDifference(rmsNorm([3, 4]), expectedNorm) < 1e-14);
});

test('stable softmax preserves known ratios after a large constant shift', () => {
  const expected = [1 / 6, 2 / 6, 3 / 6];
  const values = [1000, 1000 + Math.log(2), 1000 + Math.log(3)];
  assert.ok(maxDifference(softmax(values), expected) < 1e-13);
});

test('RoPE rotates known coordinates at the correct per-pair frequencies', () => {
  const input = [1, 0, 0, 1];
  assert.deepEqual(rope(input, 0), input);
  // At p = pi/2, pair 0 rotates by pi/2; pair 1 rotates by pi/200.
  const expected = [0, 1, -Math.sin(Math.PI / 200), Math.cos(Math.PI / 200)];
  assert.ok(maxDifference(rope(input, Math.PI / 2), expected) < 1e-14);
  assert.deepEqual(input, [1, 0, 0, 1]);
});

test('RoPE preserves vector length and its dot product depends on relative position', () => {
  const q = [1, 2, -3, 4],
    k = [-2, 3, 1, -4];
  assert.ok(Math.abs(dot(rope(q, 7), rope(q, 7)) - dot(q, q)) < 1e-12);
  const relative = dot(q, rope(k, 3 - 7));
  assert.ok(Math.abs(dot(rope(q, 7), rope(k, 3)) - relative) < 1e-12);
});

test('maxDifference compares complete shapes and rejects nonfinite or missing values', () => {
  assert.equal(maxDifference([], []), 0);
  assert.equal(
    maxDifference(
      [
        [1, 2],
        [3, 4],
      ],
      [
        [1, 2],
        [3, 5],
      ],
    ),
    1,
  );
  for (const [a, b] of [
    [[], [1]],
    [[1], [1, 2]],
    [[[1, 2]], [[1], [2]]],
    [[[1]], [1]],
    [[1], [[1]]],
  ]) {
    assert.throws(() => maxDifference(a, b), /shapes/);
    assert.throws(() => maxDifference(b, a), /shapes/);
  }
  for (const value of [NaN, Infinity, -Infinity, undefined, null, '1']) {
    assert.throws(() => maxDifference([value], [1]), /finite/);
    assert.throws(() => maxDifference([1], [value]), /finite/);
  }
  assert.throws(() => maxDifference(Array(1), [1]), /finite/);
  assert.throws(() => maxDifference(1, [1]), /tensors/);
  assert.equal(maxDifference(Array(100000).fill(1), Array(100000).fill(2)), 1);
});
