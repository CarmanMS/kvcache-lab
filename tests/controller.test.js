import test from 'node:test';
import assert from 'node:assert/strict';
import { Experiment, validatePrompt } from '../src/controller.js';

test('over-limit input is rejected without changing the experiment or truncating the draft', () => {
  const e = new Experiment(),
    original = structuredClone(e.state.prompt);
  const draft = Array.from({ length: 20 }, (_, i) => `t${i}`).join(' ');
  e.updateDraft(draft);
  assert.equal(e.applyDraft(), false);
  assert.deepEqual(e.state.prompt, original);
  assert.equal(e.state.draft, draft);
  assert.match(e.state.draftError, /20.*16/);
  assert.ok(validatePrompt('x'.repeat(161)).error);
  assert.ok(validatePrompt('   ').error);
});
test('simulation undo restores allocation but preserves the teacher’s draft and view', () => {
  const e = new Experiment(),
    before = e.pool.checkpoint();
  e.advance();
  e.state.view = 'devices';
  e.state.editorOpen = true;
  e.updateDraft('未提交 的 草稿');
  assert.equal(e.undo(), true);
  assert.equal(e.state.cursor.A, 6);
  assert.deepEqual(e.pool.checkpoint(), before);
  assert.equal(e.state.view, 'devices');
  assert.equal(e.state.editorOpen, true);
  assert.equal(e.state.draft, '未提交 的 草稿');
});
test('all model variants and device controls produce finite valid data', () => {
  const e = new Experiment();
  for (const variant of ['mha', 'gqa', 'mqa'])
    for (const qHeads of [4, 8]) {
      e.setVariant(variant);
      e.setConfig('qHeads', qHeads);
      for (const tp of [1, 2, 4, 8].filter((n) => n <= qHeads)) {
        e.setConfig('tp', tp);
        const d = e.data();
        assert.equal(d.ranks.length, tp);
        assert.ok(!d.logicalBytesLabel.includes('NaN'));
      }
    }
  assert.throws(() => e.setConfig('layers', -1));
  assert.throws(() => e.setVariant('unknown'));
});
test('ended and evicted requests remain ended until explicit reset', () => {
  const e = new Experiment();
  e.setScenario('prefix');
  e.finish('A');
  assert.equal(e.evict(), 1);
  assert.ok(e.data().pool.mappings.A.includes(null));
  e.selectRequest('B');
  e.advance();
  assert.ok(e.data().pool.mappings.A.includes(null));
});
