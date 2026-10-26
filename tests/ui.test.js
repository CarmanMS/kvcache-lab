import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { mountApp } from '../src/app.js';

function fixture(t) {
  const dom = new JSDOM('<!doctype html><div id="app"></div>', { url: 'http://localhost/' });
  const root = dom.window.document.querySelector('#app'),
    mounted = mountApp(root);
  t.after(() => {
    mounted.destroy();
    dom.window.close();
  });
  const click = (selector) => {
    const el = root.querySelector(selector);
    assert.ok(el, selector);
    assert.equal(el.disabled, false);
    el.click();
  };
  const draft = (text) => {
    const details = root.querySelector('details');
    details.open = true;
    details.dispatchEvent(new dom.window.Event('toggle'));
    const input = root.querySelector('input[name="prompt"]');
    input.value = text;
    input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
    return input;
  };
  return { dom, root, mounted, click, draft };
}
test('draft and editor remain intact across stepping, changing heads and undo', (t) => {
  const f = fixture(t);
  f.draft('老师 的 草稿 尚未提交');
  f.click('[data-action="step"]');
  f.click('[data-action="head"][data-layer="2"][data-head="3"]');
  f.click('[data-action="back"]');
  assert.equal(f.root.querySelector('details').open, true);
  assert.equal(f.root.querySelector('input[name="prompt"]').value, '老师 的 草稿 尚未提交');
});
test('render preserves scroll offsets and text caret during playback-like updates', (t) => {
  const f = fixture(t),
    input = f.draft('老师 的 新输入');
  input.focus();
  input.setSelectionRange(2, 4);
  f.root.querySelector('.model-scroll').scrollLeft = 350;
  f.root.querySelector('.token-stream').scrollLeft = 120;
  f.mounted.experiment.advance();
  f.mounted.render();
  assert.equal(f.root.querySelector('.model-scroll').scrollLeft, 350);
  assert.equal(f.root.querySelector('.token-stream').scrollLeft, 120);
  const next = f.root.querySelector('input[name="prompt"]');
  assert.equal(f.dom.window.document.activeElement, next);
  assert.deepEqual([next.selectionStart, next.selectionEnd], [2, 4]);
});
test('over-limit submission retains all input, reports an accessible error and preserves focus', (t) => {
  const f = fixture(t),
    text = Array.from({ length: 20 }, (_, i) => `t${i}`).join(' ');
  f.draft(text);
  f.click('#apply-prompt');
  assert.equal(f.mounted.experiment.state.cursor.A, 6);
  assert.equal(f.root.querySelector('input[name="prompt"]').value, text);
  assert.equal(f.root.querySelector('#prompt-error').hidden, false);
  assert.match(f.root.querySelector('#prompt-error').textContent, /20.*16/);
  assert.equal(f.dom.window.document.activeElement.name, 'prompt');
});
test('valid custom input is escaped, applied completely and remains editable', (t) => {
  const f = fixture(t);
  f.draft('hello <img src=x>');
  f.click('#apply-prompt');
  assert.deepEqual(f.mounted.experiment.state.prompt, ['hello', '<img', 'src=x>']);
  assert.equal(f.root.querySelectorAll('img').length, 0);
  assert.equal(f.root.querySelector('details').open, true);
  assert.equal(f.dom.window.document.activeElement.id, 'prompt-input');
});
test('all views expose working controls and no fixed classroom questions', (t) => {
  const f = fixture(t);
  assert.equal(f.root.querySelectorAll('.q-head').length, 12);
  assert.equal(f.root.querySelectorAll('.questions,[data-action="lesson"]').length, 0);
  f.click('[data-action="view"][data-value="devices"]');
  assert.equal(f.root.querySelectorAll('.gpu-card').length, 2);
  f.click('[data-action="view"][data-value="cache"]');
  assert.equal(f.root.querySelectorAll('.request-card').length, 2);
  assert.equal(
    f.root
      .querySelector('[data-action="scenario"][data-value="prefix"]')
      .getAttribute('aria-pressed'),
    'true',
  );
});
test('mixed independent and shared pages highlight the actual shared positions', (t) => {
  const f = fixture(t);
  f.click('[data-action="request"][data-value="B"]');
  f.click('[data-action="step"]');
  f.click('[data-action="prefill"]');
  f.click('[data-action="view"][data-value="cache"]');
  const positions = [...f.root.querySelector('.request-card .request-tokens').children].flatMap(
    (el, i) => (el.classList.contains('shared-token') ? [i] : []),
  );
  assert.deepEqual(positions, [2, 3]);
});
