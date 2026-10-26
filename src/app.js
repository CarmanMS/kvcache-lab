import { Experiment, MAX_PROMPT_TOKENS, validatePrompt } from './controller.js';
import { kvHeadFor, tokenize } from './model.js';
import { createViews } from './views.js';

/** Mount the workspace with a testable controller and explicit DOM lifecycle. */
export function mountApp(root) {
  const document = root.ownerDocument,
    browser = document.defaultView;
  const experiment = new Experiment(),
    state = experiment.state;
  const views = createViews(state, () => experiment.canUndo);
  const { icon, button, esc } = views;
  const listeners = new browser.AbortController(),
    scrollPositions = new Map();
  let timer = null,
    toastTimer = null;
  const scrollAreas = () => [...root.querySelectorAll('.token-stream,.model-scroll,.gpu-layers')];
  const scrollKey = (el, i) => el.dataset.scrollKey || `${el.className}:${i}`;

  function keepVisible(container, element) {
    if (!container || !element) return;
    const frame = container.getBoundingClientRect(),
      target = element.getBoundingClientRect();
    if (target.left < frame.left) container.scrollLeft += target.left - frame.left - 10;
    else if (target.right > frame.right) container.scrollLeft += target.right - frame.right + 10;
    if (container.classList.contains('gpu-layers')) {
      if (target.top < frame.top) container.scrollTop += target.top - frame.top - 5;
      else if (target.bottom > frame.bottom)
        container.scrollTop += target.bottom - frame.bottom + 5;
    }
  }

  function render({ reveal = false, focusPrompt = false } = {}) {
    const active = document.activeElement;
    const focusKeys =
      active && root.contains(active)
        ? [...active.attributes]
            .filter(
              (a) =>
                ['id', 'name', 'type', 'aria-label'].includes(a.name) || a.name.startsWith('data-'),
            )
            .map((a) => [a.name, a.value])
        : [];
    const selection =
      active?.tagName === 'INPUT' && active.type === 'text'
        ? [active.selectionStart, active.selectionEnd]
        : null;
    scrollAreas().forEach((el, i) =>
      scrollPositions.set(scrollKey(el, i), { left: el.scrollLeft, top: el.scrollTop }),
    );
    const d = experiment.data();
    root.innerHTML = `<header class="app-header"><a class="brand" href="#"><span class="brand-mark">K<span>V</span></span><span>KVCache <em>Lab</em><small>看见模型的记忆</small></span></a><nav aria-label="观察视角">${[
      ['model', '模型内部'],
      ['devices', '多卡部署'],
      ['cache', '请求与缓存'],
    ]
      .map(([v, label]) =>
        button(
          'view',
          `${icon(v)}${label}`,
          `data-value="${v}" class="${state.view === v ? 'active' : ''}" aria-current="${state.view === v ? 'page' : 'false'}"`,
        ),
      )
      .join(
        '',
      )}</nav><span class="local-badge"><span class="live-dot"></span> 浏览器内运行 <b>v0.1</b></span></header>
      <div class="app-layout">${views.sidebar(d)}<main id="workspace" tabindex="-1"><div class="workspace-intro"><div><span class="eyebrow">AN INTERACTIVE FIELD GUIDE</span><h1>记忆，是怎样留下来的？</h1></div><div class="live-stats"><span>教学模型 <b>${state.layers}L · ${state.qHeads}Q · ${d.config.kvHeads}KV</b></span><span>逻辑 KV 数据（去重） <b>${d.logicalBytesLabel}</b></span></div></div>
      ${views.runtime(d)}<div class="workspace-body"><section class="canvas" aria-label="交互实验画布">${state.view === 'model' ? views.modelView(d) : state.view === 'devices' ? views.devicesView(d) : views.cacheView(d)}</section>${views.inspector(d)}</div>${views.capacity(d)}<footer class="app-footer"><span>KVCache Lab · 从计算到存储</span><span>教学模拟 / 数值可复现 / 无服务器依赖</span></footer></main></div><div class="toast ${state.notice ? 'visible' : ''}" role="status" aria-live="polite">${esc(state.notice)}</div>`;
    scrollAreas().forEach((el, i) => {
      const previous = scrollPositions.get(scrollKey(el, i));
      if (previous) {
        el.scrollLeft = previous.left;
        el.scrollTop = previous.top;
      }
    });
    let focus = focusPrompt
      ? root.querySelector('input[name="prompt"]')
      : focusKeys.length
        ? [...root.querySelectorAll('button,select,input,summary')].find((el) =>
            focusKeys.every(([key, value]) => el.getAttribute(key) === value),
          )
        : null;
    if (focus?.disabled) focus = root.querySelector('#runtime-controls');
    focus?.focus({ preventScroll: true });
    if (selection && focus?.tagName === 'INPUT' && focus.type === 'text')
      focus.setSelectionRange(...selection);
    if (reveal) {
      keepVisible(
        root.querySelector('.model-scroll'),
        root.querySelector('.selected-layer .selected-head'),
      );
      keepVisible(root.querySelector('.token-stream'), root.querySelector('.token.selected'));
      root
        .querySelectorAll('.gpu-layers')
        .forEach((el) => keepVisible(el, el.querySelector('.gpu-layer.selected')));
    }
  }
  function stop() {
    browser.clearInterval(timer);
    timer = null;
    state.playing = false;
  }
  function notify(message) {
    state.notice = message;
    browser.clearTimeout(toastTimer);
    toastTimer = browser.setTimeout(() => {
      state.notice = '';
      const toast = root.querySelector('.toast');
      toast?.classList.remove('visible');
      if (toast) toast.textContent = '';
    }, 3500);
  }
  function selectKV(layer, kv, rankIndex) {
    const d = experiment.data();
    state.layer = layer;
    state.head =
      rankIndex == null
        ? (kv * state.qHeads) / d.config.kvHeads
        : d.ranks[rankIndex].qHeads.find((head) => kvHeadFor(head, d.config) === kv);
  }
  function onClick(event) {
    const target = event.target.closest('[data-action]');
    if (!target || target.disabled || !root.contains(target)) return;
    const { action, value } = target.dataset;
    try {
      switch (action) {
        case 'view':
          state.view = value;
          break;
        case 'variant':
          stop();
          experiment.setVariant(value);
          break;
        case 'request':
          stop();
          experiment.selectRequest(value);
          break;
        case 'layer':
          state.layer = Number(value);
          break;
        case 'head':
          state.layer = Number(target.dataset.layer);
          state.head = Number(target.dataset.head);
          break;
        case 'device-head':
          state.head = Number(value);
          break;
        case 'gpu-kv':
          selectKV(
            Number(target.dataset.layer),
            Number(target.dataset.kv),
            Number(target.dataset.rank),
          );
          break;
        case 'kv':
          selectKV(Number(target.dataset.layer), Number(target.dataset.kv));
          break;
        case 'history':
          selectKV(Number(target.dataset.layer), Number(target.dataset.kv));
          state.history = Number(target.dataset.token);
          state.query = Math.max(state.query, state.history);
          break;
        case 'token':
          state.query = Number(value);
          break;
        case 'attend':
          state.history = Number(value);
          break;
        case 'slot': {
          const d = experiment.data(),
            slot = Number(target.dataset.slot);
          let t = d.pool.mappings[state.request].indexOf(slot);
          if (t < 0) {
            stop();
            experiment.selectRequest(state.request === 'A' ? 'B' : 'A');
            t = d.pool.mappings[state.request].indexOf(slot);
          }
          if (t >= 0) {
            state.history = t;
            state.query = Math.max(state.query, t);
          }
          break;
        }
        case 'step':
          stop();
          experiment.advance();
          break;
        case 'prefill':
          stop();
          experiment.prefill();
          break;
        case 'prefill-request':
          stop();
          experiment.prefill(value);
          break;
        case 'back':
          stop();
          experiment.undo();
          break;
        case 'reset':
          stop();
          experiment.reset();
          notify('实验已重置。先运行 prefill，或逐 token 执行。');
          break;
        case 'play':
          if (state.playing) stop();
          else {
            state.playing = true;
            experiment.advance();
            if (state.playing)
              timer = browser.setInterval(() => {
                experiment.advance();
                if (!state.playing) stop();
                render({ reveal: true });
              }, 1100);
          }
          break;
        case 'zoom':
          state.zoom = Math.min(
            1.3,
            Math.max(0.7, Math.round((state.zoom + Number(value) * 0.1) * 10) / 10),
          );
          break;
        case 'scenario':
          stop();
          experiment.setScenario(value);
          notify('已载入两个请求，可比较前缀和逐层 K。');
          break;
        case 'finish-request':
          stop();
          experiment.finish(value);
          notify(`请求 ${value} 已结束，无活动引用的页可以回收。`);
          break;
        case 'evict':
          notify(`已回收 ${experiment.evict()} 页，活动引用页保留。`);
          break;
      }
      experiment.normalizeSelection();
      render({
        reveal: [
          'view',
          'layer',
          'head',
          'device-head',
          'gpu-kv',
          'kv',
          'history',
          'token',
          'request',
          'step',
          'prefill',
          'back',
        ].includes(action),
      });
    } catch (error) {
      stop();
      notify(error.message);
      render();
    }
  }
  function onChange(event) {
    const field = event.target.dataset.config;
    if (!field) return;
    stop();
    try {
      experiment.setConfig(
        field,
        field === 'sameIdentity' ? event.target.checked : Number(event.target.value),
      );
    } catch (error) {
      notify(error.message);
    }
    render({ reveal: true });
  }
  function onInput(event) {
    if (event.target.name !== 'prompt') return;
    experiment.updateDraft(event.target.value);
    if (state.draftError) state.draftError = validatePrompt(state.draft).error;
    const counter = root.querySelector('#prompt-count');
    if (counter)
      counter.textContent = `${tokenize(state.draft).length} / ${MAX_PROMPT_TOKENS} token`;
    const error = root.querySelector('#prompt-error');
    if (error) {
      error.textContent = state.draftError;
      error.hidden = !state.draftError;
    }
    event.target.setAttribute('aria-invalid', String(Boolean(state.draftError)));
  }
  function onSubmit(event) {
    if (event.target.id !== 'prompt-form') return;
    event.preventDefault();
    stop();
    experiment.updateDraft(new browser.FormData(event.target).get('prompt'));
    if (experiment.applyDraft()) notify(`已载入 ${state.prompt.length} 个教学 token。`);
    state.editorOpen = true;
    render({ focusPrompt: true });
  }
  root.addEventListener('click', onClick, { signal: listeners.signal });
  root.addEventListener('change', onChange, { signal: listeners.signal });
  root.addEventListener('input', onInput, { signal: listeners.signal });
  root.addEventListener('submit', onSubmit, { signal: listeners.signal });
  root.addEventListener(
    'toggle',
    (event) => {
      if (event.target.matches('details.prompt-editor')) state.editorOpen = event.target.open;
    },
    { capture: true, signal: listeners.signal },
  );
  document.addEventListener(
    'visibilitychange',
    () => {
      if (document.hidden && state.playing) {
        stop();
        render();
      }
    },
    { signal: listeners.signal },
  );
  render();
  return {
    experiment,
    render,
    destroy() {
      stop();
      browser.clearTimeout(toastTimer);
      listeners.abort();
    },
  };
}
if (typeof document !== 'undefined' && document.querySelector('#app'))
  mountApp(document.querySelector('#app'));
