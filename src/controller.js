import { configFor, forwardCached, kvHeadFor, tokenize } from './model.js';
import { CachePool, rankLayout, bytesPerToken, formatBytes } from './cache.js';

export const DEFAULT_PROMPT = Object.freeze(['请', '翻译', '：', '今天', '下雨', '了']);
export const CONTINUATION = Object.freeze(['记得', '带伞', '。', '再见']);
export const MAX_PROMPT_TOKENS = 16;
export const MAX_PROMPT_CHARACTERS = 160;

export function validatePrompt(text) {
  if (typeof text !== 'string') return { error: '请输入文本。' };
  if (Array.from(text).length > MAX_PROMPT_CHARACTERS) {
    return { error: `输入最多 ${MAX_PROMPT_CHARACTERS} 个字符，请缩短后再应用。` };
  }
  const tokens = tokenize(text);
  if (!tokens.length) return { error: '请至少输入一个教学 token。' };
  if (tokens.length > MAX_PROMPT_TOKENS) {
    return {
      error: `输入有 ${tokens.length} 个教学 token，最多 ${MAX_PROMPT_TOKENS} 个。请调整输入，内容尚未应用。`,
    };
  }
  return { tokens, error: '' };
}

/** Application state independent of DOM rendering and playback timers. */
export class Experiment {
  constructor() {
    this.state = {
      view: 'model',
      variant: 'gqa',
      layers: 3,
      qHeads: 4,
      tp: 2,
      gpusPerHost: 1,
      dtype: 2,
      pageSize: 2,
      context: 32768,
      prompt: [...DEFAULT_PROMPT],
      request: 'A',
      cursor: { A: 6, B: 0 },
      active: { A: true, B: true },
      layer: 1,
      head: 1,
      query: 5,
      history: 3,
      scenario: 'prefix',
      sameIdentity: true,
      zoom: 1,
      playing: false,
      notice: '',
      draft: DEFAULT_PROMPT.join(' '),
      editorOpen: false,
      draftError: '',
    };
    this.pool = new CachePool(this.state.pageSize);
    this.pool.extend('A', this.state.prompt);
    this.undoStack = [];
    this.memo = new Map();
  }

  get canUndo() {
    return this.undoStack.length > 0;
  }

  sequences() {
    const b = [...this.state.prompt];
    if (this.state.scenario === 'prefix') b[Math.max(0, b.length - 2)] = '晴天';
    else b[0] = b[0] === '改写' ? '分析' : '改写';
    return { A: [...this.state.prompt, ...CONTINUATION], B: [...b, ...CONTINUATION] };
  }

  normalizeSelection() {
    const s = this.state;
    s.layer = Math.max(0, Math.min(s.layer, s.layers - 1));
    s.head = Math.max(0, Math.min(s.head, s.qHeads - 1));
    s.query = Math.max(0, Math.min(s.query, s.cursor[s.request] - 1));
    s.history = Math.max(0, Math.min(s.history, s.query));
  }

  data() {
    const s = this.state;
    const config = configFor(s.variant, s.layers, s.qHeads);
    const seqs = this.sequences();
    const results = Object.fromEntries(
      ['A', 'B'].map((request) => {
        const c =
          request === 'B' && !s.sameIdentity ? { ...config, seed: config.seed + 73 } : config;
        const tokens = seqs[request].slice(0, s.cursor[request]);
        const key = JSON.stringify([tokens, c]);
        if (!this.memo.has(key)) {
          if (this.memo.size >= 32) this.memo.clear();
          this.memo.set(key, forwardCached(tokens, c));
        }
        return [request, this.memo.get(key)];
      }),
    );
    return {
      config,
      seqs,
      results,
      pool: this.pool.snapshot(),
      ranks: rankLayout(config, s.tp, s.gpusPerHost),
      count: s.cursor[s.request],
      result: results[s.request],
      kv: kvHeadFor(s.head, config),
      logicalBytesLabel: formatBytes(
        this.pool.snapshot().usedSlots * bytesPerToken(config, s.dtype),
      ),
    };
  }

  remember() {
    this.undoStack.push({
      state: structuredClone({ ...this.state, playing: false, notice: '' }),
      pool: this.pool.checkpoint(),
    });
    if (this.undoStack.length > 100) this.undoStack.shift();
  }

  undo() {
    const previous = this.undoStack.pop();
    if (!previous) return false;
    // Undo simulation operations while preserving the teacher's view and input draft.
    const ui = Object.fromEntries(
      [
        'view',
        'zoom',
        'tp',
        'gpusPerHost',
        'dtype',
        'context',
        'draft',
        'editorOpen',
        'draftError',
      ].map((key) => [key, this.state[key]]),
    );
    Object.assign(this.state, previous.state, ui, { playing: false, notice: '' });
    this.pool.restore(previous.pool);
    this.normalizeSelection();
    return true;
  }

  syncPool(request) {
    const s = this.state;
    this.pool.extend(
      request,
      this.sequences()[request].slice(0, s.cursor[request]),
      request === 'B' && !s.sameIdentity ? 'other' : 'base',
    );
  }

  reset({ prefilled = false, both = false } = {}) {
    const s = this.state;
    s.playing = false;
    s.cursor = { A: prefilled ? s.prompt.length : 0, B: both ? s.prompt.length : 0 };
    s.active = { A: true, B: true };
    s.query = Math.max(0, s.cursor[s.request] - 1);
    s.history = 0;
    this.pool = new CachePool(s.pageSize);
    this.syncPool('A');
    this.syncPool('B');
    this.undoStack.length = 0;
    this.normalizeSelection();
  }

  advance() {
    const s = this.state,
      r = s.request;
    if (!s.active[r] || s.cursor[r] >= this.sequences()[r].length) {
      s.playing = false;
      return false;
    }
    this.remember();
    s.cursor[r]++;
    s.query = s.cursor[r] - 1;
    this.syncPool(r);
    this.normalizeSelection();
    if (s.cursor[r] >= this.sequences()[r].length) s.playing = false;
    return true;
  }

  prefill(request = this.state.request) {
    this.validateRequest(request);
    const s = this.state;
    if (!s.active[request] || s.cursor[request] >= s.prompt.length) return false;
    this.remember();
    s.cursor[request] = s.prompt.length;
    this.syncPool(request);
    if (request === s.request) s.query = s.cursor[request] - 1;
    this.normalizeSelection();
    return true;
  }

  finish(request) {
    this.validateRequest(request);
    if (!this.state.active[request] || !this.state.cursor[request]) return false;
    this.remember();
    this.state.active[request] = false;
    this.pool.finish(request);
    return true;
  }

  evict() {
    if (!this.pool.snapshot().pages.some((page) => page.refs === 0)) return 0;
    this.remember();
    return this.pool.evictUnused();
  }

  validateRequest(request) {
    if (!['A', 'B'].includes(request)) throw new Error('Unknown request');
  }

  selectRequest(request) {
    this.validateRequest(request);
    this.state.request = request;
    this.state.query = Math.max(0, this.state.cursor[request] - 1);
    this.normalizeSelection();
  }

  setVariant(variant) {
    if (!['mha', 'gqa', 'mqa'].includes(variant)) throw new Error('Unknown attention variant');
    this.state.variant = variant;
    this.reset({ prefilled: true });
  }

  setScenario(scenario) {
    if (!['prefix', 'suffix'].includes(scenario)) throw new Error('Unknown request scenario');
    this.state.scenario = scenario;
    this.reset({ prefilled: true, both: true });
  }

  setConfig(field, value) {
    const allowed = {
      layers: [2, 3, 4],
      qHeads: [4, 8],
      tp: [1, 2, 4, 8],
      gpusPerHost: [1, 2, 4],
      dtype: [1, 2],
      pageSize: [2, 4],
    };
    if (field === 'sameIdentity') {
      if (typeof value !== 'boolean') throw new Error('Model identity must be boolean');
    } else if (field === 'context') {
      if (!Number.isInteger(value) || value < 1024 || value > 32768 || value % 1024) {
        throw new Error('Invalid reference context length');
      }
    } else if (!allowed[field]?.includes(value)) throw new Error('Invalid configuration value');
    if (field === 'tp' && value > this.state.qHeads) throw new Error('TP exceeds Q heads');
    this.state[field] = value;
    if (field === 'qHeads') this.state.tp = Math.min(this.state.tp, value);
    if (['layers', 'qHeads'].includes(field)) this.reset({ prefilled: true });
    if (['pageSize', 'sameIdentity'].includes(field)) this.reset({ prefilled: true, both: true });
    this.normalizeSelection();
  }

  updateDraft(text) {
    this.state.draft = text;
  }

  applyDraft() {
    const result = validatePrompt(this.state.draft);
    this.state.draftError = result.error;
    if (result.error) return false;
    this.state.prompt = result.tokens;
    this.state.request = 'A';
    this.reset({ prefilled: true });
    return true;
  }
}
