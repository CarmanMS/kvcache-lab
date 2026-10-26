import { kvHeadFor, tpPartialOutputs, tokenize } from './model.js';
import { rankLayout, bytesPerToken, formatBytes, QWEN3_32B, QWEN3_32B_SOURCE } from './cache.js';
import { maxDifference } from './math.js';

export function createViews(state, canUndo) {
  const esc = (text) =>
    String(text).replace(
      /[&<>"']/g,
      (x) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[x],
    );
  const n = (value) => Number(value).toFixed(3).replace('-0.000', '0.000');
  const range = (length) => Array.from({ length }, (_, i) => i);
  const icons = {
    model:
      '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
    devices:
      '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="14" width="18" height="7" rx="2"/><path d="M7 7.5h1m3 0h6M7 17.5h1m3 0h6"/>',
    cache:
      '<path d="M4 6h16M4 12h16M4 18h16"/><rect x="4" y="3" width="4" height="6" rx="1"/><rect x="10" y="9" width="4" height="6" rx="1"/><rect x="16" y="15" width="4" height="6" rx="1"/>',
    step: '<path d="m6 4 10 8L6 20zM19 4v16"/>',
    play: '<path d="m7 4 12 8-12 8z"/>',
    reset: '<path d="M4 11a8 8 0 1 1 2 7M4 4v7h7"/>',
    arrow: '<path d="M4 12h15m-5-5 5 5-5 5"/>',
    eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  };
  const icon = (name) =>
    `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">${icons[name] || icons.arrow}</svg>`;
  const button = (action, text, attrs = '') =>
    `<button type="button" data-action="${action}" ${attrs}>${text}</button>`;

  function vector(values, kind = '', labels = false) {
    if (!values) return '';
    return `<span class="vector ${kind} ${labels ? 'number-vector' : ''}">${values
      .map(
        (v, i) =>
          `<span title="维度 ${i}: ${n(v)}" style="--value:${Math.min(1, Math.abs(v) / 2)};--signed:${v >= 0 ? 'var(--teal)' : 'var(--orange)'}">${labels ? n(v) : ''}</span>`,
      )
      .join('')}</span>`;
  }
  function option(value, text, selected) {
    return `<option value="${value}" ${value === selected ? 'selected' : ''}>${text}</option>`;
  }
  function sidebar(d) {
    return `<aside class="settings" aria-label="实验参数">
    <div class="panel-label">EXPERIMENT SETUP <span>01</span></div>
    <h2>一台可拆开的模型</h2><p class="muted small">改变结构，观察同一份记忆。</p>
    <label class="field-label">Attention 结构</label>
    <div class="segmented">${['mha', 'gqa', 'mqa'].map((v) => button('variant', v.toUpperCase(), `data-value="${v}" aria-pressed="${state.variant === v}" class="${state.variant === v ? 'active' : ''}"`)).join('')}</div>
    <p class="setting-hint">${state.variant === 'gqa' ? '每 2 个 Q head 共用一组 K/V。' : state.variant === 'mha' ? '每个 Q head 有独立的 K/V。' : '所有 Q heads 共用一组 K/V。'}</p>
    <label class="select-field">模型层数 <select data-config="layers" aria-label="模型层数">${[2, 3, 4].map((v) => option(v, `${v} layers`, state.layers)).join('')}</select></label>
    <label class="select-field">Q heads <select data-config="qHeads" aria-label="Q heads">${[4, 8].map((v) => option(v, v, state.qHeads)).join('')}</select></label>
    <dl class="dimensions"><div><dt>KV heads</dt><dd>${d.config.kvHeads}</dd></div><div><dt>每 head 维度</dt><dd>4</dd></div><div><dt>残差宽度</dt><dd>8</dd></div></dl>
    <div class="setting-divider"></div><div class="panel-label">PLACEMENT</div>
    <label class="select-field">张量并行 TP <select data-config="tp" aria-label="张量并行 TP">${[
      1, 2, 4, 8,
    ]
      .filter((v) => v <= state.qHeads)
      .map((v) => option(v, `${v} GPU`, state.tp))
      .join('')}</select></label>
    <label class="select-field">每台机器 <select data-config="gpusPerHost" aria-label="每台机器 GPU 数">${[1, 2, 4].map((v) => option(v, `${v} GPU`, state.gpusPerHost)).join('')}</select></label>
    <label class="select-field">容量精度 <select data-config="dtype" aria-label="容量估算精度">${option(2, 'BF16 · 2 B', state.dtype)}${option(1, 'FP8 · 1 B', state.dtype)}</select></label>
    <p class="setting-hint">精度仅用于容量估算；数值计算使用浏览器浮点。</p>
    <div class="model-note"><span class="live-dot"></span> 小模型，真实计算<p>确定性教学权重与预设续写。用于观察机制，不代表训练好的语言能力。</p></div>
  </aside>`;
  }
  function runtime(d) {
    const done = !state.active[state.request];
    const atEnd = d.count >= d.seqs[state.request].length;
    const phase = d.count < state.prompt.length ? 'PREFILL' : 'DECODE';
    return `<section id="runtime-controls" class="runtime" aria-label="执行控制" tabindex="-1">
    <div class="runtime-top"><div class="request-switch">${['A', 'B'].map((r) => button('request', `请求 ${r}`, `data-value="${r}" class="${state.request === r ? 'active' : ''}" aria-pressed="${state.request === r}"`)).join('')}</div>
      <span class="phase ${done ? 'retained' : ''}">${done ? 'ENDED · 保留候选' : phase}</span><span class="runtime-count">已处理 <b>${d.count}</b> / ${d.seqs[state.request].length} token</span>
      <div class="runtime-actions">${button('back', '←', `title="撤销上一步运行状态" aria-label="撤销上一步运行状态" ${!canUndo() ? 'disabled' : ''}`)}${button('prefill', '运行 prefill', `${done || d.count >= state.prompt.length ? 'disabled' : ''}`)}${button('step', `${icon('step')} 单步`, `${done || atEnd ? 'disabled' : ''}`)}${button('play', `${icon('play')} ${state.playing ? '暂停' : '播放'}`, `class="primary" ${done || atEnd ? 'disabled' : ''}`)}${button('reset', icon('reset'), 'title="重置实验" aria-label="重置实验"')}</div></div>
    <div class="token-stream" data-scroll-key="tokens" aria-label="选择查询 token">${d.seqs[state.request].map((token, t) => `${t === state.prompt.length ? '<span class="token-divider">预设续写 →</span>' : ''}${button('token', `<small>t${t}</small><span>${esc(token)}</span>`, `data-value="${t}" class="token ${t === state.query && t < d.count ? 'selected' : ''} ${t >= d.count ? 'future' : ''}" ${t >= d.count ? 'disabled' : ''} title="查看 t${t} 的计算快照" aria-pressed="${t === state.query && t < d.count}"`)}`).join('')}</div>
    <details class="prompt-editor" ${state.editorOpen ? 'open' : ''}><summary id="prompt-editor-toggle">编辑输入序列 <span>教学分词：空格分隔；无空格时按字符</span></summary><form id="prompt-form" novalidate><label class="sr-only" for="prompt-input">请求 A 输入序列</label><input id="prompt-input" name="prompt" type="text" aria-label="请求 A 输入序列" aria-describedby="prompt-limit prompt-error" aria-invalid="${Boolean(state.draftError)}" value="${esc(state.draft)}"><button id="apply-prompt" type="submit">应用并运行 prefill</button></form><p id="prompt-limit" class="prompt-limit">最多 16 个教学 token、160 个字符；超限时保留输入。<span id="prompt-count">${tokenize(state.draft).length} / 16 token</span></p><p id="prompt-error" class="prompt-error" role="alert" ${state.draftError ? '' : 'hidden'}>${esc(state.draftError)}</p></details>
  </section>`;
  }
  function miniCache(d, layer) {
    return `<div class="mini-cache"><div class="mini-cache-title"><span>KV CACHE</span><span>${d.count} × ${d.config.kvHeads} × 4 × 2</span></div>
    ${range(d.config.kvHeads)
      .map(
        (h) =>
          `<div class="mini-cache-row"><span>KV${h}</span><div class="mini-slots">${range(
            d.seqs[state.request].length,
          )
            .map((t) => {
              const written = t < d.count;
              const slot = d.pool.mappings[state.request][t];
              return button(
                'history',
                `<i class="k-fill"></i><i class="v-fill"></i>`,
                `data-token="${t}" data-layer="${layer}" data-kv="${h}" class="mini-slot ${written ? 'written' : ''} ${written && slot == null ? 'evicted' : ''} ${t === state.history && layer === state.layer && h === d.kv ? 'chosen' : ''} ${t > state.query ? 'masked' : ''}" title="Layer ${layer} · KV${h} · t${t}${slot != null ? ` · 槽位 ${slot}` : ' · 未驻留'}" aria-label="Layer ${layer} KV head ${h} token ${t}" ${!written ? 'disabled' : ''}`,
              );
            })
            .join('')}</div></div>`,
      )
      .join('')}<div class="cache-note">K 与 V 各一份 · 每层独立</div></div>`;
  }
  function layerCard(d, layer) {
    const trace = d.result.traces[state.query]?.layers[layer];
    const qn = d.config.qHeads,
      kvn = d.config.kvHeads;
    return `<article class="layer-card ${state.layer === layer ? 'selected-layer' : ''}" aria-label="Layer ${layer}">
    <div class="layer-heading">${button('layer', `<span class="layer-number">${String(layer + 1).padStart(2, '0')}</span><span>Layer ${layer}<small>独立权重 · 独立缓存</small></span>`, `data-value="${layer}" aria-pressed="${state.layer === layer}"`)}<span class="layer-dot"></span></div>
    <div class="residual-line"><span>h<sub>${layer}</sub></span>${vector(trace?.input)}<span>8维</span></div>
    <div class="norm-label">RMSNorm · Q/K/V 投影</div>
    <div class="head-arena"><div class="arena-label">QUERY HEADS <span>本步计算</span></div>
      <div class="q-row" data-head-count="${qn}" style="--heads:${qn}">${range(qn)
        .map((h) =>
          button(
            'head',
            `<span>Q${h}</span>${vector(trace?.q[h], 'q')}`,
            `data-layer="${layer}" data-head="${h}" class="q-head ${state.layer === layer && state.head === h ? 'selected-head' : ''}" title="选择 Layer ${layer} 的 Q head ${h}" aria-pressed="${state.layer === layer && state.head === h}"`,
          ),
        )
        .join('')}</div>
      <svg class="head-links" viewBox="0 0 240 32" preserveAspectRatio="none" aria-label="Q head 到 KV head 的共享关系">${range(
        qn,
      )
        .map((h) => {
          const x = ((h + 0.5) / qn) * 240,
            y = ((kvHeadFor(h, d.config) + 0.5) / kvn) * 240;
          return `<path d="M ${x} 0 C ${x} 15, ${y} 15, ${y} 31" class="${state.layer === layer && state.head === h ? 'active-link' : ''}"/>`;
        })
        .join('')}</svg>
      <div class="kv-row" data-head-count="${kvn}" style="--heads:${kvn}">${range(kvn)
        .map((h) =>
          button(
            'kv',
            `<span><b class="k-text">K${h}</b><b class="v-text">V${h}</b></span><small>${qn / kvn} Q → 1 KV</small>`,
            `data-layer="${layer}" data-kv="${h}" class="kv-head ${state.layer === layer && d.kv === h ? 'selected-kv' : ''}" aria-pressed="${state.layer === layer && d.kv === h}" aria-label="选择 Layer ${layer} KV head ${h}"`,
          ),
        )
        .join('')}</div>
      <div class="attention-equation">softmax(QKᵀ / √d) V <span>↓</span></div>
    </div>
    ${miniCache(d, layer)}
    <div class="ffn-block"><span>＋ 残差</span><b>Norm → SwiGLU FFN</b><span>＋ 残差</span></div>
    <div class="residual-line output"><span>h<sub>${layer + 1}</sub></span>${vector(trace?.output)}<span>8维</span></div>
  </article>`;
  }
  function modelView(d) {
    return `<div class="canvas-heading"><div><span class="eyebrow">COMPUTATION → MEMORY</span><h2>每一层，都留下自己的 K 和 V</h2><p>点选一个 head，沿着同一个 token 查看实际计算。</p></div><div class="zoom-controls">${button('zoom', '−', 'data-value="-1" aria-label="缩小模型"')}<span>${Math.round(state.zoom * 100)}%</span>${button('zoom', '＋', 'data-value="1" aria-label="放大模型"')}</div></div>
    <div class="model-scroll" data-scroll-key="model"><div class="model-chain" style="zoom:${state.zoom}">${range(
      d.config.layers,
    )
      .map(
        (layer) =>
          `${layer > 0 ? `<div class="layer-connector">${icon('arrow')}<small>h${layer}</small></div>` : ''}${layerCard(d, layer)}`,
      )
      .join('')}</div></div>
    <div class="figure-caption"><span><i class="legend-q"></i> Q：当前查询</span><span><i class="legend-k"></i> K：用于匹配</span><span><i class="legend-v"></i> V：被加权读取</span><span>虚线槽位：尚未写入</span></div>
    <div class="insight-strip"><span>此刻</span><p>${d.count ? `t${state.query} 的 Q${state.head} 在 Layer ${state.layer} 读取 <strong>KV${d.kv} 的 t0…t${state.query}</strong>。${d.count - 1 > state.query ? '更晚的位置被因果掩码排除。' : '历史 K/V 保持不变。'}` : '先运行 prefill，观察每层怎样写入缓存。'}</p></div>`;
  }
  function gpuCard(d, rank) {
    const owns = rank.kvHeads.includes(d.kv);
    const bytes = 2 * d.config.layers * rank.kvHeads.length * d.config.headDim * state.dtype;
    return `<article class="gpu-card ${owns ? 'owns-selection' : ''}"><header><div><span class="gpu-symbol">▦</span><b>GPU ${rank.gpu}</b><span class="rank-label">rank ${rank.rank}</span></div><small>${formatBytes(d.pool.reservedSlots * bytes)} 页预留</small></header>
    <div class="gpu-heads"><span>Q</span>${rank.qHeads.map((h) => button('device-head', `${h}`, `data-value="${h}" class="${state.head === h ? 'active' : ''}" aria-label="选择 rank ${rank.rank} Q head ${h}"`)).join('')}<span class="kv-owner">KV ${rank.kvHeads.join(', ')}${rank.replicas.length ? ' · 含副本' : ''}</span></div>
    <div class="gpu-layers" data-scroll-key="gpu-${rank.rank}">${range(d.config.layers)
      .flatMap((l) =>
        rank.kvHeads.map(
          (kv) =>
            `<div class="gpu-layer ${state.layer === l && d.kv === kv ? 'selected' : ''}">${button('gpu-kv', `L${l} · KV${kv}`, `data-layer="${l}" data-kv="${kv}" data-rank="${rank.rank}" aria-label="选择 rank ${rank.rank} Layer ${l} KV head ${kv}"`)}<div class="gpu-slots">${range(
              d.pool.capacity,
            )
              .map((slot) => {
                const page = d.pool.pages.find((p) => p.id === Math.floor(slot / state.pageSize));
                const offset = slot % state.pageSize;
                const occupied = page && offset < page.tokens.length;
                const selected =
                  d.pool.mappings[state.request][state.history] === slot &&
                  d.kv === kv &&
                  state.layer === l;
                return `<span class="gpu-slot ${occupied ? (page.refs ? 'occupied' : 'retained') : ''} ${selected ? 'selected' : ''}" title="L${l} · KV${kv} · slot ${slot}${occupied ? ` · ${esc(page.tokens[offset])}` : ''}">${occupied ? '' : ''}</span>`;
              })
              .join('')}</div></div>`,
        ),
      )
      .join(
        '',
      )}</div><footer>每 token / 本卡 <strong>${formatBytes(bytes)}</strong>${rank.replicas.length ? '<span class="replica-tag">KV 复制</span>' : ''}</footer></article>`;
  }
  function devicesView(d) {
    const hosts = [...new Set(d.ranks.map((r) => r.host))];
    const partials = d.result.traces[state.query]
      ? tpPartialOutputs(
          d.result.traces[state.query].layers[state.layer],
          d.result.model.layers[state.layer],
          d.config,
          state.tp,
        )
      : [];
    return `<div class="canvas-heading"><div><span class="eyebrow">MODEL → GPU → HOST</span><h2>把一份模型，分布到 ${hosts.length} 台机器</h2><p>同一份逻辑序列，各 rank 持有自己需要的 head 与缓存。</p></div><span class="outline-tag">TP ${state.tp} · 每机 ${state.gpusPerHost} GPU</span></div>
    <div class="host-grid">${hosts
      .map(
        (host) =>
          `<section class="host"><header><span class="host-indicator"></span><b>机器 ${String(host + 1).padStart(2, '0')}</b><span>HOST ${host}</span></header><div class="host-gpus">${d.ranks
            .filter((r) => r.host === host)
            .map((rank) => gpuCard(d, rank))
            .join(
              '',
            )}</div>${d.ranks.filter((r) => r.host === host).length > 1 ? '<div class="interconnect">↔ 机内互连 · 输出部分和归并</div>' : ''}</section>`,
      )
      .join('')}</div>
    <div class="network-link"><span></span><b>↔ ${hosts.length > 1 ? '跨机网络' : '本机通信'} · All-reduce</b><span></span></div>
    <div class="reduction"><div><span class="eyebrow">THE SAME RESULT</span><h3>切分计算，归并成同一个输出</h3><p>Layer ${state.layer} · t${state.query} 的 attention 输出投影</p></div><div class="partial-rows">${partials.map((values, r) => `<div><span>rank ${r}</span>${vector(values, '', true)}</div>`).join('')}<div class="sum-row"><span>Σ 合计</span>${vector(d.result.traces[state.query]?.layers[state.layer].attentionOutput, '', true)}</div></div></div>
    <p class="technical-note">彩色外框表示持有所选 KV${d.kv} 的 GPU。TP 大于 KV head 数时会出现副本；每卡的逻辑 token 数不能相加。连接仅表达部署关系，不预测通信时间。</p>`;
  }
  function requestCard(d, request) {
    const tokens = d.seqs[request].slice(0, state.prompt.length);
    const active = state.active[request];
    return `<article class="request-card ${state.request === request ? 'current' : ''}"><header>${button('request', `<b>${request}</b> 请求 ${request}`, `data-value="${request}"`)}<span class="status-pill ${active ? '' : 'retained'}">${state.cursor[request] === 0 ? '等待输入' : active ? '运行中' : '结束 · 缓存保留候选'}</span></header><div class="request-tokens">${tokens.map((t, i) => `<span class="${d.pool.mappings.A[i] != null && d.pool.mappings.A[i] === d.pool.mappings.B[i] ? 'shared-token' : ''} ${i >= state.cursor[request] ? 'pending-token' : ''}"><small>t${i}</small>${esc(t)}</span>`).join('')}</div><footer><span>${request === 'B' && !state.sameIdentity ? 'model / adapter B' : 'model / adapter A'}</span><div>${button('prefill-request', '运行 prefill', `data-value="${request}" ${!active || state.cursor[request] >= state.prompt.length ? 'disabled' : ''}`)}${button('finish-request', '结束请求', `data-value="${request}" ${!active || !state.cursor[request] ? 'disabled' : ''}`)}</div></footer></article>`;
  }
  function cacheView(d) {
    const map = new Map(d.pool.pages.map((p) => [p.id, p]));
    let compare = '';
    const t = d.seqs.A.findIndex(
      (token, i) =>
        i >= 1 && token === d.seqs.B[i] && d.results.A.traces[i] && d.results.B.traces[i],
    );
    if (t >= 0)
      compare = `<div class="context-comparison"><div><span class="eyebrow">SAME TOKEN, DIFFERENT CONTEXT?</span><h3>同一个「${esc(d.seqs.A[t])}」，逐层比较 K</h3><p>相同位置 t${t} · KV head ${d.kv} · 最大数值差</p></div><div class="difference-layers">${range(
        d.config.layers,
      )
        .map((l) => {
          const diff = maxDifference(
            d.results.A.caches[l].keys[t][d.kv],
            d.results.B.caches[l].keys[t][d.kv],
          );
          return `<div><span>Layer ${l}</span><b class="${diff > 1e-6 ? 'changed' : ''}">${diff < 1e-6 ? '0.000' : n(diff)}</b><small>${diff < 1e-6 ? '当前条件下相同' : '上下文已改变表示'}</small></div>`;
        })
        .join('')}</div></div>`;
    return `<div class="canvas-heading"><div><span class="eyebrow">REQUESTS → SHARED PAGES</span><h2>共享的是计算过的前缀</h2><p>结束请求，然后淘汰空闲页，观察引用与槽位。</p></div></div>
    <div class="scenario-controls"><div class="segmented">${button('scenario', '相同前缀', 'data-value="prefix" aria-pressed="' + (state.scenario === 'prefix') + '" class="' + (state.scenario === 'prefix' ? 'active' : '') + '"')}${button('scenario', '相同后缀', 'data-value="suffix" aria-pressed="' + (state.scenario === 'suffix') + '" class="' + (state.scenario === 'suffix' ? 'active' : '') + '"')}</div><label><input type="checkbox" data-config="sameIdentity" ${state.sameIdentity ? 'checked' : ''}> 相同模型 / 适配器</label><label>页大小 <select data-config="pageSize" aria-label="页大小">${[2, 4].map((v) => option(v, `${v} tokens`, state.pageSize)).join('')}</select></label></div>
    <div class="requests-grid">${requestCard(d, 'A')}${requestCard(d, 'B')}</div>
    <div class="pool-summary"><div><b>${d.pool.sharedTokens}</b><span>token 共用完整页</span></div><div><b>${d.pool.pages.filter((p) => p.refs === 0).length}</b><span>页可淘汰</span></div><div><b>${d.pool.reservedSlots}</b><span>物理槽位已预留</span></div>${button('evict', '淘汰无引用页', `class="outline-button" ${!d.pool.pages.some((p) => p.refs === 0) ? 'disabled' : ''}`)}</div>
    <div class="pool-title"><h3>物理页池</h3><span><i class="status-dot shared"></i> 多请求共享 <i class="status-dot"></i> 活动引用 <i class="status-dot retained"></i> 保留 <i class="status-dot free"></i> 空闲</span></div>
    <div class="page-pool">${range(d.pool.capacity / state.pageSize)
      .map((id) => {
        const p = map.get(id);
        const kind = !p
          ? 'free'
          : p.refs === 0
            ? 'retained'
            : p.owners.length > 1
              ? 'shared'
              : 'live';
        return `<article class="memory-page ${kind}"><header><span>PAGE ${id}</span><small>${p ? `refs ${p.refs}` : 'FREE'}</small></header><div class="page-slots">${range(
          state.pageSize,
        )
          .map((offset) => {
            const slot = id * state.pageSize + offset;
            const position = d.pool.mappings[state.request].indexOf(slot);
            return button(
              'slot',
              `<small>${slot}</small><span>${p?.tokens[offset] ? esc(p.tokens[offset]) : '·'}</span>`,
              `data-slot="${slot}" class="${position === state.history && position >= 0 ? 'selected' : ''}" ${!p?.tokens[offset] ? 'disabled' : ''} title="物理槽位 ${slot}，各 layer 分别保存 K/V"`,
            );
          })
          .join(
            '',
          )}</div><footer>${p ? p.owners.map((o) => `<b class="owner">${o}</b>`).join('') : '未分配'}<span>${p ? (p.refs ? '读取中' : '保留，可回收') : '可供分配'}</span></footer></article>`;
      })
      .join('')}</div>
    <div class="mapping-strip"><b>请求 ${state.request} → 槽位</b><code>${d.pool.mappings[state.request].map((slot, t) => `t${t}→${slot == null ? '已回收' : slot}`).join('　') || '运行 prefill 后建立映射'}</code></div>
    <p class="technical-note">在页边界匹配已知完整页；已分配的部分页补齐后保持地址，不再事后合并。槽位号跨层对应，但每层保存自己的向量。当前图为可复现教学分配器。</p>${compare}`;
  }
  function inspector(d) {
    const trace = d.result.traces[state.query]?.layers[state.layer];
    if (!trace)
      return `<aside class="inspector"><span class="eyebrow">LIVE INSPECTOR</span><h2>等待第一份记忆</h2><p>运行 prefill，或单步处理一个 token。计算的数值会出现在这里。</p></aside>`;
    const head = trace.heads[state.head];
    const keys = d.result.caches[state.layer].keys;
    const values = d.result.caches[state.layer].values;
    const slot = d.pool.mappings[state.request][state.history];
    return `<aside class="inspector" aria-label="选中计算的实际数值"><div class="inspector-title"><span class="eyebrow">LIVE INSPECTOR</span><span class="live-dot"></span></div>
    <h2>这一刻，读到了什么</h2><div class="selection-path"><b>${state.request}</b><span>Layer ${state.layer}</span><span>Q${state.head} → KV${d.kv}</span></div>
    <div class="vector-section"><div><b class="q-text">Query</b><span>t${state.query}「${esc(d.seqs[state.request][state.query])}」</span></div>${vector(head.q, 'q', true)}<small>当前计算快照 · 不写入 KV 池</small></div>
    <div class="vector-section"><div><b class="k-text">Key</b><span>t${state.history}「${esc(d.seqs[state.request][state.history])}」</span></div>${vector(keys[state.history]?.[d.kv], 'k', true)}</div>
    <div class="vector-section"><div><b class="v-text">Value</b><span>slot ${slot == null ? '已回收' : slot}</span></div>${vector(values[state.history]?.[d.kv], 'v', true)}</div>
    <div class="attention-table-title"><b>Q${state.head} 正在关注</b><span>Q·K / √4 → softmax</span></div><div class="attention-rows">${head.weights.map((weight, t) => button('attend', `<span class="attn-token"><small>t${t}</small>${esc(d.seqs[state.request][t])}</span><span class="score">${n(head.scores[t])}</span><span class="weight-bar"><i style="width:${weight * 100}%"></i></span><b>${(weight * 100).toFixed(1)}%</b>`, `data-value="${t}" class="${state.history === t ? 'selected' : ''}" aria-pressed="${state.history === t}" aria-label="读取 token ${t}，权重 ${(weight * 100).toFixed(1)}%"`)).join('')}</div>
    <div class="vector-section result-vector"><div><b>Σ 权重 × V</b><span>head 输出</span></div>${vector(head.context, '', true)}</div>
    <div class="address"><span>缓存地址</span><code>Layer ${state.layer} / KV${d.kv}<br>slot ${slot == null ? '未驻留' : slot} / K、V</code><small>${slot == null ? '已回收；下方显示历史计算快照。原归属 rank' : '位于 rank'} ${d.ranks
      .filter((r) => r.kvHeads.includes(d.kv))
      .map((r) => r.rank)
      .join(
        '、',
      )}${d.ranks.filter((r) => r.kvHeads.includes(d.kv)).length > 1 ? '（含副本）' : ''}</small></div>
  </aside>`;
  }
  function capacity(d) {
    const total = bytesPerToken(QWEN3_32B, state.dtype) * state.context;
    const rank = rankLayout(QWEN3_32B, state.tp)[0];
    const perRank = 2 * 64 * rank.kvHeads.length * 128 * state.dtype * state.context;
    return `<section class="capacity"><div class="capacity-intro"><span class="eyebrow">FROM A SMALL MODEL TO A REAL ONE</span><h2>把尺度换成 Qwen3-32B</h2><p>64 layers · 64 Q heads · 8 KV heads · head_dim 128</p><a href="${QWEN3_32B_SOURCE}" target="_blank" rel="noreferrer">公开配置 ↗</a></div><div class="capacity-control"><label for="context-length">上下文 <b>${state.context.toLocaleString()} token</b></label><input id="context-length" type="range" min="1024" max="32768" step="1024" value="${state.context}" data-config="context"><code>2 × 64 × 8 × 128 × ${state.dtype} B</code><p>每 token ${formatBytes(bytesPerToken(QWEN3_32B, state.dtype))}</p></div><div class="capacity-values"><div><span>全模型逻辑 KV</span><strong>${formatBytes(total)}</strong></div><div><span>每张卡 · TP=${state.tp}</span><strong>${formatBytes(perRank)}</strong></div></div><p class="capacity-note">独立的真实模型容量映射，非上方小模型数值。未计页对齐、量化 scale、元数据与其他显存开销。</p></section>`;
  }

  return {
    sidebar,
    runtime,
    modelView,
    devicesView,
    cacheView,
    inspector,
    capacity,
    icon,
    button,
    esc,
  };
}
