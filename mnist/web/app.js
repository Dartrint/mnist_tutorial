/* MNIST digit recognizer - front-end logic (vanilla JS, fully offline) */
'use strict';
(function () {
  const CANVAS_SIZE = 280; // drawing canvas resolution (CSS px + internal px)
  const LINE_WIDTH = 18;   // stroke width in canvas pixels
  const DEBOUNCE_MS = 400; // auto-predict delay after the user stops drawing
  // The single shared application state object.
  const state = {
    canvas: null, ctx: null, modelSelect: null,
    drawing: false, hasInk: false, lastX: 0, lastY: 0, debounceTimer: null,
    predictSeq: 0, samplesSeq: 0, evalSeq: 0, // request counters -> drop stale responses
    pendingSource: null,                      // {label, el} when prediction came from a sample
  };

  /* ------------------------------ helpers ------------------------------ */
  const $ = (id) => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  const formatPercent = (value) =>
    (typeof value === 'number' && isFinite(value) ? (value * 100).toFixed(2) + '%' : '–');
  const kindLabel = (kind) => {
    const k = String(kind || '').toLowerCase();
    if (k === 'mlp') return 'MLP';
    if (k === 'cnn') return 'CNN';
    return k ? k.toUpperCase() : 'không rõ';
  };
  const selectedModel = () => (state.modelSelect && state.modelSelect.value) || undefined;
  function setStatus(message, kind) {
    const node = $('status-line');
    node.textContent = message || '';
    node.className = 'status-line' + (kind && kind !== 'idle' ? ' is-' + kind : '');
  }
  async function apiFetch(url, options) {
    const response = await fetch(url, options);
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) throw new Error((payload && payload.error) || 'Lỗi HTTP ' + response.status);
    return payload;
  }
  function postJSON(url, body) {
    const headers = { 'Content-Type': 'application/json' };
    return apiFetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  }

  /* ------------------------------- canvas ------------------------------ */
  function setupCanvas() {
    const ctx = state.ctx;
    Object.assign(ctx, {
      fillStyle: '#000', strokeStyle: '#fff', lineWidth: LINE_WIDTH,
      lineCap: 'round', lineJoin: 'round',
    });
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }
  function canvasPos(evt) {
    const rect = state.canvas.getBoundingClientRect();
    const point = evt.touches && evt.touches.length ? evt.touches[0] : evt;
    const x = (point.clientX - rect.left) * (CANVAS_SIZE / rect.width);
    const y = (point.clientY - rect.top) * (CANVAS_SIZE / rect.height);
    return { x, y };
  }
  function strokeSegment(x0, y0, x1, y1) {
    const ctx = state.ctx;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  function startStroke(evt) {
    evt.preventDefault();
    const pos = canvasPos(evt);
    Object.assign(state, { drawing: true, hasInk: true, pendingSource: null, lastX: pos.x, lastY: pos.y });
    strokeSegment(pos.x, pos.y, pos.x, pos.y); // single dot
  }
  function moveStroke(evt) {
    if (!state.drawing) return;
    evt.preventDefault();
    const pos = canvasPos(evt);
    strokeSegment(state.lastX, state.lastY, pos.x, pos.y);
    Object.assign(state, { lastX: pos.x, lastY: pos.y });
    schedulePredict();
  }
  function endStroke(evt) {
    if (!state.drawing) return;
    if (evt && evt.preventDefault) evt.preventDefault();
    state.drawing = false;
    schedulePredict();
  }
  function bindCanvasEvents() {
    const canvas = state.canvas;
    ['mousedown', 'touchstart'].forEach((t) => canvas.addEventListener(t, startStroke, { passive: false }));
    ['mousemove', 'touchmove'].forEach((t) => canvas.addEventListener(t, moveStroke, { passive: false }));
    ['mouseup', 'mouseleave', 'touchend', 'touchcancel'].forEach((t) =>
      canvas.addEventListener(t, endStroke, { passive: false }));
  }
  function clearCanvas(keepStatus) {
    if (state.debounceTimer) { clearTimeout(state.debounceTimer); state.debounceTimer = null; }
    setupCanvas();
    state.hasInk = false;
    state.drawing = false;
    state.pendingSource = null;
    resetPredictionPanels();
    if (!keepStatus) setStatus('', 'idle');
  }

  /* --------------------------- auto-prediction -------------------------- */
  function schedulePredict() {
    if (state.debounceTimer) clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => { state.debounceTimer = null; predictNow(null); }, DEBOUNCE_MS);
  }
  async function predictNow(source) {
    if (!state.hasInk) { setStatus('Hãy vẽ một chữ số trước khi dự đoán.', 'hint'); return; }
    const seq = ++state.predictSeq; // a newer request invalidates older ones
    state.pendingSource = source || null;
    setStatus('Đang dự đoán…', 'loading');
    try {
      const data = await postJSON('/api/predict', {
        model: selectedModel(),
        image: state.canvas.toDataURL('image/png'),
        center: true,
      });
      if (seq !== state.predictSeq) return; // stale response -> ignore it
      renderPrediction(data);
      applySampleBadge(state.pendingSource, data.digit);
      setStatus('', 'idle');
    } catch (err) {
      if (seq !== state.predictSeq) return;
      setStatus(err.message, 'error');
    }
  }
  function resetPredictionPanels() {
    $('digit-badge').textContent = '–';
    $('confidence-value').textContent = '–';
    $('result-model').textContent = '–';
    $('result-elapsed').textContent = '–';
    buildProbRows(null, null);
    const preview = $('preview-img');
    preview.removeAttribute('src');
    preview.classList.remove('is-visible');
    $('preview-hint').textContent = 'Chưa có dữ liệu. Hãy vẽ hoặc chọn một ảnh test.';
  }
  function renderPrediction(data) {
    $('digit-badge').textContent = String(data.digit);
    const elapsed = typeof data.elapsed_ms === 'number' ? data.elapsed_ms.toFixed(1) + ' ms' : '–';
    $('confidence-value').textContent = formatPercent(data.confidence);
    $('result-model').textContent = data.model || '–';
    $('result-elapsed').textContent = elapsed;
    buildProbRows(Array.isArray(data.probs) ? data.probs : null, data.digit);
    if (!data.preview) return;
    const preview = $('preview-img');
    preview.src = data.preview;
    preview.classList.add('is-visible');
    $('preview-hint').textContent = 'Ảnh 28×28 (phóng to 10×) mà model thực sự nhận được.';
  }
  function buildProbRows(probs, winner) {
    const chart = $('prob-chart');
    chart.textContent = '';
    for (let digit = 0; digit < 10; digit += 1) {
      const value = probs ? Math.max(0, Math.min(1, Number(probs[digit]) || 0)) : 0;
      const row = el('div', 'prob-row' + (probs && digit === winner ? ' is-winner' : ''));
      const track = el('div', 'prob-track');
      const fill = el('div', 'prob-fill');
      fill.style.width = (value * 100).toFixed(1) + '%';
      track.appendChild(fill);
      row.append(el('span', 'prob-label', String(digit)), track,
        el('span', 'prob-value', probs ? formatPercent(value) : '–'));
      chart.appendChild(row);
    }
  }
  function applySampleBadge(source, digit) {
    if (!source || !source.el) return;
    const node = source.el;
    const old = node.querySelector('.badge-wrong');
    if (old) old.remove();
    const wrong = Number(digit) !== Number(source.label);
    node.classList.toggle('is-wrong', wrong);
    if (wrong) node.appendChild(el('span', 'badge-wrong', 'Dự đoán ' + digit));
  }

  /* ------------------------------- models ------------------------------- */
  async function loadModels() {
    setStatus('Đang tải danh sách model…', 'loading');
    try {
      const data = await apiFetch('/api/models');
      const models = Array.isArray(data && data.models) ? data.models : [];
      const select = state.modelSelect;
      select.textContent = '';
      if (!models.length) {
        select.disabled = true;
        $('model-meta').textContent = 'Không có model khả dụng.';
        setStatus('Chưa có model nào trong models/. Hãy chạy: python -m mnist train', 'hint');
        return;
      }
      models.forEach((model) => {
        const accuracy = typeof model.test_accuracy === 'number'
          ? formatPercent(model.test_accuracy) + ' độ chính xác test'
          : 'chưa rõ độ chính xác';
        const option = el('option', '', model.name + ' · ' + kindLabel(model.kind) + ' · ' + accuracy);
        option.value = model.name;
        select.appendChild(option);
      });
      const fallback = models.find((m) => m.default) || models[0];
      select.value = (data && data.default) || fallback.name;
      select.disabled = false;
      $('model-meta').textContent = models.length + ' model khả dụng.';
      setStatus('Sẵn sàng. Hãy vẽ một chữ số.', 'idle');
    } catch (err) {
      state.modelSelect.disabled = true;
      setStatus('Không tải được danh sách model: ' + err.message, 'error');
    }
  }

  /* ------------------------------- samples ------------------------------ */
  async function loadSamples() {
    const seq = ++state.samplesSeq;
    setStatus('Đang lấy ảnh test…', 'loading');
    try {
      const data = await apiFetch('/api/samples?n=8');
      if (seq !== state.samplesSeq) return;
      renderSamples(Array.isArray(data && data.samples) ? data.samples : []);
      setStatus('', 'idle');
    } catch (err) {
      if (seq !== state.samplesSeq) return;
      setStatus('Không lấy được ảnh test: ' + err.message, 'error');
    }
  }
  function renderSamples(samples) {
    const grid = $('samples-grid');
    if (!samples.length) {
      grid.innerHTML = '<p class="placeholder">Không có ảnh test nào.</p>';
      return;
    }
    grid.innerHTML = samples.map((sample) =>
      '<button type="button" class="sample" aria-label="Ảnh test nhãn ' + sample.label + ', bấm để dự đoán">' +
      '<img src="' + sample.image + '" alt="Ảnh chữ số ' + sample.label + '">' +
      '<span class="sample-label">Nhãn: ' + sample.label + '</span></button>').join('');
    Array.from(grid.children).forEach((button, index) => {
      button.addEventListener('click', () => loadSampleIntoCanvas(samples[index], button));
    });
  }
  function loadSampleIntoCanvas(sample, button) {
    const img = new Image();
    img.onload = () => {
      clearCanvas(true);
      state.ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      state.hasInk = true;
      predictNow({ label: sample.label, el: button });
    };
    img.onerror = () => setStatus('Không tải được ảnh test.', 'error');
    img.src = sample.image;
  }

  /* ----------------------------- evaluation ----------------------------- */
  async function runEvaluation() {
    const seq = ++state.evalSeq;
    const button = $('btn-evaluate');
    button.disabled = true;
    setStatus('Đang đánh giá model (có thể mất vài giây)…', 'loading');
    try {
      const data = await postJSON('/api/evaluate', { model: selectedModel(), limit: 1000 });
      if (seq !== state.evalSeq) return;
      renderEvaluation(data);
      setStatus('', 'idle');
    } catch (err) {
      if (seq !== state.evalSeq) return;
      setStatus('Đánh giá thất bại: ' + err.message, 'error');
    } finally {
      if (seq === state.evalSeq) button.disabled = false;
    }
  }
  function statBlock(label, value) {
    const wrap = el('div', 'eval-stat');
    wrap.append(el('span', 'stat-label', label), el('span', 'stat-value', value));
    return wrap;
  }
  function renderEvaluation(data) {
    const summary = $('eval-summary');
    summary.textContent = '';
    summary.append(
      statBlock('Model', data.model || '–'),
      statBlock('Độ chính xác', formatPercent(data.accuracy)),
      statBlock('Số ảnh (n)', String(data.n)),
      statBlock('Loss', typeof data.loss === 'number' ? data.loss.toFixed(4) : '–')
    );
    renderPerClass(data.per_class_accuracy || {});
    renderConfusion(data.confusion_matrix || []);
  }
  function renderPerClass(perClass) {
    const host = $('per-class');
    host.textContent = '';
    for (let digit = 0; digit < 10; digit += 1) {
      const value = Math.max(0, Math.min(1, Number(perClass[String(digit)]) || 0));
      const row = el('div', 'pc-row');
      const track = el('div', 'pc-track');
      const fill = el('div', 'pc-fill');
      fill.style.width = (value * 100).toFixed(1) + '%';
      track.appendChild(fill);
      row.append(el('span', 'prob-label', String(digit)), track,
        el('span', 'prob-value', formatPercent(value)));
      host.appendChild(row);
    }
  }
  function renderConfusion(matrix) {
    const host = $('confusion');
    if (!matrix.length) {
      host.innerHTML = '<p class="placeholder">Không có dữ liệu ma trận.</p>';
      return;
    }
    let max = 1;
    matrix.forEach((row) => row.forEach((count) => { max = Math.max(max, count); }));
    const grid = el('div', 'cm-grid');
    grid.style.gridTemplateColumns = 'repeat(11, minmax(44px, 1fr))';
    grid.appendChild(el('div', 'cm-cell cm-head', ''));
    for (let column = 0; column < 10; column += 1) {
      grid.appendChild(el('div', 'cm-cell cm-head', String(column)));
    }
    matrix.forEach((row, trueLabel) => {
      grid.appendChild(el('div', 'cm-cell cm-rowhead', String(trueLabel)));
      row.forEach((count, predicted) => {
        const alpha = (0.06 + 0.84 * (count / max)).toFixed(3);
        const rgb = trueLabel === predicted ? '63, 185, 80' : '76, 154, 255';
        const cell = el('div', 'cm-cell', String(count));
        cell.style.background = 'rgba(' + rgb + ', ' + alpha + ')';
        grid.appendChild(cell);
      });
    });
    host.textContent = '';
    host.appendChild(grid);
  }

  /* -------------------------------- init -------------------------------- */
  function init() {
    state.canvas = $('draw-canvas');
    state.ctx = state.canvas.getContext('2d');
    state.modelSelect = $('model-select');
    setupCanvas();
    bindCanvasEvents();
    $('btn-clear').addEventListener('click', () => clearCanvas(false));
    $('btn-predict').addEventListener('click', () => predictNow(null));
    $('btn-samples').addEventListener('click', loadSamples);
    $('btn-evaluate').addEventListener('click', runEvaluation);
    state.modelSelect.addEventListener('change', () => { if (state.hasInk) predictNow(null); });
    buildProbRows(null, null);
    loadModels();
  }
  document.addEventListener('DOMContentLoaded', init);
})();
