'use strict';

const result = window.MESH_RESULTS;
const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat('zh-CN');
const show = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const count = (value) => Number.isFinite(value) ? number.format(value) : '—';
const rate = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const time = (value) => Number.isFinite(value) ? `${value.toFixed(value >= 10 ? 1 : 2)} ms` : '—';
const signed = (value) => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(3)}` : '—';
const labels = Object.fromEntries((result?.methods || []).map(({id, label}) => [id, label]));
const parameterLabels = {
  wall_spacing: '首层间距', normal_blend: '近壁法向修正', tangential_bias: '周向加密偏置',
  upper_junction: '上表面块接口', lower_junction: '下表面块接口'
};
const state = {caseIndex: 0, liveCase: null, method: 'sac', step: 0, near: true, heat: false, timer: null};
const methodButtons = new Map();
const colors = [
  [48, 34, 85], [52, 84, 155], [28, 143, 186], [88, 211, 192], [241, 234, 155]
];
const palette = Array.from({length: 161}, (_, index) => {
  const location = index / 40;
  const from = Math.min(3, Math.floor(location));
  const ratio = location - from;
  const rgb = colors[from].map((value, channel) => Math.round(value + (colors[from + 1][channel] - value) * ratio));
  return `rgb(${rgb.join(',')})`;
});

function selectedCase() {
  return state.caseIndex === -1 ? state.liveCase : result.cases[state.caseIndex];
}

function caseName(test) {
  return /^\d{4}$/.test(test.name) ? `NACA ${test.name}` : test.name;
}

function current() {
  const test = selectedCase();
  const method = test.methods[state.method];
  const baseline = test.methods.fixed.frames[0];
  return {test, method, frames: method.frames, frame: method.frames[state.step], baseline};
}

function canvasContext(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width, height};
}

function meshBounds(first, second, near) {
  const meshes = [first.mesh, second.mesh];
  const rows = near ? meshes.map((mesh) => mesh[0]) : meshes.flat();
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const row of rows) {
    for (const [x, y] of row) {
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }
  const chord = xMax - xMin;
  const paddingX = chord * (near ? .22 : .04);
  const paddingY = near ? Math.max(.14 * chord, (.67 * chord - (yMax - yMin)) / 2) : (yMax - yMin) * .04;
  return [xMin - paddingX, xMax + paddingX, yMin - paddingY, yMax + paddingY];
}

function drawMesh(canvas, frame, bounds, description) {
  const {ctx, width, height} = canvasContext(canvas);
  if (!width || !height) return;
  const [xMin, xMax, yMin, yMax] = bounds;
  const sx = (width - 24) / (xMax - xMin);
  const sy = (height - 28) / (yMax - yMin);
  const scale = state.near ? Math.max(sx, sy) : Math.min(sx, sy);
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;
  const xy = frame.mesh.map((row) => row.map(([x, y]) => [width / 2 + (x - centerX) * scale, height / 2 - (y - centerY) * scale]));
  const columns = xy[0].length - 1;
  ctx.fillStyle = '#14283e';
  ctx.fillRect(0, 0, width, height);
  drawPlotGrid(ctx, width, height);
  if (state.heat) drawCells(ctx, xy, frame.cell_quality, width, height);
  drawGridLines(ctx, xy, columns);
  drawAirfoil(ctx, xy[0]);
  drawScale(ctx, width, height, scale);
  canvas.setAttribute('aria-label', description);
}

function drawPlotGrid(ctx, width, height) {
  ctx.beginPath();
  for (let x = width / 2 % 29; x < width; x += 29) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = height / 2 % 29; y < height; y += 29) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.strokeStyle = '#223b51';
  ctx.lineWidth = .6;
  ctx.stroke();
}

function drawCells(ctx, xy, values, width, height) {
  for (let i = 0; i < xy.length - 1; i += 1) {
    for (let j = 0; j < xy[i].length - 1; j += 1) {
      const a = xy[i][j];
      const b = xy[i][j + 1];
      const c = xy[i + 1][j + 1];
      const d = xy[i + 1][j];
      if (Math.max(a[0], b[0], c[0], d[0]) < 0 || Math.min(a[0], b[0], c[0], d[0]) > width ||
          Math.max(a[1], b[1], c[1], d[1]) < 0 || Math.min(a[1], b[1], c[1], d[1]) > height) continue;
      const value = values?.[i]?.[j];
      if (!Number.isFinite(value)) continue;
      ctx.beginPath();
      ctx.moveTo(...a);
      ctx.lineTo(...b);
      ctx.lineTo(...c);
      ctx.lineTo(...d);
      ctx.closePath();
      ctx.fillStyle = value <= 0 ? '#ef547b' : palette[Math.round(Math.min(1, value) * 160)];
      ctx.fill();
    }
  }
}

function drawGridLines(ctx, xy, columns) {
  ctx.beginPath();
  for (const row of xy) {
    ctx.moveTo(...row[0]);
    for (let j = 1; j <= columns; j += 1) ctx.lineTo(...row[j]);
  }
  for (let j = 0; j < columns; j += 1) {
    ctx.moveTo(...xy[0][j]);
    for (let i = 1; i < xy.length; i += 1) ctx.lineTo(...xy[i][j]);
  }
  ctx.strokeStyle = state.heat ? '#10273966' : '#72cfcd8c';
  ctx.lineWidth = state.heat ? .55 : .66;
  ctx.stroke();
  ctx.beginPath();
  for (const block of [0, 1, 2, 3]) {
    const index = Math.round(block * columns / 4);
    ctx.moveTo(...xy[0][index]);
    for (let i = 1; i < xy.length; i += 1) ctx.lineTo(...xy[i][index]);
  }
  ctx.strokeStyle = state.heat ? '#233b6a' : '#a6bbf6';
  ctx.lineWidth = 1.35;
  ctx.stroke();
}

function drawAirfoil(ctx, contour) {
  ctx.beginPath();
  ctx.moveTo(...contour[0]);
  for (const point of contour.slice(1)) ctx.lineTo(...point);
  ctx.closePath();
  ctx.fillStyle = '#f4f9fc';
  ctx.fill();
  ctx.strokeStyle = '#e5f5ef';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawScale(ctx, width, height, scale) {
  const maxUnits = (width * .16) / scale;
  const magnitude = 10 ** Math.floor(Math.log10(maxUnits));
  const units = Math.max(1, Math.floor(maxUnits / magnitude)) * magnitude;
  const length = units * scale;
  const start = width - length - 14;
  const y = height - 14;
  ctx.fillStyle = '#14283ee0';
  ctx.fillRect(start - 8, y - 20, length + 13, 31);
  ctx.strokeStyle = '#e2eff8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(start, y - 3);
  ctx.lineTo(start, y + 3);
  ctx.moveTo(start, y);
  ctx.lineTo(start + length, y);
  ctx.moveTo(start + length, y - 3);
  ctx.lineTo(start + length, y + 3);
  ctx.stroke();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#e2eff8';
  ctx.font = '11px Consolas, monospace';
  ctx.fillText(`${Number(units.toPrecision(2))} c`, start + length / 2, y - 6);
}

function refreshMeshes() {
  const {test, frame, baseline} = current();
  const bounds = meshBounds(baseline, state.near ? baseline : frame, state.near);
  const view = state.heat ? '单元质量图' : '网格线图';
  drawMesh(byId('fixed-canvas'), baseline, bounds, `${caseName(test)} 固定参数${view}`);
  drawMesh(byId('active-canvas'), frame, bounds, `${caseName(test)} ${labels[state.method]}第 ${frame.step} 步${view}`);
}

function metric(label, value, detail, alert) {
  const item = document.createElement('div');
  item.className = 'metric';
  const title = document.createElement('span');
  const prominent = document.createElement('strong');
  const hint = document.createElement('small');
  title.textContent = label;
  prominent.textContent = value;
  prominent.classList.toggle('alert', Boolean(alert));
  hint.textContent = detail;
  item.append(title, prominent, hint);
  return item;
}

function pair(label, value) {
  const item = document.createElement('div');
  const term = document.createElement('dt');
  const definition = document.createElement('dd');
  term.textContent = label;
  definition.textContent = value;
  item.append(term, definition);
  return item;
}

function renderFrame() {
  const {method, frames, frame, baseline} = current();
  const metrics = frame.metrics;
  const lastStep = frames.at(-1).step;
  const timeline = byId('timeline');
  timeline.max = frames.length - 1;
  timeline.value = state.step;
  timeline.disabled = frames.length < 2;
  timeline.setAttribute('aria-valuetext', `第 ${frame.step} 步，共 ${lastStep} 步`);
  byId('prev').disabled = state.step === 0;
  byId('next').disabled = state.step === frames.length - 1;
  byId('play').disabled = frames.length < 2;
  byId('best').disabled = state.step === bestIndex(method);
  const frameLabel = byId('current-frame');
  if (frames.length === 1) {
    frameLabel.textContent = '单次生成';
  } else if (frame.step === method.best_step) {
    frameLabel.textContent = `第 ${frame.step} 步 · 最佳`;
  } else if (frame.step === 0) {
    frameLabel.textContent = '初始网格';
  } else {
    frameLabel.textContent = `第 ${frame.step} 步 · ${frame.accepted ? '保留' : '未保留'}`;
  }
  frameLabel.classList.toggle('accepted', frame.step > 0 && Boolean(frame.accepted));
  frameLabel.classList.toggle('rejected', frame.step > 0 && frame.accepted === false);
  byId('frame-count').textContent = `${frame.step} / ${lastStep}`;
  byId('fixed-score').textContent = `质量 ${show(baseline.metrics.score)}`;
  byId('active-score').textContent = `质量 ${show(metrics.score)}`;
  byId('case-metrics').replaceChildren(
    metric('当前综合质量分', show(metrics.score), `相对固定参数 ${signed(metrics.score - baseline.metrics.score)}`),
    metric('最小缩放 Jacobian', show(metrics.min_jacobian), `固定参数 ${show(baseline.metrics.min_jacobian)}`),
    metric('P05 缩放 Jacobian', show(metrics.p05_jacobian), `固定参数 ${show(baseline.metrics.p05_jacobian)}`),
    metric('无效单元', `${count(metrics.invalid_cells)} / ${count(metrics.cells)}`, '以单元局部 Jacobian 判定', metrics.invalid_cells > 0),
    metric('本算例整段优化耗时', time(method.elapsed_ms), `${count(method.evaluations)} 次网格评估`)
  );
  byId('parameter-values').replaceChildren(...(frame.params || []).map((value, i) => {
    const name = result.meta.parameter_names?.[i] || `参数 ${i + 1}`;
    return pair(parameterLabels[name] || name, show(value));
  }));
  const detail = result.methods.find((item) => item.id === state.method)?.description;
  byId('method-notes').textContent = detail || '';
  refreshMeshes();
}

function stopPlayback() {
  window.clearInterval(state.timer);
  state.timer = null;
  byId('play-label').textContent = '播放轨迹';
  document.querySelector('.play-icon').classList.remove('paused');
}

function setStep(index) {
  const {frames} = current();
  state.step = Math.max(0, Math.min(index, frames.length - 1));
  renderFrame();
}

function playback() {
  if (state.timer !== null) {
    stopPlayback();
    return;
  }
  const {frames} = current();
  setStep(0);
  byId('play-label').textContent = '暂停播放';
  document.querySelector('.play-icon').classList.add('paused');
  const frameTime = Math.max(140, Math.min(620, 4200 / Math.max(1, frames.length - 1)));
  state.timer = window.setInterval(() => {
    setStep(state.step + 1);
    if (state.step === frames.length - 1) stopPlayback();
  }, frameTime);
}

function selectMethod(method) {
  stopPlayback();
  state.method = method;
  const test = selectedCase();
  for (const [id, button] of methodButtons) {
    button.disabled = !test.methods[id]?.frames?.length;
    button.setAttribute('aria-pressed', String(id === method));
  }
  byId('selected-method').textContent = labels[method] || method;
  setStep(bestIndex(test.methods[method]));
}

function bestIndex(method) {
  const index = method.frames.findIndex((frame) => frame.step === method.best_step);
  return index >= 0 ? index : method.frames.length - 1;
}

function selectView(near) {
  state.near = near;
  byId('near').setAttribute('aria-pressed', String(near));
  byId('full').setAttribute('aria-pressed', String(!near));
  refreshMeshes();
}

function selectMode(heat) {
  state.heat = heat;
  byId('heat').setAttribute('aria-pressed', String(heat));
  byId('wire').setAttribute('aria-pressed', String(!heat));
  byId('legend').hidden = !heat;
  refreshMeshes();
}

function exportComparison() {
  const {test, frame, baseline} = current();
  const left = byId('fixed-canvas');
  const right = byId('active-canvas');
  const width = 1600;
  const plotWidth = (width - 96) / 2;
  const plotHeight = Math.round(plotWidth * Math.max(left.height / left.width, right.height / right.width));
  const y = 125;
  const resultImage = document.createElement('canvas');
  resultImage.width = width;
  resultImage.height = y + plotHeight + 86;
  const ctx = resultImage.getContext('2d');
  ctx.fillStyle = '#f4f7fa';
  ctx.fillRect(0, 0, width, resultImage.height);
  ctx.fillStyle = '#122538';
  ctx.font = '700 30px Microsoft YaHei UI, sans-serif';
  ctx.fillText(`${caseName(test)}  四块结构化网格`, 32, 53);
  const titleList = [`固定参数 · ${show(baseline.metrics.score)}`, `${labels[state.method]} · 第 ${frame.step} 步 · ${show(frame.metrics.score)}`];
  for (const [index, canvas] of [left, right].entries()) {
    const x = 32 + index * (plotWidth + 32);
    ctx.fillStyle = '#1b3148';
    ctx.fillRect(x, y - 38, plotWidth, 38 + plotHeight);
    ctx.fillStyle = '#eef6ff';
    ctx.font = '700 21px Microsoft YaHei UI, sans-serif';
    ctx.fillText(titleList[index], x + 16, y - 12);
    const height = plotWidth * canvas.height / canvas.width;
    ctx.drawImage(canvas, x, y + (plotHeight - height) / 2, plotWidth, height);
  }
  ctx.fillStyle = '#3b5268';
  ctx.font = '20px Microsoft YaHei UI, sans-serif';
  const caption = state.heat ? '有符号缩放 Jacobian' : '网格线 · 亮色线标记块间接口';
  ctx.fillText(`${state.near ? '翼型近场' : '完整网格'}  |  ${caption}`, 32, y + plotHeight + 30);
  if (state.heat) {
    const gradient = ctx.createLinearGradient(404, 0, 565, 0);
    colors.forEach((rgb, index) => gradient.addColorStop(index / 4, `rgb(${rgb.join(',')})`));
    ctx.fillStyle = gradient;
    ctx.fillRect(404, y + plotHeight + 16, 161, 14);
    ctx.fillStyle = '#3b5268';
    ctx.fillText('0', 387, y + plotHeight + 30);
    ctx.fillText('1', 571, y + plotHeight + 30);
    ctx.fillStyle = '#ef547b';
    ctx.fillRect(608, y + plotHeight + 17, 13, 13);
    ctx.fillStyle = '#3b5268';
    ctx.fillText('≤ 0', 629, y + plotHeight + 30);
  }
  resultImage.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${test.name.replace(/[^a-z0-9-]+/gi, '-')}-${state.method}-step-${frame.step}.png`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}

function renderCaseOptions() {
  const all = result.cases.map((test, index) => [index, caseName(test)]);
  if (state.liveCase) all.push([-1, `${caseName(state.liveCase)} · 即时运行`]);
  const options = all.map(([value, label]) => new Option(label, value));
  byId('case-select').replaceChildren(...options);
  byId('case-select').value = state.caseIndex;
}

function runLive(event) {
  event.preventDefault();
  const input = byId('live-code');
  const status = byId('live-status');
  status.classList.remove('error');
  input.removeAttribute('aria-invalid');
  try {
    const test = window.runMeshPolicy(input.value.trim());
    state.liveCase = test;
    state.caseIndex = -1;
    renderCaseOptions();
    selectMethod('sac');
    status.textContent = `已运行 ${caseName(test)}，当前显示最佳网格。`;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : '无法运行此翼型。';
    status.classList.add('error');
    input.setAttribute('aria-invalid', 'true');
    input.focus();
  }
}

function renderSummary() {
  const fixed = result.summary.fixed;
  const sac = result.summary.sac;
  const testCount = result.meta.test_cases;
  const gain = fixed.score ? 100 * (sac.score - fixed.score) / fixed.score : NaN;
  byId('hero-score').textContent = show(sac.score, 4);
  byId('hero-delta').textContent = Number.isFinite(gain) ? `${gain >= 0 ? '+' : ''}${gain.toFixed(2)}%` : '—';
  byId('hero-valid').textContent = `${count(Math.round(sac.valid_rate * testCount))}/${count(testCount)}`;
  byId('benchmark-copy').textContent = `${count(testCount)} 个留出翼型参与比较。SAC 在 ${count(sac.strict_wins_vs_fixed)} 个翼型上获得高于固定参数的质量分。`;
  const rows = result.methods.map(({id, label}) => {
    const method = result.summary[id];
    const row = document.createElement('tr');
    if (id === 'sac') row.className = 'sac-row';
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = label;
    row.append(heading);
    const fields = [show(method.score, 5), show(method.mean_min_jacobian, 4), show(method.mean_p05_jacobian, 4),
      show(method.mean_skew, 4), rate(method.valid_rate), time(method.median_ms), show(method.mean_evaluations, 1)];
    for (const value of fields) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  byId('benchmark-rows').replaceChildren(...rows);
}

function evidenceTable(title, columns, rows, collapsible = false) {
  const group = document.createElement(collapsible ? 'details' : 'section');
  group.className = 'evidence-group';
  const heading = document.createElement(collapsible ? 'summary' : 'h3');
  heading.textContent = title;
  const wrap = document.createElement('div');
  wrap.className = 'evidence-table';
  wrap.tabIndex = 0;
  wrap.setAttribute('aria-label', title);
  const table = document.createElement('table');
  table.setAttribute('aria-label', title);
  const headers = table.createTHead().insertRow();
  for (const label of columns) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    headers.append(cell);
  }
  const tbody = table.createTBody();
  for (const {label, values, highlight} of rows) {
    const row = tbody.insertRow();
    if (highlight) row.className = 'sac-row';
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = label;
    row.append(name);
    for (const value of values) row.insertCell().textContent = value;
  }
  wrap.append(table);
  group.append(heading, wrap);
  return group;
}

function renderEvidence(evidence) {
  if (!evidence?.methodology || !evidence.rows?.length) return;
  const {methodology, rows, comparisons, frontier, papers, conclusion} = evidence;
  const split = [methodology.train_cases, methodology.val_cases, methodology.test_cases].map(count).join(' / ');
  byId('evidence-context').textContent = `${count(methodology.seeds.length)} 个训练种子，每次训练 / 验证 / 测试使用 ${split} 个翼型。粗网格和细网格分别有 ${count(methodology.grid_cells)} 和 ${count(methodology.fine_cells)} 个单元。`;
  byId('evidence-conclusion').textContent = conclusion;
  const groups = [evidenceTable('几何质量与计算预算',
    ['方法', '每例网格评估数', '粗网格均分', '种子间标准差', '粗网格有效率', '首层平均间距 / 弦长', '细网格均分'],
    rows.map((row) => ({
      label: row.label, highlight: row.id === 'sac', values: [row.budget ?? '—', show(row.score, 5), show(row.seed_sd, 5),
        rate(row.valid_rate), show(row.mean_first_layer, 5), show(row.fine_score, 5)]
    }))
  )];
  if (comparisons?.length) {
    const diff = (value) => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(5)}` : '—';
    groups.push(evidenceTable('SAC 与各方法的配对比较',
      ['对照方法', 'SAC 平均分差', '严格胜出配对', '各种子平均分差范围'],
      comparisons.map((row) => ({label: row.label, values: [diff(row.mean_delta), `${count(row.wins)} / ${count(row.total)}`,
        `${diff(row.seed_min)} 至 ${diff(row.seed_max)}`]}))
    ));
  }
  if (frontier?.length) {
    groups.push(evidenceTable('查看各评估预算下的粗网格均分',
      ['每例网格评估数', '固定参数', '全局调参', '坐标搜索', '行为克隆', 'SAC'],
      frontier.map((row) => ({label: count(row.evaluations), values: [row.fixed, row.global, row.local, row.bc, row.sac].map((value) => show(value, 5))})), true
    ));
  }
  byId('evidence-body').replaceChildren(...groups);
  byId('evidence-limit').textContent = '同一留出翼型在不同种子下重复评估。胜出数按种子与翼型配对计，分差范围取各训练种子的平均分差。实验范围为二维固定四块 O 型拓扑及几何质量。';
  const sourceList = byId('evidence-sources');
  for (const paper of papers || []) {
    if (!/^https:\/\//i.test(paper.url)) continue;
    const item = document.createElement('li');
    if (paper.detail) item.textContent = `${paper.detail} `;
    const link = document.createElement('a');
    link.href = paper.url;
    link.textContent = paper.title;
    item.append(link);
    sourceList.append(item);
  }
  sourceList.hidden = !sourceList.children.length;
  byId('evidence').hidden = false;
  byId('evidence-jump').hidden = false;
}

function drawHistory() {
  const series = result.history.filter((point) => Number.isFinite(point.score) && Number.isFinite(point.step));
  const {ctx, width, height} = canvasContext(byId('history-canvas'));
  ctx.clearRect(0, 0, width, height);
  if (!series.length || width < 80) return;
  const scores = series.map((point) => point.score);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const padding = Math.max(.00005, (max - min) * .13);
  const lower = min - padding;
  const upper = max + padding;
  const first = series[0];
  const last = series.at(-1);
  const best = series.find((point) => point.step === result.meta.best_step) || series.find((point) => point.score === max);
  const frame = {left: 57, top: 20, right: width - 23, bottom: height - 35};
  const x = (step) => frame.left + (step - first.step) / Math.max(1, last.step - first.step) * (frame.right - frame.left);
  const y = (score) => frame.bottom - (score - lower) / (upper - lower) * (frame.bottom - frame.top);
  ctx.font = '12px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const score = lower + (upper - lower) * i / 4;
    const position = y(score);
    ctx.beginPath();
    ctx.moveTo(frame.left, position);
    ctx.lineTo(frame.right, position);
    ctx.strokeStyle = '#e0e8ee';
    ctx.stroke();
    ctx.fillStyle = '#4d6478';
    ctx.fillText(show(score, 4), frame.left - 10, position);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i += 1) {
    const step = Math.round(first.step + (last.step - first.step) * i / 4);
    ctx.fillText(step >= 1000 && step % 1000 === 0 ? `${step / 1000}k` : count(step), x(step), frame.bottom + 9);
  }
  ctx.textAlign = 'right';
  ctx.font = '12px Microsoft YaHei UI, sans-serif';
  ctx.fillText('环境步数', frame.right, height - 13);
  const gradient = ctx.createLinearGradient(0, frame.top, 0, frame.bottom);
  gradient.addColorStop(0, '#26a99664');
  gradient.addColorStop(1, '#26a99604');
  ctx.beginPath();
  ctx.moveTo(x(first.step), frame.bottom);
  for (const point of series) ctx.lineTo(x(point.step), y(point.score));
  ctx.lineTo(x(last.step), frame.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x(first.step), y(first.score));
  for (const point of series.slice(1)) ctx.lineTo(x(point.step), y(point.score));
  ctx.strokeStyle = '#087d75';
  ctx.lineWidth = 2.1;
  ctx.lineJoin = 'round';
  ctx.stroke();
  for (const point of series) {
    ctx.beginPath();
    ctx.arc(x(point.step), y(point.score), point === best ? 5.1 : 2.15, 0, 2 * Math.PI);
    ctx.fillStyle = point === best ? '#0b655f' : '#13948b';
    ctx.fill();
    if (point === best) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.9;
      ctx.stroke();
    }
  }
  byId('chart-caption').textContent = `选用检查点 ${count(best.step)} 步 · ${show(best.score, 5)}`;
  byId('history-description').textContent = `验证从 ${count(first.step)} 步的 ${show(first.score, 5)} 开始，最终点 ${count(last.step)} 步得分 ${show(last.score, 5)}。选用 ${count(best.step)} 步检查点。纵轴为局部范围。`;
}

function renderTraining() {
  const meta = result.meta;
  const fields = [
    ['训练 / 验证 / 测试翼型', `${count(meta.train_cases)} / ${count(meta.val_cases)} / ${count(meta.test_cases)}`],
    ['SAC 训练步数', count(meta.steps)], ['模型检查点', `${count(meta.best_step)} 步`],
    ['每轮调参步数', count(meta.horizon)], ['训练用时', `${show(meta.train_seconds, 1)} s`], ['随机种子', count(meta.seed)]
  ];
  byId('training-meta').replaceChildren(...fields.map(([label, value]) => pair(label, value)));
  drawHistory();
}

function init() {
  if (!result?.cases?.length) {
    byId('data-status').textContent = '未找到实验数据。请将结果文件与本页面放在同一目录。';
    return;
  }
  byId('data-status').hidden = true;
  renderCaseOptions();
  const select = byId('case-select');
  select.disabled = false;
  select.addEventListener('change', () => {
    state.caseIndex = Number(select.value);
    const method = selectedCase().methods[state.method]?.frames?.length ? state.method : 'sac';
    selectMethod(method);
  });
  const methodOptions = byId('method-options');
  for (const {id, label, description} of result.methods) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.title = description;
    button.addEventListener('click', () => selectMethod(id));
    methodButtons.set(id, button);
    methodOptions.append(button);
  }
  byId('near').addEventListener('click', () => selectView(true));
  byId('full').addEventListener('click', () => selectView(false));
  byId('wire').addEventListener('click', () => selectMode(false));
  byId('heat').addEventListener('click', () => selectMode(true));
  byId('play').addEventListener('click', playback);
  byId('prev').addEventListener('click', () => { stopPlayback(); setStep(state.step - 1); });
  byId('next').addEventListener('click', () => { stopPlayback(); setStep(state.step + 1); });
  byId('best').addEventListener('click', () => { stopPlayback(); setStep(bestIndex(current().method)); });
  byId('timeline').addEventListener('input', (event) => { stopPlayback(); setStep(Number(event.target.value)); });
  byId('export').disabled = false;
  byId('export').addEventListener('click', exportComparison);
  byId('live-form').hidden = typeof window.runMeshPolicy !== 'function';
  byId('live-form').addEventListener('submit', runLive);
  document.querySelectorAll('nav a').forEach((link) => {
    link.addEventListener('click', () => {
      document.querySelector('nav a[aria-current]')?.removeAttribute('aria-current');
      link.setAttribute('aria-current', 'location');
    });
  });
  selectMethod('sac');
  renderSummary();
  renderTraining();
  renderEvidence(window.MESH_EVIDENCE);
  if (typeof ResizeObserver === 'function') {
    const redraw = new ResizeObserver(() => { refreshMeshes(); drawHistory(); });
    redraw.observe(byId('fixed-canvas'));
    redraw.observe(byId('history-canvas'));
  } else {
    window.addEventListener('resize', () => { refreshMeshes(); drawHistory(); });
  }
}

init();
