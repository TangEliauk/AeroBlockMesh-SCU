/* Offline inference using weights exported by export_policy.py.
 * Geometry, observations, and quality mirror mesh_env.py and experiment.py.
 */
(() => {
  'use strict';

  const twoPi = 2 * Math.PI;
  const radialCells = 16;
  const cellsPerBlock = 24;
  const clamp = (value, low, high) => Math.max(low, Math.min(value, high));
  const f32 = Math.fround;

  function parseCase(code) {
    const name = String(code).trim();
    if (!/^[0-9]{4}$/.test(name)) throw new Error('请输入四位 NACA 编号，例如 2412。');
    const m = Number(name[0]) / 100;
    const p = Number(name[1]) / 10;
    const t = Number(name.slice(2)) / 100;
    if (m > 0.06 || t < 0.08 || t > 0.24 || (m === 0 ? p !== 0 : p < 0.2 || p > 0.7)) {
      throw new Error('支持弯度 0–6%、厚度 8–24%。对称翼型第二位为 0，其余翼型第二位为 2–7。');
    }
    return { name, m, p, t };
  }

  function surface(airfoil, theta) {
    const angle = theta < 0 ? theta + twoPi : theta >= twoPi ? theta - twoPi : theta;
    const x = 0.5 * (1 + Math.cos(angle));
    const thickness = x === 1 ? 0 : 5 * airfoil.t * (
      0.2969 * Math.sqrt(x) - x * (0.1260 + x * (0.3516 - x * (0.2843 - 0.1036 * x)))
    );
    const offset = angle <= Math.PI ? thickness : -thickness;
    if (!airfoil.m) return [x, offset];

    const { m, p } = airfoil;
    const leading = x < p;
    const denominator = leading ? p ** 2 : (1 - p) ** 2;
    const camber = m / denominator * (leading ? x * (2 * p - x) : (1 - x) * (1 + x - 2 * p));
    const tangent = Math.atan(2 * m * (p - x) / denominator);
    return [x - offset * Math.sin(tangent), camber + offset * Math.cos(tangent)];
  }

  function generate(airfoil, parameters) {
    const [spacing, normalBlend, bias, upper, lower] = parameters;
    const breaks = [0, Math.PI / 2 + 0.6 * upper, Math.PI, 1.5 * Math.PI + 0.6 * lower, twoPi];
    const columns = [];
    for (let block = 0; block < 4; block++) {
      for (let index = 0; index < cellsPerBlock; index++) {
        let theta = breaks[block] + (breaks[block + 1] - breaks[block]) * (index / cellsPerBlock);
        theta += 0.26 * bias * Math.sin(theta);
        const wall = surface(airfoil, theta);
        const before = surface(airfoil, theta - 0.0002);
        const after = surface(airfoil, theta + 0.0002);
        const dx = 0.5 + 4 * Math.cos(theta) - wall[0];
        const dy = 4 * Math.sin(theta) - wall[1];
        const nx = after[1] - before[1];
        const ny = before[0] - after[0];
        const normalScale = Math.hypot(dx, dy) / Math.hypot(nx, ny);
        columns.push({ wall, dx, dy, ox: nx * normalScale - dx, oy: ny * normalScale - dy });
      }
    }
    const exponent = 4.8 - 1.35 * spacing;
    const mesh = [];
    for (let i = 0; i <= radialCells; i++) {
      const r = Math.expm1(exponent * (i / radialCells)) / Math.expm1(exponent);
      const blend = (0.45 + 0.40 * normalBlend) * r * (1 - r) ** 14;
      const row = columns.map(({ wall, dx, dy, ox, oy }) => [
        wall[0] + r * dx + blend * ox, wall[1] + r * dy + blend * oy,
      ]);
      row.push(row[0]);
      mesh.push(row);
    }
    return mesh;
  }

  function quality(mesh) {
    const cellQuality = [];
    const all = [];
    let invalid = 0, sumJacobian = 0, maxAspect = 0, sumSkew = 0, sumAspectScore = 0, wallLength = 0;
    for (let i = 0; i < mesh.length - 1; i++) {
      const row = [];
      for (let j = 0; j < mesh[i].length - 1; j++) {
        const corners = [mesh[i][j], mesh[i + 1][j], mesh[i + 1][j + 1], mesh[i][j + 1]];
        const edges = corners.map((point, k) => {
          const next = corners[(k + 1) % 4];
          return [next[0] - point[0], next[1] - point[1]];
        });
        const lengths = edges.map(([x, y]) => Math.hypot(x, y));
        const aspect = Math.max(...lengths) / Math.max(2.2250738585072014e-308, Math.min(...lengths));
        let jacobian = 1;
        for (let k = 0; k < 4; k++) {
          const next = (k + 1) % 4;
          const cross = edges[k][0] * edges[next][1] - edges[k][1] * edges[next][0];
          const scale = lengths[k] * lengths[next];
          jacobian = Math.min(jacobian, scale > 0 ? cross / scale : 0);
        }
        jacobian = Math.max(-1, jacobian);
        if (jacobian <= 1e-10) invalid++;
        if (i === 0) wallLength += lengths[0];
        sumJacobian += jacobian;
        sumSkew += Math.acos(clamp(jacobian, 0, 1));
        sumAspectScore += 1 / (1 + Math.log1p(Math.max(aspect - 1, 0)));
        maxAspect = Math.max(maxAspect, aspect);
        row.push(jacobian);
        all.push(jacobian);
      }
      cellQuality.push(row);
    }
    all.sort((a, b) => a - b);
    const count = all.length;
    const minimum = all[0];
    const p05 = all[Math.floor(0.05 * (count - 1))];
    const mean = sumJacobian / count;
    const skew = sumSkew / count * (2 / Math.PI);
    const rawScore = 0.45 * p05 + 0.24 * mean + 0.12 * minimum + 0.09 * (1 - skew) + 0.10 * sumAspectScore / count;
    return {
      cellQuality,
      metrics: {
        score: Math.min(1, Math.max(0, rawScore) / (1 + 0.75 * invalid)),
        min_jacobian: minimum,
        invalid_cells: invalid,
        p05_jacobian: p05,
        mean_jacobian: mean,
        mean_skew: skew,
        max_aspect_ratio: maxAspect,
        mean_first_layer: wallLength / (mesh[0].length - 1),
        cells: count,
      },
    };
  }

  function observation(airfoil, parameters, metrics, step, horizon) {
    return Float32Array.from([
      airfoil.m / 0.06, airfoil.p, airfoil.t / 0.24, ...parameters,
      metrics.score, metrics.min_jacobian, metrics.p05_jacobian,
      metrics.mean_skew, metrics.invalid_cells / metrics.cells, step / horizon,
    ]);
  }

  function action(observed, policy) {
    let inputs = observed;
    for (const [index, { weights, bias }] of policy.layers.entries()) {
      const outputs = new Float32Array(bias.length);
      for (let j = 0; j < outputs.length; j++) {
        let dot = 0;
        for (let i = 0; i < inputs.length; i++) dot += inputs[i] * weights[i][j];
        const value = f32(f32(dot) + bias[j]);
        outputs[j] = index < policy.layers.length - 1 ? Math.max(0, value) : value;
      }
      inputs = outputs;
    }
    return Array.from(inputs.slice(0, policy.actionDim), (value) => f32(Math.tanh(value)));
  }

  function isBetter(candidate, incumbent) {
    const candidateValid = candidate.invalid_cells === 0;
    const incumbentValid = incumbent.invalid_cells === 0;
    return candidateValid !== incumbentValid ? candidateValid : candidate.score > incumbent.score;
  }

  function record(airfoil, step, params, previous, incumbent) {
    const mesh = generate(airfoil, params);
    const { metrics, cellQuality } = quality(mesh);
    const accepted = !incumbent || isBetter(metrics, incumbent);
    return {
      step, params, metrics, accepted, mesh, cell_quality: cellQuality,
      reward: previous ? 12 * (metrics.score - previous.score) : 0,
      best_score: accepted ? metrics.score : incumbent.score,
    };
  }

  window.runMeshPolicy = function runMeshPolicy(code) {
    const airfoil = parseCase(code);
    const policy = window.MESH_POLICY_WEIGHTS;
    if (!policy) throw new Error('模型权重加载失败。');

    const start = performance.now();
    let current = record(airfoil, 0, policy.baseParameters);
    let best = current;
    const frames = [current];
    const fixed = {
      metrics: current.metrics, elapsed_ms: performance.now() - start, evaluations: 1,
      best_step: 0, parameters: current.params, frames: [current],
    };
    for (let step = 0; step < policy.horizon; step++) {
      const observed = observation(airfoil, current.params, current.metrics, step, policy.horizon);
      const move = action(observed, policy);
      const proposal = current.params.map((value, index) => (
        f32(clamp(f32(value + f32(f32(policy.actionScale) * move[index])), -1, 1))
      ));
      current = record(airfoil, step + 1, proposal, current.metrics, best.metrics);
      frames.push(current);
      if (current.accepted) best = current;
    }

    const sac = {
      metrics: best.metrics, elapsed_ms: performance.now() - start, evaluations: frames.length,
      best_step: best.step, parameters: best.params, frames,
    };
    return { ...airfoil, methods: { fixed, sac } };
  };
})();
