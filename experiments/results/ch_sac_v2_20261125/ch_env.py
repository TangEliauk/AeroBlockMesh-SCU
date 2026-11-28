"""Pure-NumPy 7-control decoupled C-H environment and geometry feature extractor."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_wake_gmsh as wake
from generate_airfoil_gmsh import FARFIELD_RADIUS
from smartcfd.core.block_graph import BlockGraph
from smartcfd.meshing.gmsh_graph import capture, model_session
from smartcfd.meshing.tfi import generate
from smartcfd.partition.edits import Cut, split

OUTLET_X = FARFIELD_RADIUS + 1.0
BUDGETS = (1536, 5120)
GEOM_DIM = 18
OBS_DIM = 38
ACTION_DIM = 7
MAX_STEPS = 5
STEP_SCALE = 0.40

PARAM_SPECS = (
    ("split_frac", 0.20, 0.80, "linear", 0.50),
    ("exit_first_dy", 0.008, 0.30, "log", 0.03),
    ("wake_top_dx", 0.015, 0.35, "log", 0.05),
    ("aft_top_bump", 0.20, 1.00, "linear", 0.60),
    ("dcut_u", -2.5, 2.5, "linear", 0.0),
    ("dcut_l", -2.5, 2.5, "linear", 0.0),
    ("wake_deg", -6.0, 6.0, "linear", 0.0),
)
PARAM_NAMES = tuple(spec[0] for spec in PARAM_SPECS)
DEFAULT_PARAMS = {name: default for name, _, _, _, default in PARAM_SPECS}


def params_to_u(params):
    """Map physical parameters into normalized [-1, 1]^7 coordinates."""
    vals = []
    for name, lo, hi, scale, _ in PARAM_SPECS:
        v = float(params[name])
        if scale == "log":
            t = (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            t = (v - lo) / (hi - lo)
        vals.append(float(np.clip(2.0 * t - 1.0, -1.0, 1.0)))
    return np.asarray(vals, dtype=np.float32)


def u_to_params(u):
    """Map normalized [-1, 1]^7 coordinates back to physical parameter dict."""
    arr = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
    out = {}
    for idx, (name, lo, hi, scale, _) in enumerate(PARAM_SPECS):
        t = 0.5 * (float(arr[idx]) + 1.0)
        if scale == "log":
            out[name] = 10.0 ** (math.log10(lo) + t * (math.log10(hi) - math.log10(lo)))
        else:
            out[name] = lo + t * (hi - lo)
    return out


DEFAULT_U = params_to_u(DEFAULT_PARAMS)


def is_blunt_branches(branches):
    """Return True if trailing edge has open base (matching wake.counts_for_budget)."""
    return bool(branches[0][-1] != branches[1][-1])


def _sorted_interp(branch_xy, x_eval):
    pts = np.asarray(branch_xy, dtype=float)
    order = np.argsort(pts[:, 0])
    xs = pts[order, 0]
    ys = pts[order, 1]
    _, uniq_idx = np.unique(xs, return_index=True)
    return np.interp(x_eval, xs[uniq_idx], ys[uniq_idx])


def extract_geometry_features(polygon, branches):
    """Extract 18 normalized chordwise thickness, camber, LE and TE features.
    In generate_wake_gmsh.read_profile, branches = (lower_indices, upper_indices).
    """
    poly = np.asarray(polygon, dtype=float)
    lower_xy = poly[branches[0]]
    upper_xy = poly[branches[1]]
    x_dense = np.linspace(0.0, 1.0, 101)
    yu_dense = _sorted_interp(upper_xy, x_dense)
    yl_dense = _sorted_interp(lower_xy, x_dense)
    thick_dense = yu_dense - yl_dense
    camber_dense = 0.5 * (yu_dense + yl_dense)

    idx_t = int(np.argmax(thick_dense))
    idx_c = int(np.argmax(np.abs(camber_dense)))
    max_t = float(thick_dense[idx_t])
    x_max_t = float(x_dense[idx_t])
    max_c = float(camber_dense[idx_c])
    x_max_c = float(x_dense[idx_c])

    yu_005, yu_095, yu_100 = _sorted_interp(upper_xy, [0.05, 0.95, 1.0])
    yl_005, yl_095, yl_100 = _sorted_interp(lower_xy, [0.05, 0.95, 1.0])
    le_thick_005 = float(yu_005 - yl_005)
    te_upper_slope = float(np.clip(math.atan2(yu_095 - yu_100, -0.05) / 0.6, -1.5, 1.5))
    te_lower_slope = float(np.clip(math.atan2(yl_095 - yl_100, -0.05) / 0.6, -1.5, 1.5))
    te_wedge = float(np.clip(te_upper_slope - te_lower_slope, 0.0, 2.0))
    te_bisector = 0.5 * (te_upper_slope + te_lower_slope)
    blunt = 1.0 if is_blunt_branches(branches) else 0.0
    te_gap = float(abs(yu_100 - yl_100))

    x_t_samples = np.array([0.15, 0.30, 0.60, 0.85])
    x_c_samples = np.array([0.20, 0.50, 0.80])
    t_samples = (_sorted_interp(upper_xy, x_t_samples) - _sorted_interp(lower_xy, x_t_samples)) / 0.25
    c_samples = 0.5 * (_sorted_interp(upper_xy, x_c_samples) + _sorted_interp(lower_xy, x_c_samples)) / 0.10

    feats = np.array(
        [
            max_t / 0.25,
            x_max_t,
            max_c / 0.10,
            x_max_c,
            le_thick_005 / 0.10,
            te_upper_slope,
            te_lower_slope,
            te_wedge,
            te_bisector,
            blunt,
            te_gap / 0.02,
            *t_samples.tolist(),
            *c_samples.tolist(),
        ],
        dtype=np.float32,
    )
    return np.clip(feats, -3.0, 3.0)


def _prog_pts(a, b, n, first_len):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    if n <= 1 or length <= 1e-12:
        return ()
    first_len = max(1e-6, min(float(first_len), length * 0.85))
    r = wake.progression(length, n, first_len)
    seg = r ** np.arange(n)
    t = np.cumsum(seg)[:-1] / seg.sum()
    pts = a + t[:, None] * (b - a)
    return tuple((float(x), float(y)) for x, y in pts)


def _bump_pts(a, b, n, bump):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if n <= 1:
        return ()
    u = np.linspace(0.0, 1.0, n + 1)
    t = (1.0 - bump) * 0.5 * (1.0 - np.cos(np.pi * u)) + bump * u
    pts = a + t[1:-1, None] * (b - a)
    return tuple((float(x), float(y)) for x, y in pts)


def _shift_curve_endpoints(old_p0, old_p1, old_stations, new_p0, new_p1):
    if not old_stations:
        return ()
    pts = np.vstack([old_p0, np.asarray(old_stations, dtype=float), old_p1])
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 1e-12:
        t = np.linspace(0.0, 1.0, len(old_stations) + 2)[1:-1]
    else:
        t = s[1:-1] / s[-1]
    disp0 = np.asarray(new_p0, dtype=float) - np.asarray(old_p0, dtype=float)
    disp1 = np.asarray(new_p1, dtype=float) - np.asarray(old_p1, dtype=float)
    shifted = np.asarray(old_stations, dtype=float) + (1.0 - t)[:, None] * disp0 + t[:, None] * disp1
    return tuple((float(x), float(y)) for x, y in shifted)


def build_g0(airfoil_name, data_dir, budget):
    """Build baseline Gmsh-captured C-H BlockGraph g0 and geometry metadata."""
    _, polygon, branches = wake.read_profile(airfoil_name, Path(data_dir))
    blunt = is_blunt_branches(branches)
    if budget == 1536:
        counts = wake.MeshCounts(radial=12, front=14, aft=24, wake=26, base=2 if blunt else 0)
    else:
        counts = wake.counts_for_budget(budget, blunt)
    with model_session():
        blocks, seams, *_ = wake.prepare_wake_curves(
            polygon, branches, FARFIELD_RADIUS, counts, "smooth-balanced", None, gmsh.model.getCurrent()
        )
        labels = {
            int(e): gmsh.model.getPhysicalName(1, t)
            for _, t in gmsh.model.getPhysicalGroups(1)
            for e in gmsh.model.getEntitiesForPhysicalGroup(1, t)
        }
        g0 = capture({s: l for l, s in blocks.items()}, labels, roles=dict.fromkeys(seams, "wake"))
    geom = extract_geometry_features(polygon, branches)
    return g0, blunt, int(counts.quads), geom


def apply_7controls(g0: BlockGraph, params: dict) -> BlockGraph:
    """Apply 7 physical controls + topological wake split while preserving airfoil y1."""
    split_frac = float(params["split_frac"])
    exit_first_dy = float(params["exit_first_dy"])
    wake_top_dx = float(params["wake_top_dx"])
    aft_top_bump = float(params["aft_top_bump"])
    dcut_u = float(params["dcut_u"])
    dcut_l = float(params["dcut_l"])
    wake_deg = float(params["wake_deg"])

    verts = {vid: (float(xy[0]), float(xy[1])) for vid, xy in g0.vertices.items()}
    dy_wake = (OUTLET_X - 1.0) * math.tan(math.radians(wake_deg))

    moved_vids = set()
    for vid, (x, y) in list(verts.items()):
        if abs(abs(y) - FARFIELD_RADIUS) < 1e-5 and (0.2 < x < OUTLET_X - 0.5):
            dx = dcut_u if y > 0 else dcut_l
            verts[vid] = (x + dx, y)
            moved_vids.add(vid)
        elif abs(x - OUTLET_X) < 1e-5 and abs(y) < FARFIELD_RADIUS - 1e-5:
            verts[vid] = (x, y + dy_wake)
            moved_vids.add(vid)

    new_curves = {}
    for cid, c in g0.curves.items():
        old_p0 = np.array(g0.vertices[c.start], dtype=float)
        old_p1 = np.array(g0.vertices[c.end], dtype=float)
        p0 = np.array(verts[c.start], dtype=float)
        p1 = np.array(verts[c.end], dtype=float)
        n = c.intervals

        if (
            c.boundary == "farfield"
            and abs(abs(p0[1]) - FARFIELD_RADIUS) < 1e-5
            and abs(abs(p1[1]) - FARFIELD_RADIUS) < 1e-5
        ):
            if max(p0[0], p1[0]) > OUTLET_X - 1e-5:
                flip = p1[0] < p0[0]
                a, b = (p1, p0) if flip else (p0, p1)
                pts = _prog_pts(a, b, n, wake_top_dx)
                if flip:
                    pts = pts[::-1]
                new_curves[cid] = replace(c, stations=pts)
            else:
                pts = _bump_pts(p0, p1, n, aft_top_bump)
                new_curves[cid] = replace(c, stations=pts)
        elif c.start in moved_vids or c.end in moved_vids:
            new_curves[cid] = replace(c, stations=_shift_curve_endpoints(old_p0, old_p1, c.stations, p0, p1))
        else:
            new_curves[cid] = c

    g_mid = BlockGraph(verts, new_curves, g0.blocks)

    wcid = next(cid for cid, c in g_mid.curves.items() if c.role == "wake")
    g_sp = None
    for bid, blk in g_mid.blocks.items():
        abs_edges = [abs(e) for e in blk.edges]
        if wcid in abs_edges:
            side = abs_edges.index(wcid)
            n_wake = g_mid.curves[wcid].intervals
            k = max(2, min(n_wake - 2, int(round(n_wake * split_frac))))
            g_sp = split(g_mid, Cut(bid, side, k), guide="inherited")
            break
    if g_sp is None:
        raise RuntimeError("Could not locate wake seam block for split")

    new_curves2 = dict(g_sp.curves)
    for cid, c in g_sp.curves.items():
        if c.boundary != "farfield":
            continue
        p0 = np.array(g_sp.vertices[c.start], dtype=float)
        p1 = np.array(g_sp.vertices[c.end], dtype=float)
        if abs(p0[0] - OUTLET_X) < 1e-5 and abs(p1[0] - OUTLET_X) < 1e-5:
            if max(abs(p0[1]), abs(p1[1])) > FARFIELD_RADIUS - 1e-5:
                flip = abs(p1[1]) < abs(p0[1])
                a, b = (p1, p0) if flip else (p0, p1)
                pts = _prog_pts(a, b, c.intervals, exit_first_dy)
                if flip:
                    pts = pts[::-1]
                new_curves2[cid] = replace(c, stations=pts)

    return BlockGraph(g_sp.vertices, new_curves2, g_sp.blocks)


def eval_graph(g: BlockGraph, return_mesh=False):
    """Run TFI generation and compute physics-aware partitioned quality metrics."""
    mesh = generate(g)
    pts = np.asarray(mesh.points, dtype=float)
    cells = np.vstack(list(mesh.quads.values()))
    xy = pts[cells]
    edges = np.roll(xy, -1, axis=1) - xy
    previous = np.roll(edges, 1, axis=-1) if False else np.roll(edges, 1, axis=-2)
    jac = previous[..., 0] * edges[..., 1] - previous[..., 1] * edges[..., 0]
    lengths = np.linalg.norm(edges, axis=-1)
    scaled = jac / np.maximum(lengths * np.roll(lengths, 1, axis=-1), 1e-30)
    cell_j = scaled.min(axis=-1)

    la = 0.5 * (lengths[:, 0] + lengths[:, 2])
    lb = 0.5 * (lengths[:, 1] + lengths[:, 3])
    ar = np.maximum(la, lb) / np.maximum(np.minimum(la, lb), 1e-30)
    ctr = xy.mean(axis=1)

    far_mask = (np.abs(ctr[:, 1]) > 0.25) | (ctr[:, 0] > 1.5) | (ctr[:, 0] < -0.5)
    near_wall_mask = (np.abs(ctr[:, 1]) <= 0.25) & (ctr[:, 0] >= -0.5) & (ctr[:, 0] <= 1.2)
    near_wake_mask = (ctr[:, 0] >= 1.0) & (ctr[:, 0] <= 3.5) & (np.abs(ctr[:, 1]) <= 1.0)
    far_wake_mask = ctr[:, 0] > 3.5

    far_ar = float(ar[far_mask].max())
    far_ar_q95 = float(np.quantile(ar[far_mask], 0.95))
    max_ar = float(ar.max())
    q = mesh.quality
    jmin = float(q["min_scaled_jacobian"])
    j05 = float(q["q05_scaled_jacobian"])
    wall_q10 = float(q["wall_angle_q10_deg"])

    jmin_nw = float(cell_j[near_wall_mask].min()) if np.any(near_wall_mask) else jmin
    jmin_wk = float(cell_j[near_wake_mask].min()) if np.any(near_wake_mask) else jmin
    jmin_fw = float(cell_j[far_wake_mask].min()) if np.any(far_wake_mask) else jmin

    wall_term = float(np.clip((wall_q10 - 60.0) / 30.0, 0.0, 1.0))
    far_ar_pen = max(0.0, math.log10(max(far_ar, 1.0)) - math.log10(150.0))
    score = 0.40 * j05 + 0.35 * jmin + 0.25 * wall_term - 0.35 * far_ar_pen

    metrics = {
        "quads": int(q["quads"]),
        "blocks": int(len(g.blocks)),
        "min_scaled_jacobian": jmin,
        "q05_scaled_jacobian": j05,
        "wall_angle_q10_deg": wall_q10,
        "farfield_max_aspect_ratio": far_ar,
        "farfield_q95_aspect_ratio": far_ar_q95,
        "max_aspect_ratio": max_ar,
        "jmin_near_wall": jmin_nw,
        "jmin_near_wake": jmin_wk,
        "jmin_far_wake": jmin_fw,
        "score": float(score),
    }
    if return_mesh:
        return metrics, mesh
    return metrics


def evaluate_u(item, u):
    """Evaluate normalized parameter vector u on a cached item dict."""
    g0 = item["g0"] if "g0" in item else BlockGraph.from_dict(item["g0_dict"])
    g = apply_7controls(g0, u_to_params(u))
    m = eval_graph(g)
    if m["quads"] != item["quads"]:
        raise ValueError(f"Quad mismatch {m['quads']} != {item['quads']}")
    return m


def load_cached_split(path):
    """Load pickled split and hydrate in-memory BlockGraph objects."""
    import gzip
    import pickle

    with gzip.open(path, "rb") as fh:
        items = pickle.load(fh)
    for it in items:
        if "g0" not in it and "g0_dict" in it:
            it["g0"] = BlockGraph.from_dict(it["g0_dict"])
    return items


def build_obs(geom_features, u, metrics, step_idx, budget, max_steps=MAX_STEPS):
    """Build 38-dim normalized observation vector."""
    diag = np.array(
        [
            metrics["min_scaled_jacobian"],
            metrics["q05_scaled_jacobian"],
            metrics["wall_angle_q10_deg"] / 90.0,
            (math.log10(max(metrics["farfield_max_aspect_ratio"], 1.0)) - 2.0) / 2.0,
            (math.log10(max(metrics["farfield_q95_aspect_ratio"], 1.0)) - 1.5) / 2.0,
            (math.log10(max(metrics["max_aspect_ratio"], 1.0)) - 2.5) / 2.0,
            metrics["jmin_near_wall"],
            metrics["jmin_near_wake"],
            metrics["jmin_far_wake"],
            metrics["score"],
        ],
        dtype=np.float32,
    )
    ctx = np.array(
        [float(step_idx) / float(max_steps), 1.0 if budget == 1536 else 0.0, 1.0 if budget == 5120 else 0.0],
        dtype=np.float32,
    )
    obs = np.concatenate([np.asarray(geom_features, dtype=np.float32), np.asarray(u, dtype=np.float32), diag, ctx])
    return np.clip(obs, -5.0, 5.0).astype(np.float32)


class CHMeshEnv:
    """Short-horizon (T=5) continuous control MDP over a cached C-H BlockGraph."""

    def __init__(self, item, max_steps=MAX_STEPS, step_scale=STEP_SCALE):
        self.item = item
        self.max_steps = int(max_steps)
        self.step_scale = float(step_scale)
        self.u = DEFAULT_U.copy()
        self.metrics = dict(item["default_metrics"])
        self.best_u = self.u.copy()
        self.best_metrics = dict(self.metrics)
        self.step_idx = 0
        self.done = False

    def reset(self, init_u=None, init_metrics=None):
        if init_u is None:
            self.u = DEFAULT_U.copy()
            self.metrics = dict(self.item["default_metrics"])
        else:
            self.u = np.clip(np.asarray(init_u, dtype=np.float32), -1.0, 1.0)
            self.metrics = dict(init_metrics) if init_metrics is not None else evaluate_u(self.item, self.u)
        self.best_u = self.u.copy()
        self.best_metrics = dict(self.metrics)
        self.step_idx = 0
        self.done = False
        return self.observe()

    def observe(self):
        return build_obs(
            self.item["geom_features"],
            self.u,
            self.metrics,
            self.step_idx,
            self.item["budget"],
            self.max_steps,
        )

    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        u_try = np.clip(self.u + self.step_scale * act, -1.0, 1.0).astype(np.float32)
        invalid = False
        try:
            m_try = evaluate_u(self.item, u_try)
            reward = float((m_try["score"] - self.metrics["score"]) * 10.0)
            self.u = u_try
            self.metrics = m_try
            if m_try["score"] > self.best_metrics["score"]:
                self.best_u = self.u.copy()
                self.best_metrics = dict(m_try)
        except Exception:
            invalid = True
            reward = -1.0

        self.step_idx += 1
        self.done = self.step_idx >= self.max_steps
        return self.observe(), reward, self.done, {"invalid_attempt": invalid, "score": self.metrics["score"]}
