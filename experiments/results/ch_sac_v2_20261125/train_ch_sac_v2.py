"""Step 4: Multi-seed GPU training of CH-SAC v2 with regular fixed Epochs and live validation."""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

if __name__ == "__main__" and "--torch-root" in sys.argv:
    from importlib.machinery import PathFinder

    class TorchPackages:
        root = str(Path(sys.argv[sys.argv.index("--torch-root") + 1]).resolve())

        def find_spec(self, fullname, path=None, target=None):
            if path is None and fullname in {"torch", "torchgen", "functorch"}:
                return PathFinder.find_spec(fullname, [self.root])

    sys.meta_path.insert(0, TorchPackages())

OUT = Path(__file__).resolve().parent
if str(OUT) not in sys.path:
    sys.path.insert(0, str(OUT))

from ch_env import ACTION_DIM, DEFAULT_U, MAX_STEPS, STEP_SCALE, build_obs, evaluate_u, load_cached_split

_WORKER_SPLITS = {}


def _init_worker(train_cache_path, val_cache_path):
    global _WORKER_SPLITS
    _WORKER_SPLITS["train"] = load_cached_split(train_cache_path)
    _WORKER_SPLITS["val"] = load_cached_split(val_cache_path)


def _eval_worker(job):
    split_name, idx, u = job
    it = _WORKER_SPLITS[split_name][idx]
    try:
        m = evaluate_u(it, u)
        return True, m
    except Exception:
        return False, None


def _run_episodes_parallel(pool, split_name, items, indices, actor_fn, step_scale=STEP_SCALE, max_steps=MAX_STEPS):
    """Run vectorized 5-step episodes using main-process GPU inference + parallel CPU TFI workers."""
    n = len(indices)
    u_cur = np.repeat(DEFAULT_U[None, :], n, axis=0).astype(np.float32)
    m_cur = [dict(items[idx]["default_metrics"]) for idx in indices]
    ep_returns = np.zeros(n, dtype=np.float32)
    invalid_count = 0
    transitions = []

    obs_cur = np.stack(
        [
            build_obs(items[idx]["geom_features"], u_cur[k], m_cur[k], 0, items[idx]["budget"], max_steps)
            for k, idx in enumerate(indices)
        ],
        axis=0,
    )

    for step_idx in range(max_steps):
        actions = np.clip(actor_fn(obs_cur), -1.0, 1.0).astype(np.float32)
        u_try = np.clip(u_cur + step_scale * actions, -1.0, 1.0).astype(np.float32)
        jobs = [(split_name, int(indices[k]), u_try[k]) for k in range(n)]
        results = list(pool.map(_eval_worker, jobs, chunksize=max(1, n // 16)))

        obs_nxt = np.empty_like(obs_cur)
        rew = np.zeros(n, dtype=np.float32)
        done = step_idx + 1 >= max_steps

        for k, (ok, m_try) in enumerate(results):
            idx = int(indices[k])
            if ok:
                r = float((m_try["score"] - m_cur[k]["score"]) * 10.0)
                u_cur[k] = u_try[k]
                m_cur[k] = m_try
            else:
                r = -1.0
                invalid_count += 1
            ep_returns[k] += r
            rew[k] = r
            obs_nxt[k] = build_obs(
                items[idx]["geom_features"], u_cur[k], m_cur[k], step_idx + 1, items[idx]["budget"], max_steps
            )

        transitions.append((obs_cur, actions, rew, obs_nxt, np.full(n, done, dtype=bool)))
        obs_cur = obs_nxt

    return transitions, ep_returns, m_cur, invalid_count


def _summarize_val(items, metrics_list):
    scores_1536 = [m["score"] for it, m in zip(items, metrics_list) if it["budget"] == 1536]
    scores_5120 = [m["score"] for it, m in zip(items, metrics_list) if it["budget"] == 5120]
    return {
        "val_score": float(np.mean([m["score"] for m in metrics_list])),
        "val_score_1536": float(np.mean(scores_1536)),
        "val_score_5120": float(np.mean(scores_5120)),
        "val_jmin": float(np.mean([m["min_scaled_jacobian"] for m in metrics_list])),
        "val_j05": float(np.mean([m["q05_scaled_jacobian"] for m in metrics_list])),
        "val_wall_q10": float(np.mean([m["wall_angle_q10_deg"] for m in metrics_list])),
        "val_far_ar": float(np.mean([m["farfield_max_aspect_ratio"] for m in metrics_list])),
    }


def _train_one_seed(seed, args, train_items, val_items, pool):
    import torch
    from ch_sac_v2_model import CHSACAgent, ReplayBuffer

    seed_dir = OUT / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    hist_path = seed_dir / "history.csv"
    last_path = seed_dir / "last.pt"
    best_path = seed_dir / "best.pt"

    expected_rows = 1 + args.epochs // args.val_every
    if args.resume and last_path.is_file() and hist_path.is_file():
        rows = list(csv.DictReader(hist_path.open("r", encoding="utf-8")))
        if len(rows) == expected_rows:
            print(f"[seed {seed}] already complete ({len(rows)} rows), skipping.", flush=True)
            best_row = max(rows, key=lambda r: float(r["val_score"]))
            return {"seed": seed, "resumed": True, "best_val_score": float(best_row["val_score"]), "best_epoch": int(best_row["epoch"])}

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    episodes_per_epoch = args.episodes_per_epoch
    steps_per_epoch = episodes_per_epoch * MAX_STEPS
    total_updates = args.epochs * steps_per_epoch

    agent = CHSACAgent(device=args.device, lr=args.lr, gamma=args.gamma, tau=args.tau, total_updates=total_updates)
    replay = ReplayBuffer(capacity=args.buffer_capacity)

    idx_1536 = np.array([i for i, it in enumerate(train_items) if it["budget"] == 1536], dtype=int)
    idx_5120 = np.array([i for i, it in enumerate(train_items) if it["budget"] == 5120], dtype=int)
    val_indices = np.arange(len(val_items), dtype=int)

    fieldnames = [
        "epoch",
        "env_steps",
        "train_ep_return",
        "train_invalid",
        "val_score",
        "val_score_1536",
        "val_score_5120",
        "val_jmin",
        "val_j05",
        "val_wall_q10",
        "val_far_ar",
        "critic_loss",
        "actor_loss",
        "alpha",
        "q_mean",
        "epoch_sec",
    ]
    csv_fh = hist_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()
    csv_fh.flush()

    # Epoch 0: Random warmup episodes
    t0 = perf_counter()
    warm_1536 = rng.choice(idx_1536, size=args.warmup_episodes // 2, replace=True)
    warm_5120 = rng.choice(idx_5120, size=args.warmup_episodes - len(warm_1536), replace=True)
    warm_idx = np.concatenate([warm_1536, warm_5120])
    rng.shuffle(warm_idx)
    warm_trans, warm_rets, _, warm_inv = _run_episodes_parallel(
        pool,
        "train",
        train_items,
        warm_idx,
        lambda obs: rng.uniform(-1.0, 1.0, size=(len(obs), ACTION_DIM)).astype(np.float32),
    )
    for obs_b, act_b, rew_b, nxt_b, done_b in warm_trans:
        replay.add_batch(obs_b, act_b, rew_b, nxt_b, done_b)

    # Initial Epoch 0 validation
    _, _, val_m, _ = _run_episodes_parallel(
        pool, "val", val_items, val_indices, lambda obs: agent.act(obs, deterministic=True)
    )
    vs = _summarize_val(val_items, val_m)
    best_val_score = vs["val_score"]
    best_epoch = 0
    torch.save(agent.state_dict(), best_path)

    row0 = {
        "epoch": 0,
        "env_steps": 0,
        "train_ep_return": round(float(np.mean(warm_rets)), 5),
        "train_invalid": int(warm_inv),
        "val_score": round(vs["val_score"], 5),
        "val_score_1536": round(vs["val_score_1536"], 5),
        "val_score_5120": round(vs["val_score_5120"], 5),
        "val_jmin": round(vs["val_jmin"], 5),
        "val_j05": round(vs["val_j05"], 5),
        "val_wall_q10": round(vs["val_wall_q10"], 3),
        "val_far_ar": round(vs["val_far_ar"], 2),
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "alpha": round(float(agent.alpha.item()), 5),
        "q_mean": 0.0,
        "epoch_sec": round(perf_counter() - t0, 2),
    }
    writer.writerow(row0)
    csv_fh.flush()
    print(
        f"[seed {seed}] Epoch  0/{args.epochs} | steps=    0 | val_score={vs['val_score']:.4f} "
        f"(1536={vs['val_score_1536']:.4f}, 5120={vs['val_score_5120']:.4f}) Jmin={vs['val_jmin']:.4f} "
        f"farAR={vs['val_far_ar']:.1f} ({row0['epoch_sec']}s)",
        flush=True,
    )

    env_steps = 0
    for epoch in range(1, args.epochs + 1):
        ep_t0 = perf_counter()
        half = episodes_per_epoch // 2
        ep_idx = np.concatenate(
            [
                rng.choice(idx_1536, size=half, replace=True),
                rng.choice(idx_5120, size=episodes_per_epoch - half, replace=True),
            ]
        )
        rng.shuffle(ep_idx)

        trans, ep_rets, _, inv_cnt = _run_episodes_parallel(
            pool, "train", train_items, ep_idx, lambda obs: agent.act(obs, deterministic=False)
        )
        for obs_b, act_b, rew_b, nxt_b, done_b in trans:
            replay.add_batch(obs_b, act_b, rew_b, nxt_b, done_b)
        env_steps += steps_per_epoch

        losses = {"critic_loss": [], "actor_loss": [], "alpha": [], "q_mean": []}
        for _ in range(steps_per_epoch):
            batch = replay.sample(args.batch_size, rng, agent.device)
            stats = agent.update(*batch)
            for k, v in stats.items():
                if not np.isfinite(v):
                    raise FloatingPointError(f"Non-finite {k}={v} at seed {seed}, epoch {epoch}")
                losses[k].append(v)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            _, _, val_m, _ = _run_episodes_parallel(
                pool, "val", val_items, val_indices, lambda obs: agent.act(obs, deterministic=True)
            )
            vs = _summarize_val(val_items, val_m)
            if vs["val_score"] > best_val_score:
                best_val_score = vs["val_score"]
                best_epoch = epoch
                torch.save(agent.state_dict(), best_path)

            row = {
                "epoch": epoch,
                "env_steps": env_steps,
                "train_ep_return": round(float(np.mean(ep_rets)), 5),
                "train_invalid": int(inv_cnt),
                "val_score": round(vs["val_score"], 5),
                "val_score_1536": round(vs["val_score_1536"], 5),
                "val_score_5120": round(vs["val_score_5120"], 5),
                "val_jmin": round(vs["val_jmin"], 5),
                "val_j05": round(vs["val_j05"], 5),
                "val_wall_q10": round(vs["val_wall_q10"], 3),
                "val_far_ar": round(vs["val_far_ar"], 2),
                "critic_loss": round(float(np.mean(losses["critic_loss"])), 5),
                "actor_loss": round(float(np.mean(losses["actor_loss"])), 5),
                "alpha": round(float(np.mean(losses["alpha"])), 5),
                "q_mean": round(float(np.mean(losses["q_mean"])), 5),
                "epoch_sec": round(perf_counter() - ep_t0, 2),
            }
            writer.writerow(row)
            csv_fh.flush()
            print(
                f"[seed {seed}] Epoch {epoch:2d}/{args.epochs} | steps={env_steps:5d} | val_score={vs['val_score']:.4f} "
                f"(best={best_val_score:.4f}@{best_epoch}) Jmin={vs['val_jmin']:.4f} farAR={vs['val_far_ar']:.1f} "
                f"qLoss={row['critic_loss']:.4f} alpha={row['alpha']:.3f} ({row['epoch_sec']}s)",
                flush=True,
            )

    csv_fh.close()
    torch.save(agent.state_dict(), last_path)
    return {
        "seed": seed,
        "resumed": False,
        "best_val_score": float(best_val_score),
        "best_epoch": int(best_epoch),
        "final_val_score": float(vs["val_score"]),
        "total_sec": round(perf_counter() - t0, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--episodes-per-epoch", type=int, default=80)
    parser.add_argument("--warmup-episodes", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dev_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU"
    print(f"CH-SAC v2 Trainer | torch={torch.__version__} | device={args.device} ({dev_name})", flush=True)

    train_cache = str(OUT / "cache/train_g0.pkl.gz")
    val_cache = str(OUT / "cache/val_g0.pkl.gz")
    train_items = load_cached_split(train_cache)
    val_items = load_cached_split(val_cache)

    t0 = perf_counter()
    seed_summaries = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(train_cache, val_cache)
    ) as pool:
        for seed in args.seeds:
            seed_summaries.append(_train_one_seed(int(seed), args, train_items, val_items, pool))

    # Build/refresh aggregate training_summary.json across all available finished seeds
    all_seed_rows = []
    for s in (42, 43, 44, 45, 46):
        hp = OUT / "seeds" / f"seed_{s}" / "history.csv"
        if hp.is_file():
            rows = list(csv.DictReader(hp.open("r", encoding="utf-8")))
            if rows:
                best_row = max(rows, key=lambda r: float(r["val_score"]))
                all_seed_rows.append(
                    {
                        "seed": s,
                        "best_epoch": int(best_row["epoch"]),
                        "best_val_score": float(best_row["val_score"]),
                        "best_val_jmin": float(best_row["val_jmin"]),
                        "best_val_j05": float(best_row["val_j05"]),
                        "best_val_far_ar": float(best_row["val_far_ar"]),
                        "final_val_score": float(rows[-1]["val_score"]),
                        "rows": len(rows),
                    }
                )
    if all_seed_rows:
        sorted_by_best = sorted(all_seed_rows, key=lambda r: r["best_val_score"])
        median_seed = sorted_by_best[len(sorted_by_best) // 2]["seed"]
    else:
        median_seed = None

    summary = {
        "torch_version": torch.__version__,
        "device": dev_name,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "steps_per_epoch": args.episodes_per_epoch * MAX_STEPS,
        "total_online_steps": args.epochs * args.episodes_per_epoch * MAX_STEPS,
        "warmup_transitions": args.warmup_episodes * MAX_STEPS,
        "seeds_run_now": seed_summaries,
        "all_seeds_summary": all_seed_rows,
        "median_val_seed": median_seed,
        "wall_sec": round(perf_counter() - t0, 2),
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
