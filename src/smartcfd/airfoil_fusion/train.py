"""Policy training, validation and exact same-environment continuation."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import random
import time

import numpy as np
import torch

from smartcfd.airfoil_decomposition.experiment import file_sha, network_hashes, write_json
from smartcfd.airfoil_decomposition.learning import AirfoilSACAgent
from smartcfd.core.types import Transition
from smartcfd.rl.replay_buffer import ReplayBuffer

from .learning import FusionSAC, bc_update
from .protocol import read_json
from .rollout import Environments, episode, replay, selection_key, summarize

NETWORKS = ("actor", "q1", "q2", "target_q1", "target_q2")


def save_torch(path, value):
    staging = path.with_suffix(path.suffix+".pending")
    with staging.open("wb") as stream:
        torch.save(value, stream)
    staging.replace(path)


def make_agent(arm, seed):
    if arm == "E4":
        return AirfoilSACAgent(seed=seed)
    return FusionSAC(use_geometry=arm != "E4a", seed=seed)


def _weights(agent, binding, step):
    log_alpha = getattr(agent, "log_alpha", None)
    return {"binding": binding, "step": step, "networks": {name: getattr(agent, name).state_dict() for name in NETWORKS},
            "log_alpha": None if log_alpha is None else log_alpha.detach().cpu()}


def _restore_weights(agent, value):
    for name in NETWORKS:
        getattr(agent, name).load_state_dict(value["networks"][name], strict=True)
    if getattr(agent, "log_alpha", None) is not None:
        with torch.no_grad():
            agent.log_alpha.copy_(value["log_alpha"])


def get_inference_agent(path, arm, seed, prepared_hash):
    value = torch.load(path, map_location="cpu", weights_only=True)
    binding = value["binding"]
    if (binding["arm"], binding["seed"], binding["prepared_sha256"]) != (arm, seed, prepared_hash):
        raise ValueError(f"Inference checkpoint identity mismatch: {path.name}")
    agent = make_agent(arm, seed)
    _restore_weights(agent, value)
    return agent, value["step"]


def state_hash(observation):
    h = hashlib.sha256()
    for key, value in sorted(observation.items()):
        value = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def _log(stream, row):
    stream.write((json.dumps(row, allow_nan=False, ensure_ascii=True, separators=(",", ":"))+"\n").encode())


def _validate(directory, cfg, cases, checked, arm, seed, agent, binding, step):
    directory.mkdir(exist_ok=True)
    pool = Environments(cfg["geometry"], arm)
    check = {r["sample_id"]: r for r in checked}
    hashes = network_hashes(agent)
    rows = [episode(pool, sample, "learned", agent=agent, seed=seed, prechecked=check[sample["sample_id"]])[0]
            for sample in cases]
    if hashes != network_hashes(agent):
        raise ValueError("Validation changed policy tensors")
    report = {"step": step, **summarize(rows), "network_sha256": hashes}
    write_json(directory/"report.json", {**report, "per_case": rows}, replace=True)
    save_torch(directory/"weights.pt", _weights(agent, binding, step))
    print(f"{arm} seed={seed} validation step={step} success={report['successful']}/{report['total']}", flush=True)
    return report


def _binding(output, arm, seed, teacher_path=None):
    value = {"arm": arm, "seed": seed, "prepared_sha256": file_sha(output/"prepared.json")}
    if teacher_path is not None:
        value["teachers_sha256"] = file_sha(teacher_path)
    return value


def pretrain(output, cfg, cases, checked, groups, seed):
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    arm = "E5"
    destination = output/"arms"/f"{arm}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    binding = _binding(output, arm, seed, output/"teachers/manifest.json")
    if (destination/"completion.json").exists():
        completed = read_json(destination/"completion.json")
        if completed["binding"] != binding:
            raise ValueError("BC checkpoint does not match the frozen run")
        _verify_outputs(destination, completed)
        return completed
    agent, rng = make_agent(arm, seed), random.Random(seed)
    families = sorted(groups)
    with (destination/"updates.jsonl").open("wb") as log:
        for step in range(1, cfg["bc_updates"]+1):
            batch = []
            for _ in range(cfg["runtime"]["batch_size"]):
                sample = rng.choice(rng.choice(groups[rng.choice(families)]))
                batch.append((sample.observation, sample.action))
            _log(log, {"update": step, **bc_update(agent, batch)})
    val = _validate(destination/"validation_000000", cfg, cases["val"], checked["val"], arm, seed, agent, binding, 0)
    for tag in ("best", "final"):
        save_torch(destination/f"{tag}.pt", _weights(agent, binding, 0))
    return _complete(destination, binding, [val], online_steps=0, updates=cfg["bc_updates"])


def _verify_outputs(directory, receipt):
    for name, expected in receipt["checkpoints"].items():
        if file_sha(directory/name) != expected:
            raise ValueError(f"Checkpoint {name} does not match receipt in {directory.name}")


def _complete(directory, binding, history, *, online_steps, updates):
    best = max(history, key=selection_key)
    result = {"binding": binding, "online_steps": online_steps, "updates": updates,
              "validation": history, "best_step": best["step"], "final_step": history[-1]["step"],
              "checkpoints": {f"{tag}.pt": file_sha(directory/f"{tag}.pt") for tag in ("best", "final")}}
    write_json(directory/"completion.json", result, replace=True)
    return result


def _restore_training(agent, value, buffer, rng):
    _restore_weights(agent, value["weights"])
    agent.actor_optimizer.load_state_dict(value["actor_optimizer"])
    agent.q_optimizer.load_state_dict(value["critic_optimizer"])
    if getattr(agent, "alpha_optimizer", None) is not None:
        agent.alpha_optimizer.load_state_dict(value["alpha_optimizer"])
    agent.action_rng.set_state(value["action_rng"])
    torch.set_rng_state(value["torch_rng"])
    agent.updates, agent.actor_updates = value["updates"], value["actor_updates"]
    buffer.load_state_dict(value["replay"])
    rng.setstate(value["rng"])


def _checkpoint(directory, agent, buffer, rng, binding, step, history, cursor, observation, logs):
    offsets = {}
    for name, stream in logs.items():
        stream.flush()
        offsets[name] = stream.tell()
    alpha = getattr(agent, "alpha_optimizer", None)
    value = {"binding": binding, "step": step, "weights": _weights(agent, binding, step),
             "actor_optimizer": agent.actor_optimizer.state_dict(), "critic_optimizer": agent.q_optimizer.state_dict(),
             "alpha_optimizer": None if alpha is None else alpha.state_dict(),
             "action_rng": agent.action_rng.get_state(), "torch_rng": torch.get_rng_state(), "rng": rng.getstate(),
             "updates": agent.updates, "actor_updates": agent.actor_updates, "replay": buffer.state_dict(),
             "cursor": cursor, "observation_sha256": None if observation is None else state_hash(observation),
             "log_offsets": offsets, "history": history}
    save_torch(directory/"resume.pt", value)


def _guard_record(result):
    if result is None:
        return None
    return {"metrics": result["metrics"], "actions": [
        {**row, "kept": bool(result["effective_mask"][row["action"]])} for row in result["actions"]]}


def train(output, cfg, cases, checked, arm, seed, steps, groups=None):
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    allowed = (cfg["first_steps"], cfg["extended_steps"])
    if steps not in allowed:
        raise ValueError(f"SAC budget must be one of the frozen values {allowed}")
    directory = output/"arms"/f"{arm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    from_bc = arm in ("E7", "E8")
    binding = _binding(output, arm, seed, output/"teachers/manifest.json" if from_bc else None)
    bc_path = output/"arms"/f"E5_seed{seed}"/"final.pt"
    if from_bc:
        binding["bc_sha256"] = file_sha(bc_path)
    receipt_path = directory/"completion.json"
    if receipt_path.exists():
        completed = read_json(receipt_path)
        if completed["binding"] != binding:
            raise ValueError("SAC completion belongs to another study")
        if completed["online_steps"] >= steps:
            _verify_outputs(directory, completed)
            return completed
    running = directory/"resume.pt"
    agent, rng = make_agent(arm, seed), random.Random(seed)
    rt = cfg["runtime"]
    buffer = ReplayBuffer(rt["replay_capacity"], rt["replay_max_bytes"], seed)
    pools = Environments(cfg["geometry"], arm)
    excluded = {row["sample_id"]: row["reason"] for row in checked["train"] if not row["valid"]}
    samples = {row["sample_id"]: row for row in cases["train"] if row["sample_id"] not in excluded}
    if not samples or any(s["split"] != "train" for s in samples.values()):
        raise ValueError("Online SAC requires initializable train-only samples")
    cursor = {"order": list(samples), "position": len(samples), "episode": 0, "sample_id": None,
              "prefix": [], "return": 0.}
    history, step, env, obs = [], 0, None, None
    saved = None
    if running.exists():
        saved = torch.load(running, map_location="cpu", weights_only=False)
        if saved["binding"] != binding or saved["step"] > steps:
            raise ValueError("Resume checkpoint identity or training horizon mismatch")
        _restore_training(agent, saved, buffer, rng)
        cursor, step, history = saved["cursor"], saved["step"], saved["history"]
        if cursor["sample_id"] is not None:
            env, obs, _ = replay(pools, samples[cursor["sample_id"]], cursor["prefix"],
                                 seed=seed+cursor["episode"], confirm=False)
            if env.terminated or state_hash(obs) != saved["observation_sha256"]:
                raise ValueError("Resumed mid-episode observation changed")
    elif from_bc:
        actor, _ = get_inference_agent(bc_path, "E5", seed, binding["prepared_sha256"])
        agent.load_actor(actor.actor.state_dict())
        if network_hashes(actor)["actor"] != network_hashes(agent)["actor"]:
            raise ValueError("BC actor transfer changed network values")
        for family in sorted(groups):
            for trajectory in groups[family]:
                for transition in trajectory:
                    buffer.add(transition)
        if len(buffer) < rt["batch_size"]:
            raise RuntimeError("Replay contains fewer than one batch of verified BC transitions")
        for _ in range(cfg["critic_warmup"]):
            agent.update_critic(buffer.sample(rt["batch_size"]))
        if network_hashes(actor)["actor"] != network_hashes(agent)["actor"]:
            raise ValueError("Critic warmup changed the BC actor")
        write_json(directory/"warmup.json", {"bc_weights": binding["bc_sha256"], "actor": network_hashes(agent)["actor"],
                   "critic_updates": cfg["critic_warmup"], "demonstration_transitions": buffer.total_added}, replace=True)
    if not history:
        history.append(_validate(directory/"validation_000000", cfg, cases["val"], checked["val"], arm, seed, agent, binding, 0))

    with ExitStack() as stack:
        logs = {}
        for name in ("updates", "interactions", "episodes"):
            path = directory/f"{name}.jsonl"
            stream = stack.enter_context(path.open("r+b" if saved is not None and path.exists() else "w+b"))
            if saved is not None:
                if path.stat().st_size < saved["log_offsets"][name]:
                    raise ValueError(f"Truncated {name} log before resume marker")
                stream.seek(saved["log_offsets"][name])
                stream.truncate()
            logs[name] = stream
        if saved is None:
            for sample_id, reason in excluded.items():
                _log(logs["episodes"], {"sample_id": sample_id, "phase": "preflight", "success": False, "reason": reason})
            _checkpoint(directory, agent, buffer, rng, binding, step, history, cursor, obs, logs)
        while step < steps:
            if (output/"STOP_REQUESTED").exists():
                _checkpoint(directory, agent, buffer, rng, binding, step, history, cursor, obs, logs)
                raise InterruptedError("Local study was stopped at a durable checkpoint")
            if obs is None:
                for _ in range(len(samples)):
                    if cursor["position"] >= len(samples):
                        rng.shuffle(cursor["order"])
                        cursor["position"] = 0
                    sid = cursor["order"][cursor["position"]]
                    cursor["position"] += 1
                    cursor["episode"] += 1
                    cursor.update(sample_id=sid, prefix=[], **{"return": 0.})
                    env = pools.get(samples[sid])
                    obs, info = env.reset(seed=seed+cursor["episode"])
                    if not env.terminated:
                        break
                    _log(logs["episodes"], {"sample_id": sid, "episode": cursor["episode"], "decisions": 0,
                                             "success": False, "reason": info["reason"]})
                if env.terminated:
                    raise RuntimeError("No training case admits a legal online action")
            step += 1
            started = time.perf_counter()
            legal = np.flatnonzero(obs["mask"]).tolist()
            action = rng.choice(legal) if not from_bc and step <= cfg["online_warmup"] else agent.act(obs)
            guard = _guard_record(getattr(env, "_guard_report", None))
            descriptor = env.candidates[action]
            after, reward, done, truncated, info = env.step(action)
            buffer.add(Transition(obs, action, reward, after, done, truncated))
            cursor["prefix"].append(descriptor)
            cursor["return"] += reward
            _log(logs["interactions"], {"step": step, "episode": cursor["episode"], "sample_id": cursor["sample_id"],
                                       "observation": state_hash(obs), "next_observation": state_hash(after),
                                       "candidate": descriptor, "effective_actions": legal, "guard": guard,
                                       "reward": reward, "reward_components": info.get("reward_components"),
                                       "terminated": done, "truncated": truncated, "reason": info["reason"],
                                       "seconds": time.perf_counter()-started})
            if (from_bc or step >= cfg["online_warmup"]) and len(buffer) >= rt["batch_size"]:
                _log(logs["updates"], {"step": step, **agent.update(buffer.sample(rt["batch_size"]))})
            obs = after
            if done or truncated:
                _log(logs["episodes"], {"episode": cursor["episode"], "sample_id": cursor["sample_id"],
                                         "decisions": len(cursor["prefix"]), "return": cursor["return"],
                                         "success": info["success"], "reason": info["reason"]})
                obs = None
                cursor.update(sample_id=None, prefix=[], **{"return": 0.})
            if step % cfg["validation_every"] == 0 or step == steps:
                history.append(_validate(directory/f"validation_{step:06d}", cfg, cases["val"], checked["val"],
                                         arm, seed, agent, binding, step))
            if step % rt["checkpoint_every"] == 0 or step == steps:
                _checkpoint(directory, agent, buffer, rng, binding, step, history, cursor, obs, logs)
                print(f"{arm} seed={seed} checkpoint {step}/{steps} | updates={agent.updates}", flush=True)
    best = max(history, key=selection_key)
    for tag, index in (("best", best["step"]), ("final", step)):
        original = directory/f"validation_{index:06d}"/"weights.pt"
        save_torch(directory/f"{tag}.pt", torch.load(original, map_location="cpu", weights_only=True))
    return _complete(directory, binding, history, online_steps=step, updates=agent.updates)
