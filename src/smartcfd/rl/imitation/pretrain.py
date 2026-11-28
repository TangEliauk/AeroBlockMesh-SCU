"""Behavior cloning on the same encoder/head later used by discrete SAC."""
from __future__ import annotations

import copy
import random
from typing import Sequence
import torch
from smartcfd.core.types import Transition
from smartcfd.rl.agents.policy import batch_observations


def _loss(actor, records, device):
    observations = batch_observations([record.observation for record in records], device)
    actions = torch.tensor([record.action for record in records], dtype=torch.int64, device=device)
    if bool(((actions < 0) | (actions >= observations["mask"].shape[1])).any()):
        raise ValueError("BC action index outside the action space")
    if not bool(observations["mask"].gather(1, actions[:, None]).all()):
        raise ValueError("BC action is illegal in its replayed state")
    probabilities, log_probabilities = actor.distribution(observations)
    nll = -log_probabilities.gather(1, actions[:, None]).squeeze(1)
    correct = (probabilities.argmax(-1) == actions).float().sum()
    return nll.mean(), correct


def evaluate_bc(actor, records, batch_size=64, device=None):
    if not records:
        raise ValueError("BC validation records are empty")
    device = device or next(actor.parameters()).device
    actor.eval()
    total_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            loss, hits = _loss(actor, batch, device)
            total_loss += float(loss) * len(batch)
            correct += float(hits)
    return {"loss": total_loss / len(records), "accuracy": correct / len(records)}


def train_behavior_cloning(actor, train_transitions: Sequence[Transition],
                           val_transitions: Sequence[Transition], epochs=20, patience=5,
                           batch_size=64, lr=3e-4, grad_clip=1.0, seed=42,
                           device=None) -> dict:
    if not train_transitions or not val_transitions:
        raise ValueError("BC needs separate nonempty training and validation demonstrations")
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    device = device or next(actor.parameters()).device
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    rng = random.Random(seed)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    for epoch in range(1, epochs + 1):
        actor.train()
        order = list(range(len(train_transitions)))
        rng.shuffle(order)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train_transitions[i] for i in order[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(actor, batch, device)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite BC loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        validation = evaluate_bc(actor, val_transitions, batch_size, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_transitions),
                        "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]})
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(actor.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    actor.load_state_dict(best_state)
    actor.eval()
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_completed": len(history), "early_stopped": len(history) < epochs,
            "seed": seed, "training_transitions": len(train_transitions),
            "validation_transitions": len(val_transitions), "restored_best_validation_actor": True}
