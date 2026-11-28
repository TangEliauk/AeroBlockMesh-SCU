"""Bounded CPU replay with independent snapshots and retained-byte accounting."""
from __future__ import annotations

from collections import deque
from dataclasses import is_dataclass
import copy
import random
import sys
import torch
from smartcfd.core.types import Transition


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if isinstance(value, Transition):
        return Transition(snapshot(value.observation), int(value.action), float(value.reward),
                          snapshot(value.next_observation), bool(value.terminated), bool(value.truncated))
    return copy.deepcopy(value)


def retained_bytes(value, visited=None):
    """Headers and owned buffers, with tensor storage counted explicitly.

    Owned numpy data is included by getsizeof. Callers add a 10% cushion and
    per-record allowance. This caps retained replay storage, not peak process RSS.
    """
    visited = set() if visited is None else visited
    if id(value) in visited:
        return 0
    visited.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, torch.Tensor):
        return size + value.nelement() * value.element_size()
    if isinstance(value, dict):
        return size + sum(retained_bytes(k, visited) + retained_bytes(v, visited) for k, v in value.items())
    if isinstance(value, (list, tuple, deque)):
        return size + sum(retained_bytes(v, visited) for v in value)
    if is_dataclass(value):
        return size + retained_bytes(vars(value), visited)
    return size


class ReplayBuffer:
    def __init__(self, capacity=20000, max_bytes=512 * 1024 ** 2, seed=42):
        if capacity <= 0 or max_bytes <= 0:
            raise ValueError("Replay capacity and memory cap must be positive")
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self._records, self._sizes = deque(), deque()
        self.allocated_bytes = 0
        self.rng = random.Random(seed)
        self.total_added = self.total_evicted = 0

    def __len__(self):
        return len(self._records)

    def add(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError("Replay stores core.types.Transition records")
        record = snapshot(transition)
        size = int(retained_bytes(record) * 1.1) + 256
        if size > self.max_bytes:
            raise ValueError("One transition exceeds the replay retained-memory cap")
        while self._records and (len(self) >= self.capacity or self.allocated_bytes + size > self.max_bytes):
            self._records.popleft()
            self.allocated_bytes -= self._sizes.popleft()
            self.total_evicted += 1
        self._records.append(record)
        self._sizes.append(size)
        self.allocated_bytes += size
        self.total_added += 1

    append = add

    def push(self, observation, action, reward, next_observation, terminated, truncated=False):
        self.add(Transition(observation, action, reward, next_observation, terminated, truncated))

    def sample(self, batch_size):
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Batch size must be positive and no larger than replay size")
        return [snapshot(self._records[i]) for i in self.rng.sample(range(len(self)), batch_size)]

    def state_dict(self):
        return {"capacity": self.capacity, "max_bytes": self.max_bytes,
                "records": [snapshot(t) for t in self._records], "rng": self.rng.getstate(),
                "total_added": self.total_added, "total_evicted": self.total_evicted}

    def load_state_dict(self, state):
        self.__init__(state["capacity"], state["max_bytes"])
        for record in state["records"]:
            self.add(record)
        self.rng.setstate(state["rng"])
        self.total_added = state["total_added"]
        self.total_evicted = state["total_evicted"]
