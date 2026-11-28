"""Bounded reuse of verified deterministic validation trajectories."""

from collections import OrderedDict
import pickle
import time
import zlib

from smartcfd.airfoil_fusion.rollout import Environments, episode


class ValidationTrajectories:
    def __init__(self, geometry, arm, byte_limit=32 * 1024 * 1024):
        self.pool = Environments(geometry, arm, capacity=1)
        self.byte_limit = byte_limit
        self.size = 0
        self.traces = OrderedDict()
        self.next_trace = 0

    def evaluate(self, sample, checked, policy, seed):
        start = time.perf_counter()
        sample_id = sample["sample_id"]
        for key in reversed(self.traces):
            if key[:2] != (sample_id, seed):
                continue
            outcome, decisions, _ = self.traces[key]
            if all(policy.act(pickle.loads(zlib.decompress(observation)), deterministic=True) == action
                   for observation, action in decisions):
                self.traces.move_to_end(key)
                # Work counters describe operations physically performed in this rollout.
                cached = {**outcome, "seconds": time.perf_counter() - start, "trajectory_reused": True}
                for field in ("work", "guard_work"):
                    if field in outcome:
                        cached[field] = dict.fromkeys(outcome[field], 0)
                return cached

        outcome, transitions = episode(self.pool, sample, "learned", agent=policy, seed=seed,
                                       prechecked=checked, capture=True)
        outcome["seconds"] = time.perf_counter() - start
        # Only store outcomes established by a finalized ordinary rollout. All
        # saved observations retain their original legal mask and action order.
        if transitions and (transitions[-1].terminated or transitions[-1].truncated):
            # These compressed observations are generated and read entirely within this process.
            decisions = [(zlib.compress(pickle.dumps(transition.observation, protocol=5), level=1),
                          transition.action) for transition in transitions]
            size = sum(len(observation) for observation, _ in decisions)
            if size <= self.byte_limit:
                while self.size + size > self.byte_limit:
                    _, (_, _, removed) = self.traces.popitem(last=False)
                    self.size -= removed
                key = (sample_id, seed, self.next_trace)
                self.next_trace += 1
                self.traces[key] = (outcome, decisions, size)
                self.size += size
        return outcome
