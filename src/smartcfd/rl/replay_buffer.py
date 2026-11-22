"""Bounded replay storage for continuous NACA mesh transitions."""

import numpy as np


class Replay:
    def __init__(self, capacity, observations, actions):
        self.offset = 0
        self.size = 0
        self.capacity = capacity
        self.arrays = tuple(
            np.empty((capacity, dimensions), dtype=np.float32)
            for dimensions in (observations, actions, 1, observations, 1)
        )

    def add(self, *transition):
        for array, value in zip(self.arrays, transition):
            array[self.offset] = value
        self.offset = (self.offset + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch(self, rng, count):
        indices = rng.integers(self.size, size=count)
        return tuple(array[indices] for array in self.arrays)
