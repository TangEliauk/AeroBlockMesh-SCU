"""Shared geometry case types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    m: float
    p: float
    t: float
