"""Independent DiPrete-inspired geometric decomposition prototype.

This module exports no CFD mesh or old CFD training workflow. Local geometric
features/reward are explicitly inspired by the paper, not a reproduction claim.
"""
from .cases import generate_cases
from .env import (ACTION_VERSION, ENVIRONMENT_VERSION, OBSERVATION_VERSION,
                  DecompositionConfig, OrthogonalBlockEnv)

__all__ = ["DecompositionConfig", "OrthogonalBlockEnv", "generate_cases",
           "ENVIRONMENT_VERSION", "ACTION_VERSION", "OBSERVATION_VERSION"]
