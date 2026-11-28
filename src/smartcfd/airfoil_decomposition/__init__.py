"""Airfoil C-mapping-constrained decomposition; not free CAD topology."""
from .config import (AirfoilDecompositionConfig, ENVIRONMENT_VERSION,
                     ACTION_VERSION, OBSERVATION_VERSION, PARTITION_VERSION)
from .env import AirfoilBlockEnv

__all__ = ['AirfoilDecompositionConfig', 'ENVIRONMENT_VERSION', 'ACTION_VERSION',
           'OBSERVATION_VERSION', 'PARTITION_VERSION', 'AirfoilBlockEnv']
