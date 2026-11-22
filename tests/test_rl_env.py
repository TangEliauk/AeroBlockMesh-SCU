import unittest

import numpy as np

from smartcfd.geometry.airfoil import naca
from smartcfd.meshing.structured import BASE_PARAMS, PARAM_DIM
from smartcfd.rl.env import HORIZON, OBSERVATION_DIM, mesh_metrics, observation, run_case, transition
from smartcfd.pipeline.workflow import load_sac_policy
from smartcfd.rl.trainer import dataset


class NacaEnvironmentTests(unittest.TestCase):
    def test_benchmark_splits_are_disjoint(self):
        splits = [{case.name for case in cases} for cases in dataset()]
        self.assertEqual([len(split) for split in splits], [144, 16, 24])
        self.assertEqual(len(set.union(*splits)), 184)

    def test_observation_shape_and_action_limits(self):
        case = naca("2412")
        observed = observation(case, BASE_PARAMS, mesh_metrics(case, BASE_PARAMS), 0)
        self.assertEqual(observed.shape, (OBSERVATION_DIM,))
        self.assertEqual(observed.dtype, np.float32)
        np.testing.assert_array_equal(transition(np.ones(PARAM_DIM), np.ones(PARAM_DIM)),
                                      np.ones(PARAM_DIM))

    def test_packaged_policy_repairs_the_invalid_reference_case(self):
        case = naca("6409")
        self.assertEqual(run_case(case, "fixed")["metrics"]["invalid_cells"], 1)
        selected = run_case(case, "sac", load_sac_policy())
        self.assertEqual(selected["evaluations"], HORIZON + 1)
        self.assertEqual(selected["metrics"]["invalid_cells"], 0)
