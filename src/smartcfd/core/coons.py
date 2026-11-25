"""Evaluate a Coons patch or selected strips from four sampled oriented edges."""

import numpy as np


def fractions(xy):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return arc / arc[-1]


def coons(bottom, right, top, left, *, columns=slice(None), rows=slice(None)):
    """Bottom/top run left to right; left/right run bottom to top. Slices keep both axes."""
    u = ((fractions(bottom) + fractions(top))[columns] / 2)[None, :, None]
    v = ((fractions(left) + fractions(right))[rows] / 2)[:, None, None]
    ruled = (1-v)*bottom[columns] + v*top[columns]
    left_linear = (1-v)*left[0] + v*left[-1]
    right_linear = (1-v)*right[0] + v*right[-1]
    return ruled + (1-u)*(left[rows, None]-left_linear) + u*(right[rows, None]-right_linear)
