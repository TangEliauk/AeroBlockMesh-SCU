"""Serialization for conforming structured mesh coordinates."""

from pathlib import Path

import numpy as np


def write_plot3d(path: Path, mesh: np.ndarray) -> tuple[int, int, int]:
    """Write a formatted multizone XYZ file with no IBLANK records.

    PLOT3D reads each coordinate array in i-fastest order. The exported
    circumferential i direction is counterclockwise, so we reverse the radial
    j direction to keep a positive i-j orientation. All nodes have z=0.
    """
    nj, columns, dimensions = mesh.shape
    if dimensions != 2 or nj < 2 or columns < 5 or (columns - 1) % 4:
        raise ValueError("Expected a two-dimensional grid with four equally indexed sections")
    if not np.array_equal(mesh[:, 0], mesh[:, -1]):
        raise ValueError("The first and last section nodes must match")

    cells_per_block = (columns - 1) // 4
    ni, nk = cells_per_block + 1, 1
    zeros = np.zeros(ni * nj)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("4\n")
        for _ in range(4):
            stream.write(f"{ni} {nj} {nk}\n")
        for block in range(4):
            first = block * cells_per_block
            nodes = mesh[::-1, first : first + ni]
            for component in (nodes[..., 0].ravel(), nodes[..., 1].ravel(), zeros):
                np.savetxt(stream, component, fmt="%.17e")
    return ni, nj, nk
