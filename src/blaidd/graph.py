'''
here there is stuff to build the graph
'''

from math import sqrt

import numpy as np
import numpy.typing as npt


def dist2d(p1: npt.ArrayLike, p2: npt.ArrayLike) -> float:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def seek_valid_edges(centroids: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    # calculating a dist matrix with cdist at this stage could help
    
    return [
        (i, j)
        for i in range(centroids.shape[0])
        for j in range(i, centroids.shape[0])
        if is_edge_valid(i, j, centroids)
    ]


def is_edge_valid(
    i: int,
    j: int,
    centroids: npt.NDArray[np.float64],
) -> bool:
    ray = dist2d(centroids[i], centroids[j]) / 2
    center = (centroids[i] + centroids[j]) / 2

    for z in range(centroids.shape[0]):
        if z == i or z == j:
            continue

        point_ray = dist2d(center, centroids[z])
        if point_ray < ray:
            return False


    return True