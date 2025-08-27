from math import sqrt

import numpy as np
import numpy.typing as npt

Point2D = npt.ArrayLike
Point3D = tuple[float, float, float]
Point = tuple[float, ...]

DistMatrix = list[list[float]]

def dist2d(p1: Point2D, p2: Point2D) -> float:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def dist3d(p1: Point3D, p2: Point3D) -> float:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def middle_point(p1: np.array, p2: np.array) -> np.array:
    return (p1 + p2) / 2


def circle_cut_edges(centroids: list[Point2D]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = {
        (i, j)
        for i in range(len(centroids))
        for j in range(i, len(centroids))
    }


    # cutting the edges by middle points
    

    return list(edges)


def centroids_dist_matrix2d(centroids: list[Point2D]) -> DistMatrix:
    dist_matrix: DistMatrix = [
        [0.0] * len(centroids)
        for _ in range(len(centroids))
    ]

    for i in range(len(centroids)):
        for j in range(i, len(centroids)):
            dist = dist2d(centroids[i], centroids[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix


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