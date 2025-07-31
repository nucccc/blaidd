import polars as pl
from dataclasses import dataclass, field
from math import sqrt
from typing import Any


def color_series(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
) -> pl.Series:
    label_centroids = build_label_centroids(
        df,
        x_col,
        y_col,
        c_col,
    )

    graph = _build_graph(label_centroids)
    scale = colors_scale(len(graph.nodes))


def colors_scale(n: int) -> list[float]:
    step = 1.0 / (n - 1)
    return [i * step for i in range(n-1)] + [1.0]


def build_label_centroids(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
) -> dict[Any, tuple[float, float]]:
    return {
        label : (gdf[x_col].mean(), gdf[y_col].mean())
        for label, gdf in df.group_by(c_col)
    }


@dataclass
class CentroidNode:
    label: Any
    edges: list[tuple[float, 'CentroidNode']] = field(default_factory=list)
    color_scale: float | None = None


@dataclass
class CentroidGraph:
    nodes: list[CentroidNode]

    def farthest_nodes(self) -> tuple[CentroidNode, CentroidNode]:
        max_dist = 0.0
        far1 = None
        far2 = None
        for node in self.nodes:
            for dist, other_node in node.edges:
                if dist > max_dist:
                    max_dist = dist
                    far1 = node
                    far2 = other_node

        return far1, far2


def _build_graph(
    label_centroids: dict[Any, tuple[float, float]]
) -> CentroidGraph:
    nodes_by_label = {
        label: CentroidNode(label=label)
        for label in label_centroids.keys()
    }

    label_list = list(label_centroids.keys())

    for i in range(len(label_list)):
        for j in range(i+1, len(label_list)):
            label1 = label_list[i]
            label2 = label_list[j]
            x1, y1 = label_centroids[label1]
            x2, y2 = label_centroids[label2]

            dist = centroids_dist(x1, y1, x2, y2)

            node1 = nodes_by_label[label1]
            node2 = nodes_by_label[label2]
            node1.edges.append((dist, node2))
            node2.edges.append((dist, node1))

    nodes = list(nodes_by_label.values())

    # for every node the edges shall be ordered by distance
    for node in nodes:
        node.edges.sort(key= lambda x : x[0])

    return CentroidGraph(nodes=nodes)


def centroids_dist(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return sqrt(((x1 - x2)** 2) + ((y1 - y2)**2))