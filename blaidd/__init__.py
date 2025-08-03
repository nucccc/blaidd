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

    assign_colors(graph)

    return pl.Series(
        name='color_graphed',
        values=[graph.get_label_color(label) for label in df[c_col]]
    )


def assign_colors(graph: 'CentroidGraph'):
    scale = Scale(len(graph.nodes))

    # select the two farthest nodes
    farthest1, farthest2 = graph.farthest_nodes()
    farthest1.color_scale = scale.pop_first()
    farthest2.color_scale = scale.pop_first()

    while graph.n_uncolored() > 0:
        closest_uncolored = graph.pick_closest_uncolored()
        closest_uncolored.color_scale = scale.pop_farthest(
            other_colors = closest_uncolored.pick_closest_colors()
        )



ScaleElem = int

class Scale:

    def __init__(self, n: int):
        self.scale = colors_scale(n)

    def pop_first(self) -> ScaleElem:
        return self.scale.pop(0)

    def pop_farthest(self, other_colors: list[ScaleElem]) -> ScaleElem:
        max_score = None
        farthest = None
        for color in self.scale:
            score = sum(abs(color - other_color) for other_color in other_colors)
            if max_score is None or score > max_score:
                max_score = score
                farthest = color
        
        assert farthest is not None
        self.scale.remove(farthest)
        return farthest

def colors_scale(n: int) -> list[ScaleElem]:
    return list(range(n))
    step = 1.0 / (n - 1)
    return [i * step for i in range(n-1)] + [1.0]


def build_label_centroids(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
) -> dict[Any, tuple[float, float]]:
    return {
        label[0] : (gdf[x_col].mean(), gdf[y_col].mean())
        for label, gdf in df.group_by(c_col)
    }


@dataclass
class CentroidNode:
    label: Any
    # edges shall be ordered for closest to farthest
    edges: list[tuple[float, 'CentroidNode']] = field(default_factory=list)
    color_scale: ScaleElem | None = None

    def colored(self) -> bool:
        return not self.uncolored()
    
    def uncolored(self) -> bool:
        return self.color_scale is None

    def min_dist_to_colored(self) -> float:
        return min(dist for dist, node in self.edges if node.colored())
    
    def pick_closest_colors(self, n_closest: int = 4) -> list[ScaleElem]:
        closest_colors: list[ScaleElem] = list()
        for dist, node in self.edges[:n_closest]:
            if node.colored():
                closest_colors.append(node.color_scale)

        return closest_colors


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
    
    def n_uncolored(self) -> int:
        return sum(node.uncolored() for node in self.nodes)
    
    def pick_closest_uncolored(self) -> CentroidNode:
        '''
        picks the closest uncolored node to a colored one
        '''
        min_dist = None
        closest_uncolored = None

        for node in self.nodes:
            if node.colored():
                continue

            dist = node.min_dist_to_colored()
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest_uncolored = node

        return closest_uncolored
    
    def get_label_color(self, label: Any) -> ScaleElem:
        for node in self.nodes:
            if node.label == label:
                return node.color_scale


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