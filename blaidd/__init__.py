import numpy.typing as npt
import random
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import polars as pl
from matplotlib import pyplot as plt


from blaidd.centroids import calc_centroids
from blaidd.cut import seek_valid_edges, dist2d


def new_stuff(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
) -> list[float | int]:
    labels, centroids = calc_centroids(df, x_col, y_col, c_col)

    edges = seek_valid_edges(centroids)

    graph = new_graph_build(centroids, edges, labels)

    bees_graph = abc(graph, n_bees=100, max_stuck=50, n_cycles=500)

    return pl.Series(
        name='color_graphed',
        values=[bees_graph.get_label_color(label) for label in df[c_col]]
    )


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

    assign_colors(graph, df, x_col, y_col, c_col)

    return pl.Series(
        name='color_graphed',
        values=[graph.get_label_color(label) for label in df[c_col]]
    )


def assign_colors(graph: 'CentroidGraph', df, x_col, y_col, c_col):
    scale = Scale(len(graph.nodes))

    # select the two farthest nodes
    farthest1, farthest2 = graph.farthest_nodes()
    farthest1.color_scale = scale.pop_first()
    farthest2.color_scale = scale.pop_first()

    while graph.n_uncolored() > 0:
        df = df.with_columns(
            pl.Series(
                name='color_graphed',
                values=[graph.get_label_color(label) for label in df[c_col]]
            )
        )
        plt.scatter(
            x = df['x'],
            y = df['y'],
            c = df['color_graphed'],
            s = 4
        )
        plt.show()

        closest_uncolored = graph.pick_closest_uncolored()
        closest_uncolored.color_scale = scale.pop_farthest(
            other_colors = closest_uncolored.pick_closest_colors()
        )


def color_series_with_bees(
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

    bees_graph = abc(graph, n_bees=100, max_stuck=50, n_cycles=500)

    return pl.Series(
        name='color_graphed',
        values=[bees_graph.get_label_color(label) for label in df[c_col]]
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
            score = sqrt(sum((abs(color - other_color)**2) for other_color in other_colors))
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
            
    def perform_switches(self, switches: list[tuple[int, int]]):
        for switch in switches:
            c = self.nodes[switch[0]].color_scale
            self.nodes[switch[0]].color_scale = self.nodes[switch[1]].color_scale
            self.nodes[switch[1]].color_scale = c

    def reset_switches(self, switches: list[tuple[int, int]]):
        for switch in reversed(switches):
            c = self.nodes[switch[0]].color_scale
            self.nodes[switch[0]].color_scale = self.nodes[switch[1]].color_scale
            self.nodes[switch[1]].color_scale = c


def new_graph_build(
    centroids: npt.ArrayLike,
    edges: list[tuple[int, int]],
    labels: list
):
    nodes = [
        CentroidNode(label=label)
        for label in labels
    ]

    # adding edges to nodes
    for i, j in edges:
        dist = dist2d(centroids[i], centroids[j])
        nodes[i].edges.append((dist, nodes[j]))
        nodes[j].edges.append((dist, nodes[i]))

    # TODO: maybe sorting the node edges by distance would be a good idea

    result = CentroidGraph(nodes=nodes)

    return result


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


# ok everything was fun and games but now it would be nice
# to a have an artificial bee colony / genetic algorithm or something like
# that to play with


def calc_fitness(graph: CentroidGraph, n_neighbors: int = 4) -> float:
    score = 0.0
    for node in graph.nodes:
        node_score = 0.0
        # n_node_neighbors shall be recalculated because you never know
        n_node_neighbors = 0
        for dist, other_node in node.edges[:n_neighbors]:
            node_score += (min(abs(other_node.color_scale - node.color_scale), 6)) ** 2
            n_node_neighbors += 1
        node_score = sqrt(node_score) / n_node_neighbors
        score += node_score
    score = score / len(graph.nodes)
    return score


def random_switch(end: int) -> tuple[int, int]:
    first = random.randint(0, end - 1)
    second = random.randint(0, end - 1)
    if second == first:
        second += 1
        if second >= end:
            second = 0
    return (first, second)


def abc(
    graph: CentroidGraph,
    n_bees: int,
    max_stuck: int,
    n_cycles: int,
) -> CentroidGraph:
    bees: list[Bee] = [
        Bee(
            uncolored_graph=graph,
            max_stuck=max_stuck,
        )
        for _ in range(n_bees)
    ]

    for _ in range(n_cycles):
        for bee in bees:
            bee.iterate()

    best_graph = None
    best_fitness = None
    for bee in bees:
        if best_fitness is None or bee.fitness > best_fitness:
            best_fitness = bee.fitness
            best_graph = bee.graph

    return best_graph


class Bee:

    def __init__(
        self,
        uncolored_graph: CentroidGraph,
        max_stuck: int
    ):
        self.graph = deepcopy(uncolored_graph)
        self.max_stuck = max_stuck
        self._initial_assignment()

    def _initial_assignment(self):
        self._reset()

    def _reset(self):
        self._reset_colors()

        self.fitness = calc_fitness(self.graph)
        self.n_stuck = 0

    def _reset_colors(self):
        colors = list(range(len(self.graph.nodes)))
        random.shuffle(colors)
        for node, color in zip(self.graph.nodes, colors):
            node.color_scale = color

    def iterate(self, n_switches: int = 1):
        switches: list[tuple[int]] = [
            random_switch(len(self.graph.nodes))
            for _ in range(n_switches)
        ]
        self.graph.perform_switches(switches)
        new_fitness = calc_fitness(self.graph)

        if new_fitness > self.fitness:
            self.fitness = new_fitness
            self.n_stuck = 0
        else:
            self.n_stuck += 1
            if self.n_stuck >= self.max_stuck:
                self._reset()
            else:
                self.graph.reset_switches(switches)



@dataclass
class ABCParams:
    n_bees: int
    bee_life: int


class ABC:

    def __init__(
        self,
        params: ABCParams,
        uncolored_graph: CentroidGraph
    ):
        self.params = params
        self._uncolored_graph = deepcopy(uncolored_graph)
