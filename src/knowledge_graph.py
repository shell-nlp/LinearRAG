from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import igraph as ig


class KnowledgeGraph(ABC):
    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node_id: str, content: str, node_type: str, **attrs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, source: str, target: str, weight: float, **attrs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def personalized_pagerank(
        self,
        reset_weights: Mapping[str, float],
        damping: float,
        node_type: str | None = None,
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError


class IgraphKnowledgeGraph(KnowledgeGraph):
    def __init__(self):
        self._graph = ig.Graph(directed=False)
        self._node_index: dict[str, int] = {}

    def clear(self) -> None:
        self._graph = ig.Graph(directed=False)
        self._node_index = {}

    def add_node(self, node_id: str, content: str, node_type: str, **attrs: Any) -> None:
        if node_id in self._node_index:
            vertex = self._graph.vs[self._node_index[node_id]]
            vertex["content"] = content
            vertex["node_type"] = node_type
            for key, value in attrs.items():
                vertex[key] = value
            return

        self._graph.add_vertex(name=node_id, content=content, node_type=node_type, **attrs)
        self._node_index[node_id] = self._graph.vcount() - 1

    def add_edge(self, source: str, target: str, weight: float, **attrs: Any) -> None:
        if source == target:
            return
        if source not in self._node_index or target not in self._node_index:
            raise KeyError("Both nodes must exist before adding an edge")

        source_idx = self._node_index[source]
        target_idx = self._node_index[target]
        edge_id = self._graph.get_eid(source_idx, target_idx, directed=False, error=False)
        if edge_id == -1:
            self._graph.add_edge(source_idx, target_idx, weight=weight, **attrs)
            return

        current_weight = float(self._graph.es[edge_id]["weight"])
        self._graph.es[edge_id]["weight"] = current_weight + weight
        for key, value in attrs.items():
            self._graph.es[edge_id][key] = value

    def personalized_pagerank(
        self,
        reset_weights: Mapping[str, float],
        damping: float,
        node_type: str | None = None,
    ) -> list[tuple[str, float]]:
        if self._graph.vcount() == 0:
            return []

        reset = [0.0] * self._graph.vcount()
        positive_weight = False
        for node_id, weight in reset_weights.items():
            if weight <= 0 or node_id not in self._node_index:
                continue
            reset[self._node_index[node_id]] = float(weight)
            positive_weight = True

        if not positive_weight:
            return []

        scores = self._graph.personalized_pagerank(
            vertices=range(self._graph.vcount()),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset,
            implementation="prpack",
        )

        ranked: list[tuple[str, float]] = []
        for vertex in self._graph.vs:
            if node_type and vertex["node_type"] != node_type:
                continue
            ranked.append((vertex["name"], float(scores[vertex.index])))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._graph.write_graphml(str(target))
