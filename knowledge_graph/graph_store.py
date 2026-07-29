"""
knowledge_graph/graph_store.py
--------------------------------
In-memory Knowledge Graph backed by NetworkX.
Supports save/load as JSON for persistence.

In production, swap for Neo4jStore (see neo4j_store.py stub).
The interface is identical — just change `backend` in config.yaml.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx

from .schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)


class NetworkXGraphStore:
    """
    Wraps a networkx.MultiDiGraph to store typed KG nodes and edges.

    Node attributes: all fields of KGNode
    Edge attributes: all fields of KGEdge (keyed by edge_type string)
    """

    def __init__(self):
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ─── Write API ───────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> None:
        self._g.add_node(node.node_id, **node.to_dict())

    def add_nodes(self, nodes: Iterable[KGNode]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, edge: KGEdge) -> None:
        if not self._g.has_node(edge.src_id):
            logger.debug(f"  Missing src node: {edge.src_id}")
            return
        if not self._g.has_node(edge.dst_id):
            logger.debug(f"  Missing dst node: {edge.dst_id}")
            return
        self._g.add_edge(
            edge.src_id,
            edge.dst_id,
            key=edge.edge_type.value,
            **edge.to_dict(),
        )

    def add_edges(self, edges: Iterable[KGEdge]) -> None:
        for e in edges:
            self.add_edge(e)

    # ─── Read API ────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[KGNode]:
        if not self._g.has_node(node_id):
            return None
        return KGNode.from_dict(dict(self._g.nodes[node_id]))

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> List[KGNode]:
        result = []
        for _, dst_id, data in self._g.out_edges(node_id, data=True):
            if edge_types:
                if EdgeType(data.get("edge_type")) not in edge_types:
                    continue
            n = self.get_node(dst_id)
            if n:
                result.append(n)
        return result

    def get_nodes_by_type(self, node_type: NodeType) -> List[KGNode]:
        return [
            KGNode.from_dict(dict(d))
            for _, d in self._g.nodes(data=True)
            if d.get("node_type") == node_type.value
        ]

    def get_nodes_by_paper(self, paper_id: str) -> List[KGNode]:
        return [
            KGNode.from_dict(dict(d))
            for _, d in self._g.nodes(data=True)
            if d.get("paper_id") == paper_id
        ]

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        edge_counts: Dict[str, int] = {}
        for _, d in self._g.nodes(data=True):
            t = d.get("node_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        for _, _, d in self._g.edges(data=True):
            t = d.get("edge_type", "Unknown")
            edge_counts[t] = edge_counts.get(t, 0) + 1
        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "node_types":  type_counts,
            "edge_types":  edge_counts,
        }

    # ─── Subgraph / path API (used by Self-Play random walk) ─────────────

    def get_subgraph(self, node_ids: List[str]) -> "NetworkXGraphStore":
        sub = NetworkXGraphStore()
        sub._g = self._g.subgraph(node_ids).copy()
        return sub

    def random_walk(
        self,
        start_id: str,
        max_hops: int = 3,
        edge_weight_key: str = "weight",
    ) -> List[str]:
        """
        Weighted random walk from start_id.
        Returns list of visited node IDs (path).
        """
        import random
        path = [start_id]
        current = start_id
        for _ in range(max_hops):
            out_edges = list(self._g.out_edges(current, data=True))
            if not out_edges:
                break
            weights = [e[2].get(edge_weight_key, 1.0) for e in out_edges]
            total = sum(weights)
            if total == 0:
                break
            probs = [w / total for w in weights]
            chosen = random.choices(out_edges, weights=probs, k=1)[0]
            nxt = chosen[1]
            path.append(nxt)
            current = nxt
        return path

    # ─── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": nid, **dict(d)}
                for nid, d in self._g.nodes(data=True)
            ],
            "edges": [
                dict(d) for _, _, d in self._g.edges(data=True)
            ],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved → {p}  "
                    f"({self.node_count()} nodes, {self.edge_count()} edges)")

    def load(self, path: str) -> None:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._g = nx.MultiDiGraph()
        for n in data["nodes"]:
            nid = n.pop("id", n.get("node_id"))
            # meta_* 키를 metadata dict로 복원
            meta = {k[5:]: v for k, v in list(n.items()) if k.startswith("meta_")}
            for k in list(meta.keys()):
                n.pop(f"meta_{k}", None)
            if meta:
                n["metadata"] = {**n.get("metadata", {}), **meta}
            self._g.add_node(nid, **n)
        for e in data["edges"]:
            src = e.get("src_id")
            dst = e.get("dst_id")
            key = e.get("edge_type", "CONTAINS")
            self._g.add_edge(src, dst, key=key, **e)
        logger.info(f"Graph loaded ← {p}  "
                    f"({self.node_count()} nodes, {self.edge_count()} edges)")

    @classmethod
    def merge(cls, stores: List["NetworkXGraphStore"]) -> "NetworkXGraphStore":
        """Merge multiple stores into one (for cross-doc federation)."""
        merged = cls()
        for store in stores:
            merged._g.update(store._g)
        return merged


# ─── Factory: select backend from config ─────────────────────────────────────

def create_store(kg_cfg: dict):
    """
    Return the appropriate graph store based on config.yaml.

    config.yaml:
        knowledge_graph:
          backend: "networkx"   # dev / single-machine
          # backend: "neo4j"    # production / large scale

    Returns NetworkXGraphStore or Neo4jGraphStore with identical interface.
    """
    backend = kg_cfg.get("backend", "networkx").lower()

    if backend == "neo4j":
        from .neo4j_store import Neo4jGraphStore
        neo4j_cfg = kg_cfg.get("neo4j", {})
        return Neo4jGraphStore(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            user=neo4j_cfg.get("user", "neo4j"),
            password=neo4j_cfg.get("password", "password"),
            database=neo4j_cfg.get("database", "neo4j"),
        )

    # default: networkx
    return NetworkXGraphStore()
