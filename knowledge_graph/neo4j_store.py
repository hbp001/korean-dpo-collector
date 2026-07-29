"""
knowledge_graph/neo4j_store.py
--------------------------------
Production-grade Knowledge Graph backed by Neo4j.
Interface is identical to NetworkXGraphStore — swap via config.yaml:

    knowledge_graph:
      backend: "neo4j"

Requirements:
    pip install neo4j

Neo4j 실행 (Docker):
    docker run -d \
        --name neo4j \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        neo4j:5
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, Iterable, List, Optional

from .schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Neo4j-backed Knowledge Graph store.
    Same public interface as NetworkXGraphStore.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package not installed.\n"
                "Run: pip install neo4j"
            )
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._ensure_constraints()
        logger.info(f"Neo4j connected → {uri} / db={database}")

    # ─── Internal helpers ────────────────────────────────────────────────

    def _run(self, query: str, **params) -> List[Dict[str, Any]]:
        with self._driver.session(database=self._db) as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraint on node_id (idempotent)."""
        self._run(
            "CREATE CONSTRAINT kg_node_id IF NOT EXISTS "
            "FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE"
        )

    # ─── Write API ───────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> None:
        d = node.to_dict()
        # Neo4j cannot store nested dicts natively — flatten metadata
        flat_meta = {f"meta_{k}": str(v) for k, v in d.pop("metadata", {}).items()}
        props = {**d, **flat_meta}
        self._run(
            "MERGE (n:KGNode {node_id: $node_id}) "
            "SET n += $props, n.node_type = $node_type",
            node_id=node.node_id,
            props=props,
            node_type=node.node_type.value,
        )

    def add_nodes(self, nodes: Iterable[KGNode]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, edge: KGEdge) -> None:
        etype = edge.edge_type.value
        meta_flat = {f"meta_{k}": str(v) for k, v in edge.metadata.items()}
        props = {
            "src_id":     edge.src_id,
            "dst_id":     edge.dst_id,
            "edge_type":  etype,
            "weight":     edge.weight,
            "confidence": edge.confidence,
            **meta_flat,
        }
        # MERGE prevents duplicate edges of the same type between same nodes
        self._run(
            f"MATCH (a:KGNode {{node_id: $src_id}}) "
            f"MATCH (b:KGNode {{node_id: $dst_id}}) "
            f"MERGE (a)-[r:{etype}]->(b) "
            f"SET r += $props",
            src_id=edge.src_id,
            dst_id=edge.dst_id,
            props=props,
        )

    def add_edges(self, edges: Iterable[KGEdge]) -> None:
        for e in edges:
            self.add_edge(e)

    # ─── Read API ────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[KGNode]:
        rows = self._run(
            "MATCH (n:KGNode {node_id: $node_id}) RETURN properties(n) AS props",
            node_id=node_id,
        )
        if not rows:
            return None
        return self._row_to_node(rows[0]["props"])

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> List[KGNode]:
        if edge_types:
            type_filter = "|".join(et.value for et in edge_types)
            query = (
                f"MATCH (a:KGNode {{node_id: $node_id}})"
                f"-[:{type_filter}]->(b:KGNode) "
                f"RETURN properties(b) AS props"
            )
        else:
            query = (
                "MATCH (a:KGNode {node_id: $node_id})-[]->(b:KGNode) "
                "RETURN properties(b) AS props"
            )
        rows = self._run(query, node_id=node_id)
        return [self._row_to_node(r["props"]) for r in rows]

    def get_nodes_by_type(self, node_type: NodeType) -> List[KGNode]:
        rows = self._run(
            "MATCH (n:KGNode {node_type: $ntype}) RETURN properties(n) AS props",
            ntype=node_type.value,
        )
        return [self._row_to_node(r["props"]) for r in rows]

    def get_nodes_by_paper(self, paper_id: str) -> List[KGNode]:
        rows = self._run(
            "MATCH (n:KGNode {paper_id: $pid}) RETURN properties(n) AS props",
            pid=paper_id,
        )
        return [self._row_to_node(r["props"]) for r in rows]

    def node_count(self) -> int:
        rows = self._run("MATCH (n:KGNode) RETURN count(n) AS cnt")
        return rows[0]["cnt"] if rows else 0

    def edge_count(self) -> int:
        rows = self._run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return rows[0]["cnt"] if rows else 0

    def stats(self) -> Dict[str, Any]:
        node_rows = self._run(
            "MATCH (n:KGNode) RETURN n.node_type AS t, count(*) AS c"
        )
        edge_rows = self._run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c"
        )
        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "node_types":  {r["t"]: r["c"] for r in node_rows},
            "edge_types":  {r["t"]: r["c"] for r in edge_rows},
        }

    # ─── Subgraph / random walk (used by Self-Play) ──────────────────────

    def random_walk(
        self,
        start_id: str,
        max_hops: int = 3,
        edge_weight_key: str = "weight",
    ) -> List[str]:
        """Weighted random walk from start_id, returns visited node_id list."""
        path = [start_id]
        current = start_id
        for _ in range(max_hops):
            rows = self._run(
                "MATCH (a:KGNode {node_id: $nid})-[r]->(b:KGNode) "
                "RETURN b.node_id AS nid, r.weight AS w",
                nid=current,
            )
            if not rows:
                break
            weights = [float(r.get("w") or 1.0) for r in rows]
            total = sum(weights)
            if total == 0:
                break
            chosen = random.choices(rows, weights=[w / total for w in weights], k=1)[0]
            nxt = chosen["nid"]
            path.append(nxt)
            current = nxt
        return path

    # ─── Persistence (export/import via JSON) ────────────────────────────

    def save(self, path: str) -> None:
        """Export the entire graph to a JSON file (same format as NetworkX store)."""
        import json
        from pathlib import Path

        node_rows = self._run(
            "MATCH (n:KGNode) RETURN properties(n) AS props"
        )
        edge_rows = self._run(
            "MATCH (a:KGNode)-[r]->(b:KGNode) "
            "RETURN properties(r) AS props"
        )
        data = {
            "nodes": [{"id": r["props"].get("node_id"), **r["props"]} for r in node_rows],
            "edges": [r["props"] for r in edge_rows],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Neo4j graph exported → {p}  "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    def load(self, path: str) -> None:
        """Import a JSON file (produced by save() or NetworkXGraphStore.save())."""
        import json
        from pathlib import Path

        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)

        for n in data.get("nodes", []):
            n.pop("id", None)
            node = self._row_to_node(n)
            self.add_node(node)

        for e in data.get("edges", []):
            try:
                edge = KGEdge.from_dict({
                    "src_id":     e["src_id"],
                    "dst_id":     e["dst_id"],
                    "edge_type":  e["edge_type"],
                    "weight":     float(e.get("weight", 1.0)),
                    "confidence": float(e.get("confidence", 1.0)),
                    "metadata":   {},
                })
                self.add_edge(edge)
            except Exception as exc:
                logger.warning(f"Skipping edge load: {exc}")

        logger.info(
            f"Neo4j graph imported ← {path}  "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    def clear(self) -> None:
        """Delete all KGNode nodes and their relationships."""
        self._run("MATCH (n:KGNode) DETACH DELETE n")
        logger.info("Neo4j graph cleared.")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ─── Internal deserialization ────────────────────────────────────────

    @staticmethod
    def _row_to_node(props: Dict[str, Any]) -> KGNode:
        """Convert a flat Neo4j property map back to a KGNode."""
        # Reconstruct metadata from flattened meta_* keys
        meta = {
            k[5:]: v
            for k, v in props.items()
            if k.startswith("meta_")
        }
        return KGNode(
            node_id=props.get("node_id", ""),
            node_type=NodeType(props.get("node_type", "TextBlock")),
            paper_id=props.get("paper_id", ""),
            page=int(props.get("page", -1)),
            content=props.get("content", ""),
            image_path=props.get("image_path"),
            metadata=meta,
        )
