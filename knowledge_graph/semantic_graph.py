"""
knowledge_graph/semantic_graph.py
-----------------------------------
Phase 1-C: Semantic Relation Graph

Uses sentence-transformers to compute embeddings and create
SUPPORTS / ILLUSTRATES / COMPARES / etc. edges between nodes
that are semantically related above a configurable threshold.

VLM-based relation extraction is stubbed here and activated in Phase 2.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _infer_edge_type(
    src: KGNode, dst: KGNode, sim: float
) -> EdgeType:
    """
    Heuristic to assign an edge type based on node types.
    In Phase 2 this will be replaced by VLM-based classification.
    """
    s, d = src.node_type, dst.node_type

    if d in (NodeType.FIGURE, NodeType.TABLE):
        return EdgeType.ILLUSTRATES
    if s == NodeType.CLAIM and d == NodeType.CLAIM:
        return EdgeType.SUPPORTS if sim > 0.85 else EdgeType.COMPARES
    if d == NodeType.EQUATION:
        return EdgeType.QUANTIFIES
    if s == NodeType.TEXT_BLOCK and d == NodeType.TEXT_BLOCK:
        return EdgeType.SUPPORTS
    return EdgeType.SUPPORTS


class SemanticGraphBuilder:
    """
    Computes text embeddings for KG nodes and adds semantic edges
    between nodes whose cosine similarity exceeds the threshold.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.threshold: float = cfg.get("similarity_threshold", 0.75)
        self.max_edges: int   = cfg.get("max_edges_per_node", 10)
        self.batch_size: int  = cfg.get("batch_size", 32)
        self._model = None   # lazy load

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 임베딩 모델을 설정으로 분기 (기본값은 기존과 동일 → 영어 KG 결과 불변).
                # 한국어 코퍼스는 config에서 다국어 모델을 지정한다.
                model_name = self.cfg.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info(f"Loading embedding model: {model_name}")
                self._model = SentenceTransformer(model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic graph will be skipped. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def build(
        self,
        paper_id: str,
        nodes: List[KGNode],
    ) -> List[KGEdge]:
        """
        Compute embeddings for text-bearing nodes and add semantic edges.
        Returns new edges (does not mutate nodes).
        """
        model = self._get_model()
        if model is None:
            return []

        # Filter nodes that have meaningful text content
        candidate_nodes = [
            n for n in nodes
            if n.content and len(n.content.strip()) > 20
            and n.node_type != NodeType.PAPER
        ]

        if len(candidate_nodes) < 2:
            return []

        texts = [n.content[:512] for n in candidate_nodes]  # truncate

        logger.debug(f"[{paper_id}] computing embeddings for {len(texts)} nodes…")
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        edges: List[KGEdge] = []
        n = len(candidate_nodes)

        for i in range(n):
            similarities: List[Tuple[float, int]] = []
            for j in range(n):
                if i == j:
                    continue
                sim = _cosine(embeddings[i], embeddings[j])
                if sim >= self.threshold:
                    similarities.append((sim, j))

            # Keep top-K neighbours per node
            similarities.sort(reverse=True)
            for sim, j in similarities[: self.max_edges]:
                src = candidate_nodes[i]
                dst = candidate_nodes[j]
                etype = _infer_edge_type(src, dst, sim)
                edges.append(KGEdge(
                    src_id=src.node_id,
                    dst_id=dst.node_id,
                    edge_type=etype,
                    weight=round(sim, 4),
                    confidence=round(sim, 4),
                    metadata={
                        "method": "embedding",
                        "similarity": round(sim, 4),
                    },
                ))

        logger.debug(
            f"[{paper_id}] semantic: {len(edges)} edges "
            f"(threshold={self.threshold})"
        )
        return edges
