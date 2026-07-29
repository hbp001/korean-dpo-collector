"""
knowledge_graph/cross_doc_federation.py
-----------------------------------------
Phase 1-D: Cross-Document KG Federation

Merges KGs from multiple papers by detecting shared concepts
(same entity / model / method) and linking them with
SAME_CONCEPT edges. This enables cross-document reasoning.

Algorithm:
1. Collect all Concept / Claim / TextBlock nodes across papers.
2. Embed them with sentence-transformers.
3. For pairs above `coreference_threshold`, add SAME_CONCEPT edge.
4. Optionally merge near-duplicate concept nodes (future work).
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


# Node types considered for cross-doc linking
_LINKABLE = {NodeType.CONCEPT, NodeType.CLAIM, NodeType.TEXT_BLOCK}

# Minimum text length to be considered a meaningful concept
_MIN_LEN = 30


class CrossDocFederator:
    """
    Takes the combined node set from all papers and produces
    SAME_CONCEPT edges that cross paper boundaries.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.threshold: float = cfg.get("coreference_threshold", 0.85)
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not available — "
                    "cross-doc federation skipped."
                )
        return self._model

    def build(
        self,
        all_nodes: Dict[str, List[KGNode]],  # {paper_id: [nodes]}
    ) -> List[KGEdge]:
        """
        Build cross-document SAME_CONCEPT edges.
        `all_nodes` maps paper_id → list of KGNode.
        Returns only the new cross-doc edges.
        """
        model = self._get_model()
        if model is None:
            return []

        # Flatten to (paper_id, node) pairs, filtering linkable nodes
        candidates: List[Tuple[str, KGNode]] = []
        for pid, nodes in all_nodes.items():
            for n in nodes:
                if (
                    n.node_type in _LINKABLE
                    and n.content
                    and len(n.content.strip()) >= _MIN_LEN
                ):
                    candidates.append((pid, n))

        if len(candidates) < 2:
            return []

        logger.info(
            f"Cross-doc federation: embedding {len(candidates)} "
            f"candidate nodes from {len(all_nodes)} papers…"
        )

        texts = [c[1].content[:512] for c in candidates]
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # 수정 후
        edges: List[KGEdge] = []
        n = len(candidates)

        # 메모리 확인 후 배치/전체 방식 자동 선택
        # 73k 노드 × 384차원 float32 = 약 107MB (embeddings)
        # sim_matrix = n×n float32 → 73k²×4bytes ≈ 21GB → 배치 처리 필요
        BATCH_SIZE = 2048  # 행 단위로 나눠서 계산
        logger.info(
            f"Computing similarity in batches "
            f"(n={n}, batch={BATCH_SIZE}, threshold={self.threshold})..."
        )

        # paper_id 룩업 배열 (같은 논문 제외용)
        pids = [candidates[i][0] for i in range(n)]

        for i_start in range(0, n, BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, n)
            batch_emb = embeddings[i_start:i_end]  # (batch, dim)

            # batch × n 유사도 행렬 (normalize되어 있으므로 내적 = 코사인)
            sim_block = batch_emb @ embeddings.T   # (batch, n)

            for local_i, global_i in enumerate(range(i_start, i_end)):
                pid_i, node_i = candidates[global_i]

                # j > i (상삼각), 같은 논문 제외, threshold 이상
                row = sim_block[local_i]            # (n,)
                js  = np.where(
                    (row >= self.threshold) &
                    (np.arange(n) > global_i)
                )[0]

                for j in js:
                    pid_j = pids[j]
                    if pid_i == pid_j:
                        continue
                    node_j = candidates[j][1]
                    sim = float(row[j])
                    edges.append(KGEdge(
                        src_id=node_i.node_id,
                        dst_id=node_j.node_id,
                        edge_type=EdgeType.SAME_CONCEPT,
                        weight=round(sim, 4),
                        confidence=round(sim, 4),
                        metadata={
                            "method": "crossdoc_embedding",
                            "paper_src": pid_i,
                            "paper_dst": pid_j,
                            "similarity": round(sim, 4),
                        },
                    ))

            if i_start % (BATCH_SIZE * 10) == 0:
                logger.info(
                    f"  Cross-doc progress: {i_end}/{n} rows, "
                    f"{len(edges)} edges so far"
                )

        logger.info(
            f"Cross-doc federation: {len(edges)} SAME_CONCEPT edges "
            f"(threshold={self.threshold})"
        )
        return edges
