"""
knowledge_graph/kg_pipeline.py
--------------------------------
Phase 1 entry point.

Runs all three graph-building stages in order:
  1. Structural Graph   (rule-based hierarchy)
  2. Reference Graph    (in-text cross-references)
  3. Semantic Graph     (embedding-based similarity)
  4. Cross-Doc Federation (only when >1 paper)

Usage:
    from knowledge_graph.kg_pipeline import KGPipeline
    pipeline = KGPipeline(cfg["knowledge_graph"], data_root)
    store = pipeline.run(paper_ids=None)   # None = all papers
    store.save("outputs/kg.json")
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .cross_doc_federation import CrossDocFederator
from .graph_store import NetworkXGraphStore, create_store
from .parser import PaddleJSONParser
from .reference_graph import ReferenceGraphBuilder
from .schema import KGNode
from .semantic_graph import SemanticGraphBuilder
from .structural_graph import StructuralGraphBuilder
from .vlm_enricher import VLMEnricher

logger = logging.getLogger(__name__)


class KGPipeline:
    """
    Full Phase-1 pipeline: parse → structural → reference → semantic → cross-doc.
    """

    def __init__(self, kg_cfg: dict, data_root: str):
        self.cfg = kg_cfg
        self.data_root = data_root

        self.parser    = PaddleJSONParser(data_root)
        self.struct    = StructuralGraphBuilder(kg_cfg.get("structural", {}))
        self.ref       = ReferenceGraphBuilder(kg_cfg.get("reference", {}))
        self.semantic  = SemanticGraphBuilder(kg_cfg.get("semantic", {}))
        self.crossdoc  = CrossDocFederator(kg_cfg.get("cross_document", {}))
        self.vlm       = VLMEnricher(
            cfg=kg_cfg.get("vlm_enrichment", {}),
            vllm_cfg=self._vllm_cfg,
        )

    @property
    def _vllm_cfg(self) -> dict:
        """config.yaml의 models.vllm 섹션을 읽어옵니다."""
        try:
            from config.config_loader import get_config
            return get_config().get("models", {}).get("vllm", {})
        except Exception:
            return {}

    def run(
        self,
        paper_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> NetworkXGraphStore:
        """
        Build the full KG.

        Args:
            paper_ids: list of paper folder names to process.
                       None → process all papers found.
            save_path: if given, save the graph to this JSON path.

        Returns:
            Populated NetworkXGraphStore.
        """
        available = self.parser.list_papers()
        if paper_ids is None:
            paper_ids = available
        else:
            paper_ids = [p for p in paper_ids if p in available]

        logger.info(f"=== KG Pipeline: {len(paper_ids)} papers ===")
        t0 = time.time()

        # Per-paper stores (merged later for cross-doc)
        per_paper_nodes: Dict[str, List[KGNode]] = {}
        master_store = create_store(self.cfg)

        for i, pid in enumerate(paper_ids):
            logger.info(f"[{i+1}/{len(paper_ids)}] Processing: {pid}")
            t_paper = time.time()

            # ── Parse ──────────────────────────────────────────────
            elements = self.parser.parse_paper(pid)
            if not elements:
                logger.warning(f"  No elements found — skipping {pid}")
                continue

            # ── Stage 1: Structural ────────────────────────────────
            nodes, struct_edges = self.struct.build(pid, elements)

            # ── Stage 2: Reference ─────────────────────────────────
            ref_edges = self.ref.build(pid, nodes)

            # ── Stage 3: Semantic ──────────────────────────────────
            if self.cfg.get("semantic", {}).get("enabled", True):
                sem_edges = self.semantic.build(pid, nodes)
            else:
                sem_edges = []

            # ── Stage 4: VLM Enrichment ────────────────────────────
            # 구조/참조 엣지를 모두 합친 뒤 VLM에 전달
            all_edges_so_far = struct_edges + ref_edges + sem_edges
            vlm_nodes, vlm_edges = self.vlm.enrich(pid, nodes, all_edges_so_far)

            # ── Load into store ────────────────────────────────────
            master_store.add_nodes(nodes)
            master_store.add_nodes(vlm_nodes)          # Concept/Claim 노드
            master_store.add_edges(struct_edges)
            master_store.add_edges(ref_edges)
            master_store.add_edges(sem_edges)
            master_store.add_edges(vlm_edges)          # VLM 분류 엣지

            per_paper_nodes[pid] = nodes + vlm_nodes

            dt = time.time() - t_paper
            logger.info(
                f"  → {len(nodes)+len(vlm_nodes)} nodes | "
                f"{len(struct_edges)} structural | "
                f"{len(ref_edges)} reference | "
                f"{len(sem_edges)} semantic | "
                f"{len(vlm_nodes)} vlm_nodes | "
                f"{len(vlm_edges)} vlm_edges | "
                f"{dt:.1f}s"
            )

        # ── Stage 4: Cross-Document Federation ─────────────────────────
        if (
            len(per_paper_nodes) > 1
            and self.cfg.get("cross_document", {}).get("enabled", True)
        ):
            logger.info("Running cross-document federation…")
            cross_edges = self.crossdoc.build(per_paper_nodes)
            master_store.add_edges(cross_edges)
            logger.info(f"  → {len(cross_edges)} SAME_CONCEPT edges added")

        total_time = time.time() - t0
        stats = master_store.stats()
        logger.info(
            f"=== KG complete in {total_time:.1f}s ===\n"
            f"  Nodes: {stats['total_nodes']}  "
            f"Edges: {stats['total_edges']}\n"
            f"  Node types: {stats['node_types']}\n"
            f"  Edge types: {stats['edge_types']}"
        )

        if save_path:
            master_store.save(save_path)

        return master_store
