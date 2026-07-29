"""
knowledge_graph/structural_graph.py
-------------------------------------
Phase 1-A: Structural Graph

Builds the document hierarchy graph from parsed elements:
  Paper → Section → Subsection → Paragraph/TextBlock
  Paper → Figure / Table / Equation

Edges created: CONTAINS, HAS_CAPTION
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from .schema import (
    EdgeType, KGEdge, KGNode, NodeType, ParsedElement
)

logger = logging.getLogger(__name__)


# ─── Section heading patterns ───────────────────────────────────────────────
_SECTION_RE = re.compile(
    r"^(\d+\.?\d*\.?\d*)\s+(.+)$"   # "2.1 Methods" or "3 Results"
)
_ABSTRACT_TITLES = {"abstract", "introduction", "conclusion",
                    "related work", "background", "discussion",
                    "acknowledgment", "acknowledgements", "references"}


def _is_section_title(text: str) -> bool:
    text = text.strip()
    if _SECTION_RE.match(text):
        return True
    if text.lower() in _ABSTRACT_TITLES:
        return True
    # All-caps short text is often a section header
    if text.isupper() and len(text) < 60:
        return True
    return False


def _section_depth(text: str) -> int:
    """Return rough nesting depth based on numbering (1.2.3 → 3)."""
    m = _SECTION_RE.match(text.strip())
    if m:
        return m.group(1).count(".") + 1
    return 1


class StructuralGraphBuilder:
    """
    Converts a flat list of ParsedElements into KG nodes and edges
    representing the document's structural hierarchy.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build(
        self,
        paper_id: str,
        elements: List[ParsedElement],
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        Returns (nodes, edges) for the structural graph of one paper.
        """
        nodes: List[KGNode] = []
        edges: List[KGEdge] = []

        # 1. Paper root node
        paper_node = KGNode(
            node_id=f"{paper_id}__paper",
            node_type=NodeType.PAPER,
            paper_id=paper_id,
            page=0,
            content=paper_id,
        )
        nodes.append(paper_node)

        # 2. Walk elements — track current section context
        current_section_id: str | None = None
        section_counter = 0
        text_counter = 0
        fig_counter = 0
        table_counter = 0
        eq_counter = 0

        for elem in elements:
            # ── Skip authors / low-confidence ────────────────────────
            if elem.element_type == "author":
                continue
            if elem.confidence < 0.4:
                continue

            # ── Section/title detection ──────────────────────────────
            if elem.element_type in ("title", "text") and _is_section_title(elem.content):
                section_counter += 1
                sec_id = f"{paper_id}__sec_{section_counter}"
                sec_node = KGNode(
                    node_id=sec_id,
                    node_type=NodeType.TEXT_BLOCK,
                    paper_id=paper_id,
                    page=elem.page,
                    content=elem.content,
                    metadata={
                        "section_title": elem.content,
                        "depth": _section_depth(elem.content),
                        "is_section": True,
                    },
                )
                nodes.append(sec_node)

                # Paper CONTAINS section
                edges.append(KGEdge(
                    src_id=paper_node.node_id,
                    dst_id=sec_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=0.3,
                    metadata={"method": "rule"},
                ))
                current_section_id = sec_id
                continue

            # ── Regular text block ───────────────────────────────────
            if elem.element_type in ("title", "text"):
                if not elem.content.strip():
                    continue
                text_counter += 1
                tb_id = f"{paper_id}__text_{text_counter}"
                tb_node = KGNode(
                    node_id=tb_id,
                    node_type=NodeType.TEXT_BLOCK,
                    paper_id=paper_id,
                    page=elem.page,
                    content=elem.content,
                    metadata={"confidence": elem.confidence},
                )
                nodes.append(tb_node)
                parent_id = current_section_id or paper_node.node_id
                edges.append(KGEdge(
                    src_id=parent_id,
                    dst_id=tb_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=0.3,
                    metadata={"method": "rule"},
                ))

            # ── Figure ──────────────────────────────────────────────
            elif elem.element_type == "figure":
                fig_counter += 1
                fig_id = f"{paper_id}__fig_{fig_counter}"
                fig_node = KGNode(
                    node_id=fig_id,
                    node_type=NodeType.FIGURE,
                    paper_id=paper_id,
                    page=elem.page,
                    content=elem.caption or "",
                    image_path=elem.image_path,
                    metadata={
                        "confidence": elem.confidence,
                        "has_image": elem.image_path is not None,
                    },
                )
                nodes.append(fig_node)

                parent_id = current_section_id or paper_node.node_id
                edges.append(KGEdge(
                    src_id=parent_id,
                    dst_id=fig_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=0.3,
                    metadata={"method": "rule"},
                ))

                # Caption as separate TextBlock
                if elem.caption and elem.caption.strip():
                    cap_id = f"{paper_id}__cap_fig_{fig_counter}"
                    cap_node = KGNode(
                        node_id=cap_id,
                        node_type=NodeType.TEXT_BLOCK,
                        paper_id=paper_id,
                        page=elem.page,
                        content=elem.caption,
                        metadata={"is_caption": True},
                    )
                    nodes.append(cap_node)
                    edges.append(KGEdge(
                        src_id=fig_id,
                        dst_id=cap_id,
                        edge_type=EdgeType.HAS_CAPTION,
                        weight=0.5,
                        metadata={"method": "rule"},
                    ))

            # ── Table ────────────────────────────────────────────────
            elif elem.element_type == "table":
                table_counter += 1
                tab_id = f"{paper_id}__tab_{table_counter}"
                tab_node = KGNode(
                    node_id=tab_id,
                    node_type=NodeType.TABLE,
                    paper_id=paper_id,
                    page=elem.page,
                    content=elem.content,
                    image_path=elem.image_path,
                    metadata={"confidence": elem.confidence},
                )
                nodes.append(tab_node)

                parent_id = current_section_id or paper_node.node_id
                edges.append(KGEdge(
                    src_id=parent_id,
                    dst_id=tab_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=0.3,
                    metadata={"method": "rule"},
                ))

                if elem.caption and elem.caption.strip():
                    cap_id = f"{paper_id}__cap_tab_{table_counter}"
                    cap_node = KGNode(
                        node_id=cap_id,
                        node_type=NodeType.TEXT_BLOCK,
                        paper_id=paper_id,
                        page=elem.page,
                        content=elem.caption,
                        metadata={"is_caption": True},
                    )
                    nodes.append(cap_node)
                    edges.append(KGEdge(
                        src_id=tab_id,
                        dst_id=cap_id,
                        edge_type=EdgeType.HAS_CAPTION,
                        weight=0.5,
                        metadata={"method": "rule"},
                    ))

        logger.debug(
            f"[{paper_id}] structural: {len(nodes)} nodes, {len(edges)} edges"
        )
        return nodes, edges
