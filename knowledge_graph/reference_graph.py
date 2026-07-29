"""
knowledge_graph/reference_graph.py
------------------------------------
Phase 1-B: Reference Graph

Parses cross-reference expressions in text blocks
("As shown in Figure 3", "Table 2 summarizes", etc.)
and creates REFERENCES edges between TextBlock nodes and
Figure / Table / Equation nodes.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from .schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)


# ─── Regex patterns for in-text references ─────────────────────────────────
_FIGURE_REFS = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?\s*|fig(?:ure)?s?\.?\s*)(\d+[a-zA-Z]?)\b",
    re.IGNORECASE,
)
_TABLE_REFS = re.compile(
    r"\b(?:Tab(?:le)?s?\.?\s*)(\d+[a-zA-Z]?)\b",
    re.IGNORECASE,
)
_EQ_REFS = re.compile(
    r"\b(?:Eq(?:uation)?s?\.?\s*\(?|eq\.\s*\(?)(\d+)\b",
    re.IGNORECASE,
)


# ─── 한국어 상호참조 패턴 (config에서 language: "ko"일 때 추가 적용) ────────
# 영어 패턴은 그대로 두고 **추가로** 매칭한다 → 영어 KG 결과는 변하지 않는다.
_KO_FIGURE_REFS = re.compile(r"(?:그림|사진|도표|도)\s*(\d+[a-zA-Z]?)")
_KO_TABLE_REFS  = re.compile(r"(?:표|테이블)\s*(\d+[a-zA-Z]?)")
_KO_EQ_REFS     = re.compile(r"(?:식|수식|방정식)\s*\(?(\d+)\)?")


def _extract_refs(text: str, include_korean: bool = False) -> Dict[str, List[str]]:
    """Return dict of reference type → list of referenced numbers."""
    refs = {
        "figure": _FIGURE_REFS.findall(text),
        "table":  _TABLE_REFS.findall(text),
        "eq":     _EQ_REFS.findall(text),
    }
    if include_korean:
        refs["figure"] = refs["figure"] + _KO_FIGURE_REFS.findall(text)
        refs["table"]  = refs["table"]  + _KO_TABLE_REFS.findall(text)
        refs["eq"]     = refs["eq"]     + _KO_EQ_REFS.findall(text)
    return refs


class ReferenceGraphBuilder:
    """
    Adds REFERENCES edges to an existing node set by scanning text nodes
    for cross-reference expressions.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # language: "ko"면 한국어 참조 패턴(그림/표/식)을 추가로 매칭한다.
        # 기본값 "en"에서는 기존 동작과 완전히 동일하다.
        self._include_korean = str(cfg.get("language", "en")).lower().startswith("ko")

    def build(
        self,
        paper_id: str,
        nodes: List[KGNode],
    ) -> List[KGEdge]:
        """
        Given the already-built structural nodes, detect in-text
        references and return new REFERENCES edges.
        """
        # Build lookup maps: sequential index → node_id
        fig_map:  Dict[str, str] = {}  # "1" -> node_id of fig_1
        tab_map:  Dict[str, str] = {}
        eq_map:   Dict[str, str] = {}
        text_nodes: List[KGNode] = []

        fig_seq = 0
        tab_seq = 0
        eq_seq  = 0

        for node in nodes:
            if node.node_type == NodeType.FIGURE:
                fig_seq += 1
                fig_map[str(fig_seq)] = node.node_id
            elif node.node_type == NodeType.TABLE:
                tab_seq += 1
                tab_map[str(tab_seq)] = node.node_id
            elif node.node_type == NodeType.EQUATION:
                eq_seq += 1
                eq_map[str(eq_seq)] = node.node_id
            elif node.node_type == NodeType.TEXT_BLOCK:
                text_nodes.append(node)

        edges: List[KGEdge] = []
        ref_count = 0

        for tb in text_nodes:
            if not tb.content:
                continue
            refs = _extract_refs(tb.content, self._include_korean)

            for fig_num in refs["figure"]:
                # Normalise: e.g. "3a" → "3"
                num = re.sub(r"[a-zA-Z]$", "", fig_num)
                if num in fig_map:
                    edges.append(KGEdge(
                        src_id=tb.node_id,
                        dst_id=fig_map[num],
                        edge_type=EdgeType.REFERENCES,
                        weight=0.8,
                        confidence=0.9,
                        metadata={
                            "method": "rule",
                            "evidence": f"ref Figure {fig_num}",
                        },
                    ))
                    ref_count += 1

            for tab_num in refs["table"]:
                num = re.sub(r"[a-zA-Z]$", "", tab_num)
                if num in tab_map:
                    edges.append(KGEdge(
                        src_id=tb.node_id,
                        dst_id=tab_map[num],
                        edge_type=EdgeType.REFERENCES,
                        weight=0.8,
                        confidence=0.9,
                        metadata={
                            "method": "rule",
                            "evidence": f"ref Table {tab_num}",
                        },
                    ))
                    ref_count += 1

            for eq_num in refs["eq"]:
                if eq_num in eq_map:
                    edges.append(KGEdge(
                        src_id=tb.node_id,
                        dst_id=eq_map[eq_num],
                        edge_type=EdgeType.REFERENCES,
                        weight=0.8,
                        confidence=0.9,
                        metadata={
                            "method": "rule",
                            "evidence": f"ref Eq {eq_num}",
                        },
                    ))
                    ref_count += 1

        logger.debug(
            f"[{paper_id}] reference: {ref_count} REFERENCES edges added"
        )
        return edges
