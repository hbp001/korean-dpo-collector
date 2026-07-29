"""
knowledge_graph/schema.py
--------------------------
Typed dataclasses for Knowledge Graph nodes and edges.
These are the canonical data structures shared across all KG phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────

class NodeType(str, Enum):
    TEXT_BLOCK = "TextBlock"
    FIGURE     = "Figure"
    TABLE      = "Table"
    EQUATION   = "Equation"
    CONCEPT    = "Concept"
    CLAIM      = "Claim"
    PAPER      = "Paper"       # root node for each document


class EdgeType(str, Enum):
    CONTAINS      = "CONTAINS"
    HAS_CAPTION   = "HAS_CAPTION"
    REFERENCES    = "REFERENCES"
    ILLUSTRATES   = "ILLUSTRATES"
    QUANTIFIES    = "QUANTIFIES"
    DEFINES       = "DEFINES"
    SUPPORTS      = "SUPPORTS"
    CONTRADICTS   = "CONTRADICTS"
    DERIVES_FROM  = "DERIVES_FROM"
    COMPARES      = "COMPARES"
    SAME_CONCEPT  = "SAME_CONCEPT"   # cross-document coreference


# ─────────────────────────────────────────
# Node schema
# ─────────────────────────────────────────

@dataclass
class KGNode:
    """A single node in the Knowledge Graph."""
    node_id:   str                      # unique id: "{paper_id}_{type}_{idx}"
    node_type: NodeType
    paper_id:  str                      # source paper folder name
    page:      int   = -1               # page number (-1 = unknown)
    content:   str   = ""               # text or caption
    image_path: Optional[str] = None    # path to image file (Figure/Table)
    metadata:  Dict[str, Any] = field(default_factory=dict)
    # metadata keys used:
    #   section_title, subsection_title, confidence,
    #   bbox (x1,y1,x2,y2), element_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id":    self.node_id,
            "node_type":  self.node_type.value,
            "paper_id":   self.paper_id,
            "page":       self.page,
            "content":    self.content,
            "image_path": self.image_path,
            "metadata":   self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KGNode:
        d["node_type"] = NodeType(d["node_type"])
        return cls(**d)


# ─────────────────────────────────────────
# Edge schema
# ─────────────────────────────────────────

@dataclass
class KGEdge:
    """A directed edge in the Knowledge Graph."""
    src_id:    str
    dst_id:    str
    edge_type: EdgeType
    weight:    float = 1.0
    confidence: float = 1.0           # 0-1, lower for inferred edges
    metadata:  Dict[str, Any] = field(default_factory=dict)
    # metadata keys:
    #   evidence_text, method (rule | embedding | vllm | crossdoc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_id":     self.src_id,
            "dst_id":     self.dst_id,
            "edge_type":  self.edge_type.value,
            "weight":     self.weight,
            "confidence": self.confidence,
            "metadata":   self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KGEdge:
        d["edge_type"] = EdgeType(d["edge_type"])
        return cls(**d)


# ─────────────────────────────────────────
# Parsed document element (from JSON)
# ─────────────────────────────────────────

@dataclass
class ParsedElement:
    """Intermediate representation parsed from parsing_paddle.json."""
    paper_id:     str
    page:         int
    element_type: str              # "text" | "figure" | "table" | "title" | "author"
    content:      str
    image_path:   Optional[str]
    confidence:   float
    caption:      Optional[str]
    metadata:     Dict[str, Any] = field(default_factory=dict)
