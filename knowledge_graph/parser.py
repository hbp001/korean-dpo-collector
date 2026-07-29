"""
knowledge_graph/parser.py
--------------------------
Parses parsing_paddle.json into a list of ParsedElement objects.
Designed to be extended for new parsing formats with zero code change
to downstream KG builders — just swap the parser.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import ParsedElement

logger = logging.getLogger(__name__)


class PaddleJSONParser:
    """
    Reads a parsing_paddle.json file and yields ParsedElement objects.

    Directory layout expected:
        data/
          <paper_name>/
            parsing_paddle.json
            images/
              img_in_*.jpg
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def list_papers(self) -> List[str]:
        """Return all paper folder names that contain a parsing_paddle.json."""
        papers = []
        for item in sorted(self.data_root.iterdir()):
            if item.is_dir() and (item / "parsing_paddle.json").exists():
                papers.append(item.name)
        logger.info(f"Found {len(papers)} papers under {self.data_root}")
        return papers

    def parse_paper(self, paper_id: str) -> List[ParsedElement]:
        """Parse a single paper and return all its elements."""
        json_path = self.data_root / paper_id / "parsing_paddle.json"
        if not json_path.exists():
            logger.warning(f"JSON not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        elements: List[ParsedElement] = []

        for page_data in data.get("pages", []):
            page_num = page_data.get("page_number", -1)

            # ── Title elements ──────────────────────────────────────────
            for elem in page_data.get("title_elements", []):
                text = elem.get("text", "").strip()
                if not text:
                    continue
                elements.append(ParsedElement(
                    paper_id=paper_id,
                    page=page_num,
                    element_type="title",
                    content=text,
                    image_path=None,
                    confidence=elem.get("confidence", 1.0),
                    caption=None,
                ))

            # ── Author elements ─────────────────────────────────────────
            for elem in page_data.get("author_elements", []):
                text = elem.get("text", "").strip()
                if not text:
                    continue
                elements.append(ParsedElement(
                    paper_id=paper_id,
                    page=page_num,
                    element_type="author",
                    content=text,
                    image_path=None,
                    confidence=elem.get("confidence", 1.0),
                    caption=None,
                ))

            # ── Text elements ───────────────────────────────────────────
            for elem in page_data.get("text_elements", []):
                text = elem.get("text", "").strip()
                if not text:
                    continue
                elements.append(ParsedElement(
                    paper_id=paper_id,
                    page=page_num,
                    element_type="text",
                    content=text,
                    image_path=None,
                    confidence=elem.get("confidence", 1.0),
                    caption=None,
                ))

            # ── Visual elements (Figures) ────────────────────────────────
            for idx, elem in enumerate(page_data.get("visual_elements", [])):
                img_rel = elem.get("image_path", "")
                img_abs = self._resolve_image(paper_id, img_rel) if img_rel else None
                caption = elem.get("caption", "")
                elements.append(ParsedElement(
                    paper_id=paper_id,
                    page=page_num,
                    element_type="figure",
                    content=caption or "",
                    image_path=img_abs,
                    confidence=elem.get("confidence", 1.0),
                    caption=caption,
                    metadata={"element_index": idx},
                ))

            # ── Table elements ──────────────────────────────────────────
            for idx, elem in enumerate(page_data.get("table_elements", [])):
                table_text = elem.get("text", "").strip()
                img_rel = elem.get("image_path", "")
                img_abs = self._resolve_image(paper_id, img_rel) if img_rel else None
                caption = elem.get("caption", "")
                elements.append(ParsedElement(
                    paper_id=paper_id,
                    page=page_num,
                    element_type="table",
                    content=table_text or caption or "",
                    image_path=img_abs,
                    confidence=elem.get("confidence", 1.0),
                    caption=caption,
                    metadata={"element_index": idx},
                ))

        logger.debug(f"  [{paper_id}] parsed {len(elements)} elements across "
                     f"{len(data.get('pages', []))} pages")
        return elements

    def parse_all(self) -> Dict[str, List[ParsedElement]]:
        """Parse every paper and return {paper_id: [elements]}."""
        result = {}
        for paper_id in self.list_papers():
            result[paper_id] = self.parse_paper(paper_id)
        return result

    def _resolve_image(self, paper_id: str, rel_path: str) -> Optional[str]:
        """Convert a relative image path from JSON to an absolute path."""
        candidate = self.data_root / paper_id / rel_path
        if candidate.exists():
            return str(candidate)
        # Try without leading path components
        filename = Path(rel_path).name
        candidate2 = self.data_root / paper_id / "images" / filename
        if candidate2.exists():
            return str(candidate2)
        logger.debug(f"  Image not found: {rel_path} in {paper_id}")
        return None
