"""
knowledge_graph/vlm_enricher.py
---------------------------------
VLM 기반 KG 노드/엣지 보강 모듈

세 가지 역할을 수행합니다:

① Figure/Table 노드 내용 이해
   - 이미지를 Qwen3-VL에 입력 → 시각 요소 설명 생성
   - KGNode.content와 metadata["vlm_description"] 보강

② 시각-텍스트 관계 분류
   - (이미지 + 주변 텍스트) → VLM이 엣지 타입 결정
   - ILLUSTRATES / SUPPORTS / QUANTIFIES / COMPARES / CONTRADICTS

③ Concept / Claim 노드 추출
   - 텍스트 블록에서 VLM이 핵심 주장/개념 식별
   - 새로운 Concept/Claim 노드와 DEFINES/SUPPORTS 엣지 생성

config.yaml:
    knowledge_graph:
      vlm_enrichment:
        enabled: true
        figure_description: true      # ①
        visual_text_relation: true    # ②
        concept_extraction: true      # ③
        max_figures_per_paper: 20     # GPU 메모리 절약
        max_text_nodes_for_concept: 30
        batch_size: 4
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)


# ─── VLM 관계 분류용 프롬프트 ─────────────────────────────────────────────

_RELATION_PROMPT = """You are analyzing a scientific paper.

[Figure/Table content]:
{visual_description}

[Nearby text]:
{text_content}

What is the relationship between this visual element and the text?
Choose ONE from: ILLUSTRATES, SUPPORTS, QUANTIFIES, COMPARES, CONTRADICTS, DERIVES_FROM

Answer with ONLY the relation type (one word).
"""

_FIGURE_DESC_PROMPT = """Describe this scientific figure/table concisely in 2-3 sentences.
Focus on:
- What type of visualization it is (graph, chart, diagram, table, etc.)
- What data or concept it shows
- Key findings or values visible

Be specific and technical."""

_CONCEPT_EXTRACT_PROMPT = """Extract the main scientific claims and concepts from this text.

Text: {text}

List up to 3 key claims or concepts as short phrases (5-15 words each).
Format: one per line, no numbering, no explanation.
Only output the phrases."""


# ─── 엣지 타입 매핑 ────────────────────────────────────────────────────────

_VLM_TO_EDGE: Dict[str, EdgeType] = {
    "ILLUSTRATES":  EdgeType.ILLUSTRATES,
    "SUPPORTS":     EdgeType.SUPPORTS,
    "QUANTIFIES":   EdgeType.QUANTIFIES,
    "COMPARES":     EdgeType.COMPARES,
    "CONTRADICTS":  EdgeType.CONTRADICTS,
    "DERIVES_FROM": EdgeType.DERIVES_FROM,
}


class VLMEnricher:
    """
    Qwen3-VL을 사용하여 KG 노드와 엣지를 시각적 맥락으로 보강합니다.
    """

    def __init__(self, cfg: dict, vllm_cfg: dict):
        self.cfg = cfg
        self.vllm_cfg = vllm_cfg
        self.enabled = cfg.get("enabled", False)
        self.do_figure_desc    = cfg.get("figure_description", True)
        self.do_visual_rel     = cfg.get("visual_text_relation", True)
        self.do_concept        = cfg.get("concept_extraction", True)
        self.max_figures       = cfg.get("max_figures_per_paper", 20)
        self.max_text_nodes    = cfg.get("max_text_nodes_for_concept", 30)
        self.batch_size        = cfg.get("batch_size", 4)
        self._model     = None
        self._processor = None

    # ─── 모델 로드 (lazy) ─────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = self.vllm_cfg.get("name", "Qwen/Qwen3-VL-4B-Instruct")
            dtype      = self.vllm_cfg.get("dtype", "auto")
            logger.info(f"Loading VLM: {model_name}")

            load_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": "auto",
            }
            if self.vllm_cfg.get("use_flash_attention", False):
                load_kwargs["attn_implementation"] = "flash_attention_2"
                load_kwargs["torch_dtype"] = torch.bfloat16

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(model_name)
            logger.info("VLM loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"VLM load failed: {e}. VLM enrichment skipped.")
            return False

    # ─── VLM 추론 공통 함수 ───────────────────────────────────────────────

    def _infer(self, messages: List[Dict]) -> str:
        """messages 형식으로 VLM 추론 수행, 텍스트 응답 반환."""
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            max_tokens = self.vllm_cfg.get("max_new_tokens", 256)
            generated = self._model.generate(**inputs, max_new_tokens=max_tokens)
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], generated)
            ]
            result = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return result[0].strip() if result else ""
        except Exception as e:
            logger.debug(f"VLM inference error: {e}")
            return ""

    def _build_image_message(self, image_path: str, prompt: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text",  "text": prompt},
            ],
        }]

    def _build_text_message(self, prompt: str) -> List[Dict]:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # ─── ① Figure/Table 노드 내용 이해 ───────────────────────────────────

    def enrich_visual_nodes(self, nodes: List[KGNode]) -> List[KGNode]:
        """
        이미지가 있는 Figure/Table 노드에 VLM 설명을 추가합니다.
        nodes를 직접 수정(in-place)하고 반환합니다.
        """
        visual_nodes = [
            n for n in nodes
            if n.node_type in (NodeType.FIGURE, NodeType.TABLE)
            and n.image_path
            and Path(n.image_path).exists()
        ][: self.max_figures]

        if not visual_nodes:
            return nodes

        logger.info(f"  [VLM] Describing {len(visual_nodes)} visual nodes…")

        for node in visual_nodes:
            messages = self._build_image_message(
                node.image_path, _FIGURE_DESC_PROMPT
            )
            description = self._infer(messages)
            if description:
                # 기존 content(캡션)와 VLM 설명을 합쳐서 저장
                node.metadata["vlm_description"] = description
                # content를 캡션 + VLM 설명으로 보강 (임베딩에 활용됨)
                if node.content:
                    node.content = f"{node.content}\n[VLM]: {description}"
                else:
                    node.content = f"[VLM]: {description}"
                logger.debug(f"    → {node.node_id}: {description[:80]}…")

        return nodes

    # ─── ② 시각-텍스트 관계 분류 ─────────────────────────────────────────

    def classify_visual_text_relations(
        self,
        nodes: List[KGNode],
        existing_edges: List[KGEdge],
    ) -> List[KGEdge]:
        """
        Figure/Table 노드와 인접 TextBlock 사이의 관계를
        VLM으로 분류하여 새로운 엣지를 생성합니다.

        - 이미 REFERENCES 엣지가 있는 (TextBlock → Figure) 쌍을 우선 처리
        - 없으면 같은 페이지의 TextBlock과 Figure를 페어링
        """
        node_map: Dict[str, KGNode] = {n.node_id: n for n in nodes}

        # 이미 REFERENCES 엣지로 연결된 (text, visual) 쌍 수집
        pairs: List[Tuple[KGNode, KGNode]] = []
        for edge in existing_edges:
            if edge.edge_type != EdgeType.REFERENCES:
                continue
            src = node_map.get(edge.src_id)
            dst = node_map.get(edge.dst_id)
            if src and dst and dst.node_type in (NodeType.FIGURE, NodeType.TABLE):
                pairs.append((src, dst))

        # REFERENCES 쌍이 없으면 같은 페이지 기준으로 페어링
        if not pairs:
            visual_nodes = [
                n for n in nodes
                if n.node_type in (NodeType.FIGURE, NodeType.TABLE)
                and n.image_path
                and Path(n.image_path).exists()
            ][: self.max_figures]

            text_nodes = [
                n for n in nodes
                if n.node_type == NodeType.TEXT_BLOCK
                and len(n.content) > 50
            ]

            for vis in visual_nodes:
                # 같은 페이지 텍스트 우선, 없으면 인접 페이지
                same_page = [t for t in text_nodes if t.page == vis.page]
                nearby    = same_page or [
                    t for t in text_nodes if abs(t.page - vis.page) <= 1
                ]
                for txt in nearby[:3]:   # 노드당 최대 3개 텍스트와 쌍
                    pairs.append((txt, vis))

        if not pairs:
            return []

        logger.info(f"  [VLM] Classifying {len(pairs)} visual-text relations…")
        new_edges: List[KGEdge] = []

        for text_node, vis_node in pairs:
            if not (vis_node.image_path and Path(vis_node.image_path).exists()):
                continue

            # VLM 설명이 이미 있으면 재사용
            vis_desc = vis_node.metadata.get("vlm_description", vis_node.content)

            prompt = _RELATION_PROMPT.format(
                visual_description=vis_desc[:300],
                text_content=text_node.content[:400],
            )
            messages = self._build_image_message(vis_node.image_path, prompt)
            response = self._infer(messages).upper().strip()

            # 응답에서 엣지 타입 파싱
            etype = None
            for key in _VLM_TO_EDGE:
                if key in response:
                    etype = _VLM_TO_EDGE[key]
                    break

            if etype is None:
                etype = EdgeType.ILLUSTRATES   # fallback

            new_edges.append(KGEdge(
                src_id=text_node.node_id,
                dst_id=vis_node.node_id,
                edge_type=etype,
                weight=0.85,
                confidence=0.8,
                metadata={
                    "method": "vlm_relation",
                    "vlm_response": response[:50],
                },
            ))
            logger.debug(
                f"    {text_node.node_id} →[{etype.value}]→ {vis_node.node_id}"
            )

        logger.info(f"  [VLM] {len(new_edges)} visual-text relation edges added")
        return new_edges

    # ─── ③ Concept / Claim 노드 추출 ─────────────────────────────────────

    def extract_concepts(
        self,
        paper_id: str,
        nodes: List[KGNode],
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        텍스트 블록에서 VLM이 핵심 개념/주장을 추출하여
        Concept/Claim 노드와 DEFINES/SUPPORTS 엣지를 생성합니다.
        """
        text_nodes = [
            n for n in nodes
            if n.node_type == NodeType.TEXT_BLOCK
            and len(n.content.strip()) > 100
            and not n.metadata.get("is_caption")
            and not n.metadata.get("is_section")
        ][: self.max_text_nodes]

        if not text_nodes:
            return [], []

        logger.info(f"  [VLM] Extracting concepts from {len(text_nodes)} text nodes…")

        new_nodes: List[KGNode] = []
        new_edges: List[KGEdge] = []
        concept_counter = 0

        for text_node in text_nodes:
            prompt = _CONCEPT_EXTRACT_PROMPT.format(
                text=text_node.content[:600]
            )
            messages = self._build_text_message(prompt)
            response = self._infer(messages)

            if not response:
                continue

            # 응답을 줄 단위로 파싱
            concepts = [
                line.strip(" -•*")
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 5
            ][:3]

            for concept_text in concepts:
                concept_counter += 1
                concept_id = f"{paper_id}__concept_{concept_counter}"

                concept_node = KGNode(
                    node_id=concept_id,
                    node_type=NodeType.CONCEPT,
                    paper_id=paper_id,
                    page=text_node.page,
                    content=concept_text,
                    metadata={
                        "method": "vlm_extraction",
                        "source_node": text_node.node_id,
                    },
                )
                new_nodes.append(concept_node)

                # TextBlock → Concept: DEFINES 엣지
                new_edges.append(KGEdge(
                    src_id=text_node.node_id,
                    dst_id=concept_id,
                    edge_type=EdgeType.DEFINES,
                    weight=0.9,
                    confidence=0.75,
                    metadata={"method": "vlm_extraction"},
                ))

        logger.info(
            f"  [VLM] {len(new_nodes)} concept nodes, "
            f"{len(new_edges)} DEFINES edges added"
        )
        return new_nodes, new_edges

    # ─── 통합 실행 ────────────────────────────────────────────────────────

    def enrich(
        self,
        paper_id: str,
        nodes: List[KGNode],
        existing_edges: List[KGEdge],
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        세 단계를 순서대로 실행합니다.
        Returns: (추가 노드, 추가 엣지) — 기존 것은 포함하지 않음
        """
        if not self.enabled:
            return [], []

        if not self._load_model():
            return [], []

        added_nodes: List[KGNode] = []
        added_edges: List[KGEdge] = []

        # ① 이미지 노드 내용 이해 (in-place 수정)
        if self.do_figure_desc:
            self.enrich_visual_nodes(nodes)

        # ② 시각-텍스트 관계 분류
        if self.do_visual_rel:
            vis_edges = self.classify_visual_text_relations(nodes, existing_edges)
            added_edges.extend(vis_edges)

        # ③ 개념/주장 추출
        if self.do_concept:
            c_nodes, c_edges = self.extract_concepts(paper_id, nodes)
            added_nodes.extend(c_nodes)
            added_edges.extend(c_edges)

        return added_nodes, added_edges
