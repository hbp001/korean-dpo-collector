"""
models/eval_dataset.py
-----------------------
평가 데이터셋 로더

지원 포맷:
  1. 자체 구축 JSONL  (Self-Play로 생성된 데이터)
  2. KG 기반 자동 생성  (kg.json에서 직접 생성)

JSONL 포맷 (한 줄 = 하나의 샘플):
{
    "question_id":   "q_000001",
    "question":      "What does Figure 1 show?",
    "question_type": "VQA",          // VQA | MCQ | Reasoning | InstructionFollowing
    "ground_truths": ["system architecture", "three-layer system"],
    "image_paths":   ["/workspace/data/paper_A/images/fig1.jpg"],
    "context":       "Figure 1 shows the system architecture...",
    "paper_id":      "paper_A",
    "difficulty":    1
}
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question_id:    str
    question:       str
    question_type:  str
    ground_truths:  List[str]
    image_paths:    List[str]
    context:        str
    paper_id:       str
    difficulty:     int = 1
    metadata:       Dict[str, Any] = field(default_factory=dict)


class EvalDataset:
    """평가 데이터셋 로더 및 관리 클래스."""

    def __init__(self, samples: List[EvalSample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]

    def filter_by_type(self, question_type: str) -> "EvalDataset":
        return EvalDataset([s for s in self._samples if s.question_type == question_type])

    def filter_by_difficulty(self, difficulty: int) -> "EvalDataset":
        return EvalDataset([s for s in self._samples if s.difficulty == difficulty])

    def sample(self, n: int, seed: int = 42) -> "EvalDataset":
        rng = random.Random(seed)
        return EvalDataset(rng.sample(self._samples, min(n, len(self._samples))))

    def stats(self) -> Dict[str, Any]:
        type_counts = {}
        diff_counts = {}
        for s in self._samples:
            type_counts[s.question_type] = type_counts.get(s.question_type, 0) + 1
            diff_counts[s.difficulty]    = diff_counts.get(s.difficulty, 0) + 1
        return {
            "total":           len(self._samples),
            "by_type":         type_counts,
            "by_difficulty":   diff_counts,
            "papers":          len({s.paper_id for s in self._samples}),
            "with_image":      sum(1 for s in self._samples if s.image_paths),
        }

    # ─── 저장 / 로드 ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for s in self._samples:
                record = {
                    "question_id":   s.question_id,
                    "question":      s.question,
                    "question_type": s.question_type,
                    "ground_truths": s.ground_truths,
                    "image_paths":   s.image_paths,
                    "context":       s.context,
                    "paper_id":      s.paper_id,
                    "difficulty":    s.difficulty,
                    "metadata":      s.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"EvalDataset saved → {p} ({len(self._samples)} samples)")

    @classmethod
    def from_jsonl(cls, path: str) -> "EvalDataset":
        """JSONL 파일에서 로드."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append(EvalSample(
                    question_id=d.get("question_id", f"q_{len(samples):06d}"),
                    question=d.get("question", d.get("question_text", "")),
                    question_type=d.get("question_type", "VQA"),
                    ground_truths=d.get("ground_truths", [d.get("answer", "")]),
                    image_paths=d.get("image_paths", []),
                    context=d.get("context", d.get("context_text", "")),
                    paper_id=d.get("paper_id", "unknown"),
                    difficulty=d.get("difficulty", 1),
                    metadata=d.get("metadata", {}),
                ))
        logger.info(f"EvalDataset loaded ← {path} ({len(samples)} samples)")
        return cls(samples)

    @classmethod
    def from_self_play_output(cls, jsonl_path: str) -> "EvalDataset":
        """
        Self-Play가 생성한 training_data_*.jsonl에서 평가 데이터 로드.
        question/chosen 쌍을 ground_truth로 사용합니다.
        """
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append(EvalSample(
                    question_id=f"sp_{len(samples):06d}",
                    question=d.get("question", ""),
                    question_type=d.get("type", "VQA"),
                    ground_truths=[d.get("chosen", "")],
                    image_paths=d.get("images", []),
                    context="",
                    paper_id="self_play",
                    difficulty=1,
                    metadata={"reward": d.get("reward", 0.0)},
                ))
        logger.info(
            f"EvalDataset from Self-Play ← {jsonl_path} ({len(samples)} samples)"
        )
        return cls(samples)

    @classmethod
    def from_kg(
        cls,
        kg_path: str,
        n_samples: int = 100,
        seed: int = 42,
    ) -> "EvalDataset":
        """
        kg.json에서 Figure/Table 노드를 기반으로
        간단한 평가 질문을 자동 생성합니다.
        """
        from knowledge_graph.graph_store import NetworkXGraphStore
        from knowledge_graph.schema import NodeType

        store = NetworkXGraphStore()
        store.load(kg_path)

        rng = random.Random(seed)
        samples = []
        counter = 0

        # Figure 노드 기반 질문 생성
        fig_nodes = store.get_nodes_by_type(NodeType.FIGURE)
        tab_nodes = store.get_nodes_by_type(NodeType.TABLE)
        candidates = fig_nodes + tab_nodes
        rng.shuffle(candidates)

        for node in candidates[:n_samples]:
            if not node.content or len(node.content.strip()) < 10:
                continue
            counter += 1
            q_text = (
                f"What does this {node.node_type.value.lower()} show or describe?"
                if not node.content.startswith("[VLM]")
                else f"Based on the description, what is the main finding shown?"
            )
            samples.append(EvalSample(
                question_id=f"kg_{counter:06d}",
                question=q_text,
                question_type="VQA",
                ground_truths=[node.content[:200]],
                image_paths=[node.image_path] if node.image_path else [],
                context=node.content,
                paper_id=node.paper_id,
                difficulty=1,
                metadata={"node_id": node.node_id, "auto_generated": True},
            ))

        logger.info(f"EvalDataset from KG: {len(samples)} auto-generated samples")
        return cls(samples)
