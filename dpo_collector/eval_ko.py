"""
dpo_collector/eval_ko.py
-------------------------
고정 한국어 평가셋 구축·로드 + ANLS / Accuracy / F1 평가.

CLAUDE_dpo.md §1-4 (평가셋은 고정) 을 지키기 위한 세 가지 장치:

  1. **논문 단위 split 동결** — 한국어 논문을 `held_out`(평가용)과 `collect`(수집용)로
     한 번 나눈 뒤 `splits.json` 에 저장한다. 이후에는 이 파일이 진실이며, 논문이
     추가돼도 기존 배정은 바뀌지 않는다(새 논문만 수집 풀에 붙는다).
     → 평가 문항이 수집 풀에서 나오는 일이 구조적으로 불가능해진다.
  2. **draft → 확정 2단계** — KG gold answer 로 만든 후보는 `eval_ko_draft.jsonl` 에
     쌓이고, 사람이 확인한 것만 `eval_ko.jsonl` 로 확정된다. 확정본은 동결 대상.
  3. **동결 체크섬** — 확정본을 만들 때 내용 해시를 `splits.json` 에 남겨, 이후
     평가셋이 바뀌면 경고한다(체크포인트 간 지표 비교가 무의미해지는 것을 막는다).

평가 지표는 코어 `models.eval_metrics.compute_all_metrics` 를 그대로 재사용한다
(ANLS·Accuracy 는 한국어에서 정상 동작. F1 은 어절 단위 토큰화라 조사 차이만큼
보수적으로 나온다 — 절대값보다 체크포인트 간 상대 변화를 보는 용도).
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .kg_bridge import KGBridge, KGPath, clean_content
from .question_gen import josa as _josa
from .store import KST, _file_lock

logger = logging.getLogger(__name__)

#: 평가 문항 ID 접두사
_QID_PREFIX = "eko_"

#: 문항으로 채택할 gold answer 의 최소/최대 길이 (너무 짧으면 채점 불가, 길면 서술형)
_GOLD_MIN, _GOLD_MAX = 2, 200

# 캡션 앞머리의 라벨. gold answer 가 "Fig.3 정류전류 중 AC성분" 처럼 시작하면
# 모델이 내용만 맞혀도 라벨 때문에 점수가 깎이므로 정답에서 떼어낸다.
_CAPTION_LABEL_RE = re.compile(
    r"^\s*[<〈\[【(]?\s*"
    r"(?P<kind>Fig(?:ure)?\.?|Table|그\s?림|표|사진|도표|수식|식)"
    r"\s*\.?\s*(?P<num>\d+(?:[-.]\d+)?)?\s*"
    r"[>〉\]】)]?\s*[.:、,]?\s*",
    re.IGNORECASE,
)

#: 라벨 종류 → 자료 종류 ("figure" | "table")
_LABEL_KIND = {
    "fig": "figure", "fig.": "figure", "figure": "figure", "figure.": "figure",
    "그림": "figure", "그 림": "figure", "사진": "figure", "도표": "figure",
    "table": "table", "표": "table",
    "수식": "equation", "식": "equation",
}


def _label_kind(text: str) -> Optional[str]:
    """캡션 앞머리 라벨이 가리키는 자료 종류. 라벨이 없으면 None."""
    m = _CAPTION_LABEL_RE.match(text or "")
    if not m:
        return None
    return _LABEL_KIND.get(m.group("kind").strip().lower())


#: 정답이 내용을 담고 있는지 판정할 내용어 (한글 2자 이상 / 영문 4자 이상 / 숫자)
_CONTENT_WORD_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{4,}|\d+(?:\.\d+)?")
_HANGUL_RE = re.compile(r"[가-힣]")


def is_usable_gold(text: str, require_hangul: bool = True) -> bool:
    """
    평가 정답으로 쓸 만한 문자열인지.

    - 내용어가 하나도 없으면 채점이 불가능하다 (예: 캡션이 "(a)" 뿐인 경우).
    - `require_hangul` 이면 한글이 포함되어야 한다. 한국어 논문이라도 도표 캡션이
      영어로만 달린 경우가 많은데, 한국어 평가셋의 정답으로는 부적절하다.
    """
    if not _CONTENT_WORD_RE.search(text or ""):
        return False
    if require_hangul and not _HANGUL_RE.search(text or ""):
        return False
    return True


def strip_caption_label(text: str) -> str:
    """
    캡션에서 앞머리 라벨을 떼어낸 내용만 남긴다.
    라벨을 떼고 나서 내용이 없으면(라벨뿐인 캡션) 원문을 그대로 돌려준다.
    """
    stripped = _CAPTION_LABEL_RE.sub("", text or "", count=1).strip()
    return stripped if stripped else (text or "").strip()


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


# ─── 평가 문항 (코어 EvalDataset JSONL 스키마와 호환) ──────────────────────

@dataclass
class EvalItem:
    """`models.eval_dataset.EvalSample` 과 동일한 필드 구성 (§5.3)."""

    question_id:   str
    question:      str
    question_type: str = "VQA"
    ground_truths: List[str] = field(default_factory=list)
    image_paths:   List[str] = field(default_factory=list)
    context:       str = ""
    paper_id:      str = ""
    difficulty:    int = 1
    #: draft 단계에서만 쓰는 보조 필드 — 확정 시 제거된다
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_metadata:
            d.pop("metadata", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalItem":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ─── 논문 split 동결 ───────────────────────────────────────────────────────

class PaperSplit:
    """
    한국어 논문을 평가용(held-out) / 수집용으로 나누고 그 배정을 동결한다.

        split = PaperSplit("dpo_collector/outputs/splits.json")
        split.ensure(all_korean_papers, n_held_out=40, seed=42)
        split.collect_papers   # 수집 풀
        split.held_out_papers  # 평가 전용 — 수집에 절대 쓰지 않는다
    """

    def __init__(self, splits_json: str):
        self.path = Path(splits_json)
        self._data: Optional[Dict[str, Any]] = None

    # ── 로드/저장 ─────────────────────────────────────────────────────────

    def _read(self) -> Dict[str, Any]:
        if self._data is not None:
            return self._data
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"[EvalKo] splits.json 파싱 실패: {e}")
                self._data = {}
        else:
            self._data = {}
        self._data.setdefault("held_out", [])
        self._data.setdefault("collect", [])
        self._data.setdefault("created_at", None)
        self._data.setdefault("eval_frozen_at", None)
        self._data.setdefault("eval_sha256", None)
        return self._data

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._data = data

    # ── 분할 ──────────────────────────────────────────────────────────────

    def ensure(
        self,
        papers: Sequence[str],
        n_held_out: int = 40,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        split 이 없으면 새로 만들고, 있으면 **기존 배정을 유지한 채** 새 논문만
        수집 풀에 추가한다. held_out 은 절대 바뀌지 않는다(동결).
        """
        with _file_lock(self.path):
            data = self._read()
            known = set(data["held_out"]) | set(data["collect"])
            new_papers = [p for p in papers if p not in known]

            if not data["held_out"] and not data["collect"]:
                # 최초 분할 — 재현 가능하도록 정렬 후 고정 시드로 섞는다
                pool = sorted(papers)
                rng = random.Random(seed)
                rng.shuffle(pool)
                n = min(int(n_held_out), max(0, len(pool) - 1))
                data["held_out"] = sorted(pool[:n])
                data["collect"] = sorted(pool[n:])
                data["created_at"] = _now_iso()
                data["seed"] = seed
                logger.info(
                    f"[EvalKo] 논문 split 생성 — 평가 {len(data['held_out'])}편 / "
                    f"수집 {len(data['collect'])}편 (seed={seed})"
                )
            elif new_papers:
                # 이미 동결된 split 에 논문이 추가된 경우: 수집 풀에만 붙인다
                data["collect"] = sorted(set(data["collect"]) | set(new_papers))
                logger.info(
                    f"[EvalKo] 새 논문 {len(new_papers)}편을 수집 풀에 추가 "
                    "(평가셋 배정은 변경하지 않음)"
                )

            self._write(data)
            return data

    @property
    def held_out_papers(self) -> List[str]:
        return list(self._read()["held_out"])

    @property
    def collect_papers(self) -> List[str]:
        return list(self._read()["collect"])

    @property
    def is_initialized(self) -> bool:
        d = self._read()
        return bool(d["held_out"] or d["collect"])

    def check_overlap(self, collect_paper_ids: Sequence[str]) -> List[str]:
        """
        **수집에 쓰려는** 논문 목록이 평가 전용(held-out)을 침범했는지 검사한다.
        반환값이 비어 있어야 정상이다.

        ⚠ 평가 문항의 논문 목록을 넣으면 당연히 전부 반환된다(평가 문항은 held-out 에서
          나오는 것이 정상). 이 함수의 입력은 **수집 풀** 쪽 목록이어야 한다.
        """
        held = set(self.held_out_papers)
        return sorted(set(collect_paper_ids) & held)

    # ── 평가셋 동결 ───────────────────────────────────────────────────────

    def freeze_eval(self, eval_path: str) -> str:
        """확정된 평가셋의 해시를 기록한다. 이후 변경을 감지하는 기준."""
        digest = _sha256_file(eval_path)
        with _file_lock(self.path):
            data = self._read()
            data["eval_sha256"] = digest
            data["eval_frozen_at"] = _now_iso()
            data["eval_path"] = str(eval_path)
            self._write(data)
        logger.info(f"[EvalKo] 평가셋 동결 — sha256={digest[:16]}… ({eval_path})")
        return digest

    def verify_eval_frozen(self, eval_path: str) -> Tuple[bool, str]:
        """
        평가셋이 동결 시점과 동일한지 확인한다.

        Returns: (동일 여부, 설명)
        """
        data = self._read()
        recorded = data.get("eval_sha256")
        if not recorded:
            return False, "평가셋이 아직 동결되지 않았습니다 (freeze_eval 미실행)."
        if not Path(eval_path).exists():
            return False, f"평가셋 파일이 없습니다: {eval_path}"
        current = _sha256_file(eval_path)
        if current != recorded:
            return False, (
                "⚠️ 평가셋이 동결 이후 변경되었습니다 — "
                "이전 체크포인트 지표와 직접 비교하면 안 됩니다."
            )
        return True, "평가셋이 동결 상태 그대로입니다."


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── 평가셋 구축 / 로드 ────────────────────────────────────────────────────

class EvalSetKo:
    """
    고정 한국어 평가셋 관리.

        ev = EvalSetKo(eval_jsonl="…/eval_ko.jsonl", draft_jsonl="…/eval_ko_draft.jsonl",
                       split=PaperSplit("…/splits.json"))
        ev.build_draft(bridge, n=120)     # KG gold answer 기반 후보 생성
        # (사람이 draft 를 검토·수정)
        ev.confirm(question_ids=[...])    # 확정본 생성 + 동결
        ds = ev.load()                    # 평가 시 로드
    """

    def __init__(
        self,
        eval_jsonl: str,
        draft_jsonl: Optional[str] = None,
        split: Optional[PaperSplit] = None,
        require_hangul_gold: bool = True,
    ):
        self.eval_path = Path(eval_jsonl)
        self.draft_path = (
            Path(draft_jsonl) if draft_jsonl
            else self.eval_path.with_name(self.eval_path.stem + "_draft.jsonl")
        )
        self.split = split
        #: 정답에 한글이 있어야 채택할지 (한국어 평가셋 기본값)
        self.require_hangul_gold = require_hangul_gold

    # ── 후보(draft) 생성 ──────────────────────────────────────────────────

    def build_draft(
        self,
        bridge: KGBridge,
        n: int = 120,
        max_hops: int = 2,
        require_image: bool = True,
        seed: int = 42,
        overwrite: bool = False,
    ) -> List[EvalItem]:
        """
        held-out 논문의 KG 에서 gold answer 를 뽑아 평가 문항 후보를 만든다.

        질문 문장은 **모델 없이** 엣지 타입별 한국어 템플릿으로 만든다 — 평가셋 문항을
        학습 대상 모델로 생성하면 그 모델에 유리하게 편향되기 때문이다.
        사람이 draft 를 다듬는 것을 전제로 한 초안이다.
        """
        if self.draft_path.exists() and not overwrite:
            raise FileExistsError(
                f"draft 가 이미 있습니다: {self.draft_path} "
                "(덮어쓰려면 overwrite=True)"
            )
        if self.split is None or not self.split.is_initialized:
            raise RuntimeError(
                "논문 split 이 준비되지 않았습니다 — PaperSplit.ensure() 를 먼저 호출하세요."
            )

        held = set(self.split.held_out_papers)
        graph_papers = set(bridge.papers_in_graph())
        usable = sorted(held & graph_papers)
        if not usable:
            raise RuntimeError(
                f"KG 안에 held-out 논문이 없습니다. "
                f"(held-out {len(held)}편 중 KG 포함 0편) — "
                "평가용 논문을 포함해 KG 를 구축해야 합니다."
            )
        logger.info(
            f"[EvalKo] draft 생성 — held-out {len(usable)}편에서 최대 {n}문항"
        )

        rng = random.Random(seed)
        items: List[EvalItem] = []
        # 중복 판정은 **KG 경로(node_ids)** 기준이다.
        # 질문 문구는 템플릿이라 서로 겹치지만, 대상 이미지와 정답이 다르면 별개 문항이다.
        seen_paths: set = set()

        # 논문을 고르게 돌면서 문항을 모은다 (한 논문에 치우치지 않게)
        attempts = 0
        while len(items) < n and attempts < n * 30:
            attempts += 1
            pid = usable[attempts % len(usable)]
            paths = bridge.sample_paths(
                n=1, paper_id=pid, max_hops=max_hops,
                prefer_visual=True, require_image=require_image,
                seed=rng.randrange(1 << 30),
            )
            if not paths:
                continue
            key = tuple(paths[0].node_ids)
            if key in seen_paths:
                continue
            item = self._path_to_item(paths[0], len(items) + 1)
            if item is None:
                continue
            seen_paths.add(key)
            items.append(item)

        self._write_jsonl(self.draft_path, items, include_metadata=True)
        logger.info(
            f"[EvalKo] draft {len(items)}문항 저장 → {self.draft_path}\n"
            "  ── 사람 검토 가이드 (확정 전 필수) ──\n"
            "  1) 정답이 그림/표의 실제 내용과 맞는지 확인 (캡션 오매칭이 남아 있을 수 있음)\n"
            "  2) `ground_truths` 에 **허용 표현을 추가**하세요. 이게 지표에 가장 크게 영향을 줍니다.\n"
            "     예: [\"캐시 사이즈 변화에 따른 히트율\", \"캐시 크기에 따른 히트율\", \"캐시 크기별 히트율\"]\n"
            "     정답이 1개뿐이면 의미가 맞아도 표현이 다르다는 이유로 0점 처리됩니다.\n"
            "  3) 답을 특정할 수 없는 문항(정답이 지나치게 일반적)은 제외\n"
            "  4) 검토 후 `confirm(question_ids=[…])` 으로 50~100문항만 확정하세요."
        )
        return items

    #: 엣지 타입별 평가 문항 템플릿 (모델을 쓰지 않는다 — 편향 방지).
    #: `{target}` 에는 "그림 3" 처럼 대상 자료를 가리키는 말이 들어간다.
    _Q_TEMPLATES = {
        "HAS_CAPTION": "{target_eun} 무엇을 나타내는가?",
        "QUANTIFIES":  "{target}에 제시된 수치는 얼마인가?",
        "DEFINES":     "{target}에서 설명하는 개념은 무엇으로 정의되는가?",
        "REFERENCES":  "본문이 참조하는 {target_eun} 무엇을 보여주는가?",
        "COMPARES":    "{target}에서 비교되는 두 대상의 차이는 무엇인가?",
        "SUPPORTS":    "{target_i} 뒷받침하는 주장은 무엇인가?",
        "SINGLE":      "{target_eun} 무엇을 나타내는가?",
    }

    #: 노드 타입 → 질문에 쓸 한국어 표기 / 라벨 종류
    _TARGET_KO = {"Figure": "그림", "Table": "표", "Equation": "수식"}
    _TARGET_KIND = {"Figure": "figure", "Table": "table", "Equation": "equation"}

    @staticmethod
    def _target_node(path: KGPath):
        """문항이 가리키는 대상 노드 (이미지가 함께 제공되는 시각 노드 우선)."""
        return next(
            (n for n in path.nodes
             if n.node_type.value in ("Figure", "Table", "Equation")),
            path.nodes[0],
        )

    def _path_to_item(self, path: KGPath, idx: int) -> Optional[EvalItem]:
        """
        KG 경로 → 평가 문항 후보. 아래 중 하나라도 걸리면 None(문항으로 쓰지 않음):
          - gold answer 가 너무 짧거나 길다
          - 대상 노드 종류와 캡션 라벨이 어긋난다
            (예: Figure 노드인데 캡션이 "표 12-1 …" — 파싱 단계의 캡션 오매칭)

        질문에 자료 번호는 넣지 않는다. KG 의 순번(`__fig_4`)은 **등장 순서**라
        문서에 인쇄된 번호("그림 3")와 어긋나는 경우가 있어, 번호를 넣으면 질문이
        엉뚱한 대상을 가리키게 된다. 이미지가 함께 제공되므로 "이 그림"으로 충분하다.
        """
        raw_gold = " ".join((path.gold_answer or "").split())
        if not raw_gold:
            return None

        target = self._target_node(path)
        target_kind = self._TARGET_KIND.get(target.node_type.value)

        # 캡션 라벨이 대상 종류와 다르면 캡션이 잘못 붙은 것 → 버린다
        label = _label_kind(raw_gold)
        if label and target_kind and label != target_kind:
            return None

        gold = strip_caption_label(raw_gold)
        if not (_GOLD_MIN <= len(gold) <= _GOLD_MAX):
            return None
        if not is_usable_gold(gold, require_hangul=self.require_hangul_gold):
            return None

        # 정답 후보를 여러 개 둔다 — `compute_all_metrics` 는 리스트 중 최고 점수를 쓴다.
        # 라벨을 뗀 내용이 기본이고, 라벨이 붙은 캡션 원문도 허용한다
        # (모델이 "그림 3 캐시 히트율" 처럼 답해도 내용이 맞으면 인정).
        golds = [gold]
        if raw_gold != gold:
            golds.append(raw_gold)

        ko = self._TARGET_KO.get(target.node_type.value, "자료")
        primary = path.edge_types[0] if path.edge_types else "SINGLE"
        question = self._Q_TEMPLATES.get(primary, self._Q_TEMPLATES["SINGLE"]).format(
            target=f"이 {ko}",
            target_eun=f"이 {ko}" + _josa(ko, "은", "는"),
            target_i=f"이 {ko}" + _josa(ko, "이", "가"),
        )

        return EvalItem(
            question_id=f"{_QID_PREFIX}{idx:06d}",
            question=question,
            question_type=path.question_type,
            ground_truths=golds,
            image_paths=list(path.image_paths),
            context=path.context[:800],
            paper_id=path.paper_id,
            difficulty=path.difficulty,
            metadata={
                "source": "kg_draft",
                "edge_types": path.edge_types,
                "node_ids": path.node_ids,
                "needs_review": True,
            },
        )

    # ── 확정 ──────────────────────────────────────────────────────────────

    def confirm(
        self,
        question_ids: Optional[Sequence[str]] = None,
        freeze: bool = True,
    ) -> List[EvalItem]:
        """
        draft 중 채택할 문항만 확정본으로 옮기고 동결한다.

        Args:
            question_ids: 채택할 문항 ID. None 이면 draft 전체
                          (§5.3 은 사람이 확인한 50~100문항을 권장).
        """
        drafts = self.load_draft()
        if not drafts:
            raise RuntimeError(f"draft 가 비어 있습니다: {self.draft_path}")

        if question_ids is None:
            picked = drafts
            logger.warning(
                "[EvalKo] question_ids 미지정 — draft 전체를 확정합니다. "
                "평가 신뢰도를 위해 사람이 검토한 문항만 고르는 것을 권장합니다."
            )
        else:
            want = set(question_ids)
            picked = [d for d in drafts if d.question_id in want]
            missing = want - {d.question_id for d in picked}
            if missing:
                logger.warning(f"[EvalKo] draft 에 없는 ID 무시: {sorted(missing)[:5]}…")

        # 수집 풀과 겹치지 않는지 최종 확인 (§1-4)
        if self.split is not None:
            collect = set(self.split.collect_papers)
            leaked = sorted({p.paper_id for p in picked} & collect)
            if leaked:
                raise RuntimeError(
                    f"평가 문항이 수집 풀 논문에서 나왔습니다: {leaked[:5]} — "
                    "평가셋과 수집 풀은 반드시 분리되어야 합니다."
                )

        self._write_jsonl(self.eval_path, picked, include_metadata=False)
        logger.info(f"[EvalKo] 평가셋 확정 {len(picked)}문항 → {self.eval_path}")

        if freeze and self.split is not None:
            self.split.freeze_eval(str(self.eval_path))
        return picked

    # ── 로드 ──────────────────────────────────────────────────────────────

    def load(self, verify_frozen: bool = True) -> List[EvalItem]:
        """확정 평가셋 로드. 동결 이후 변경되었으면 경고한다."""
        items = self._read_jsonl(self.eval_path)
        if verify_frozen and self.split is not None and items:
            ok, msg = self.split.verify_eval_frozen(str(self.eval_path))
            if not ok:
                logger.warning(f"[EvalKo] {msg}")
        return items

    def load_draft(self) -> List[EvalItem]:
        return self._read_jsonl(self.draft_path)

    def as_core_dataset(self):
        """
        코어 `models.eval_dataset.EvalDataset` 인스턴스로 변환한다.
        (코어의 필터/샘플링 유틸을 그대로 쓰고 싶을 때)
        """
        from models.eval_dataset import EvalDataset

        return EvalDataset.from_jsonl(str(self.eval_path))

    @property
    def exists(self) -> bool:
        return self.eval_path.exists()

    def stats(self) -> Dict[str, Any]:
        items = self.load(verify_frozen=False)
        by_type: Dict[str, int] = {}
        by_diff: Dict[int, int] = {}
        for it in items:
            by_type[it.question_type] = by_type.get(it.question_type, 0) + 1
            by_diff[it.difficulty] = by_diff.get(it.difficulty, 0) + 1
        frozen_msg = ""
        if self.split is not None:
            _, frozen_msg = self.split.verify_eval_frozen(str(self.eval_path))
        return {
            "n_items": len(items),
            "n_papers": len({it.paper_id for it in items}),
            "with_image": sum(1 for it in items if it.image_paths),
            "by_question_type": by_type,
            "by_difficulty": dict(sorted(by_diff.items())),
            "path": str(self.eval_path),
            "frozen": frozen_msg,
        }

    # ── JSONL 입출력 ──────────────────────────────────────────────────────

    @staticmethod
    def _write_jsonl(
        path: Path, items: Sequence[EvalItem], include_metadata: bool
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(
                    json.dumps(it.to_dict(include_metadata), ensure_ascii=False) + "\n"
                )

    @staticmethod
    def _read_jsonl(path: Path) -> List[EvalItem]:
        if not path.exists():
            return []
        out: List[EvalItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(EvalItem.from_dict(json.loads(line)))
                except Exception as e:
                    logger.warning(f"[EvalKo] {path}:{lineno} 파싱 실패: {e}")
        return out


# ─── 평가 실행 ─────────────────────────────────────────────────────────────

def evaluate(
    backend,
    items: Sequence[EvalItem],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    prompt_template: Optional[str] = None,
    language: str = "한국어",
    progress_cb=None,
) -> Dict[str, Any]:
    """
    주어진 백엔드로 평가셋을 풀고 ANLS / Accuracy / F1 을 계산한다.

    평가는 **greedy(temperature=0)** 로 돌린다 — 샘플링하면 체크포인트 간 비교에
    노이즈가 섞인다. 답변도 짧게 받는다(단답 채점 기준이므로).

    Returns:
        `compute_all_metrics` 결과 + {"n_items", "predictions"}
    """
    from models.eval_metrics import compute_all_metrics

    if not items:
        return {"n_samples": 0, "anls": 0.0, "accuracy": 0.0, "f1": 0.0,
                "n_items": 0, "predictions": []}

    # 정답(gold)이 도표 캡션에서 나온 **짧은 명사구**이므로 모델도 같은 형식으로
    # 답하게 유도해야 한다. 서술형 문장으로 답하면 내용이 맞아도 ANLS(문자열 유사도)가
    # 0 에 가깝게 나와 체크포인트 간 변화를 읽을 수 없다(실측: 서술형 유도 시 ANLS 0.0).
    template = prompt_template or (
        "이미지를 보고 다음 질문에 {language}로 답하세요.\n"
        "답은 논문의 도표 제목처럼 짧은 명사구 하나로만 쓰세요.\n"
        "- 문장으로 설명하지 마세요 (\"~를 나타냅니다\", \"~입니다\" 금지)\n"
        "- \"그림 1\", \"표 2\" 같은 번호 표기는 빼세요\n"
        "- 예: 캐시 크기에 따른 히트율 변화\n\n"
        "질문: {question}\n답:"
    )

    preds: List[str] = []
    for i, it in enumerate(items):
        prompt = template.format(language=language, question=it.question)
        try:
            out = backend.infer(
                it.image_paths, prompt,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[EvalKo] {it.question_id} 추론 실패: {e}")
            out = ""
        preds.append((out or "").strip())
        if progress_cb:
            progress_cb((i + 1) / len(items), f"평가 {i + 1}/{len(items)}")

    result = compute_all_metrics(
        preds,
        [it.ground_truths for it in items],
        [it.question_type for it in items],
    )
    result["n_items"] = len(items)
    result["predictions"] = preds
    logger.info(
        f"[EvalKo] 평가 완료 — n={len(items)} "
        f"ANLS={result['anls']:.4f} Accuracy={result['accuracy']:.4f} F1={result['f1']:.4f}"
    )
    return result


def metrics_only(result: Dict[str, Any]) -> Dict[str, float]:
    """`training_history.json::eval` 에 넣을 지표만 추린다."""
    return {
        k: round(float(result.get(k, 0.0)), 4)
        for k in ("anls", "accuracy", "f1")
    }


# ─── 설정 파일 연동 ────────────────────────────────────────────────────────

def from_config(
    config_path: str = "dpo_collector/config_dpo.yaml",
) -> Tuple[EvalSetKo, PaperSplit]:
    """`config_dpo.yaml` 의 `paths` / `eval` 섹션으로 평가셋·split 을 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths", {})
    split = PaperSplit(
        paths.get("splits_json", "dpo_collector/outputs/splits.json")
    )
    evalset = EvalSetKo(
        eval_jsonl=paths.get("eval_jsonl", "dpo_collector/outputs/eval_ko.jsonl"),
        draft_jsonl=paths.get("eval_draft_jsonl"),
        split=split,
        require_hangul_gold=bool(
            cfg.get("eval", {}).get("require_hangul_gold", True)
        ),
    )
    return evalset, split


# ─── CLI ───────────────────────────────────────────────────────────────────

def _main() -> int:  # pragma: no cover - 수동 운영용
    import argparse

    ap = argparse.ArgumentParser(
        description="한국어 고정 평가셋 split / draft 생성 / 확정 / 통계"
    )
    ap.add_argument("--config", default="dpo_collector/config_dpo.yaml")
    ap.add_argument(
        "command",
        choices=["split", "draft", "confirm", "stats"],
        help="split=논문 분할, draft=문항 후보 생성, confirm=확정+동결, stats=통계",
    )
    ap.add_argument("--n", type=int, default=None, help="draft 문항 수")
    ap.add_argument("--held-out", type=int, default=None, help="평가 전용 논문 수")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    ev_cfg = cfg.get("eval", {})
    evalset, split = from_config(args.config)

    if args.command == "stats":
        print(json.dumps(evalset.stats(), ensure_ascii=False, indent=2))
        print(json.dumps(
            {"held_out": len(split.held_out_papers), "collect": len(split.collect_papers)},
            ensure_ascii=False, indent=2,
        ))
        return 0

    from .kg_bridge import from_config as kg_from_config

    bridge = kg_from_config(args.config)

    if args.command == "split":
        papers = bridge.list_papers(language=cfg.get("language", "ko"))
        data = split.ensure(
            papers,
            n_held_out=args.held_out or int(ev_cfg.get("n_held_out_papers", 40)),
            seed=int(ev_cfg.get("split_seed", 42)),
        )
        print(f"평가 전용(held-out): {len(data['held_out'])}편")
        print(f"수집 풀(collect)   : {len(data['collect'])}편")
        return 0

    if args.command == "draft":
        bridge.build_or_load()
        items = evalset.build_draft(
            bridge,
            n=args.n or int(ev_cfg.get("target_size", 120)),
            require_image=bool(ev_cfg.get("require_image", True)),
            seed=int(ev_cfg.get("split_seed", 42)),
            overwrite=args.overwrite,
        )
        print(f"draft {len(items)}문항 → {evalset.draft_path}")
        print("사람이 검토·수정한 뒤 `confirm` 을 실행하세요.")
        return 0

    if args.command == "confirm":
        picked = evalset.confirm()
        print(f"확정 {len(picked)}문항 → {evalset.eval_path} (동결 완료)")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
