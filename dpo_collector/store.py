"""
dpo_collector/store.py
-----------------------
DPO 선호 페어 저장소.

두 벌로 저장한다 (CLAUDE_dpo.md §5):
  ① **학습용** `dpo_pairs.jsonl` — 전체 메타(KG 출처, 후보 전체, 난이도 …) 포함, append-only.
  ② **공유용** `export/dpo_share.jsonl` — 타 기관 제공용 표준 포맷. 이미지는 상대경로로 복사.

설계 원칙:
  - **append-only.** 수정/삭제는 파일을 다시 쓰지 않고 tombstone 레코드를 덧붙인다
    (`status="deleted"`). 읽을 때 pair_id 별 마지막 레코드가 유효 상태다.
  - **파일 락.** Gradio 는 동시 요청을 처리하므로 pair_id 발번과 append 가 겹칠 수 있다.
    `fcntl.flock` 으로 직렬화한다.
  - **저장 전 검증.** `chosen != rejected`, 양쪽 non-empty, 이미지 경로 유효,
    `lang == config.language` (§11 체크리스트).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 저장 타임스탬프 기준 시간대 (§5.1 예시가 +09:00)
KST = timezone(timedelta(hours=9))

#: pair_id 접두사 — `kdpo_000001`
_PAIR_ID_PREFIX = "kdpo_"
_PAIR_ID_RE = re.compile(rf"^{_PAIR_ID_PREFIX}(\d+)$")

#: 지원하는 공유 포맷
EXPORT_FORMATS = ("rlaif_v", "hf_conversational")

#: 질문 유형 (§5.1)
QUESTION_TYPES = ("VQA", "Reasoning", "MCQ", "InstructionFollowing")


class PairValidationError(ValueError):
    """저장 전 검증 실패."""


# ─── 파일 락 ───────────────────────────────────────────────────────────────

@contextmanager
def _file_lock(path: Path):
    """
    같은 jsonl 을 동시에 건드리는 요청을 직렬화한다.
    락 파일은 대상 파일 옆에 `.lock` 으로 만든다 (대상 파일 자체를 잠그면
    append 모드 재오픈과 얽혀 다루기 번거롭다).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception as e:  # pragma: no cover - 비 POSIX 환경
            logger.debug(f"[Store] 파일 락 미지원, 무시하고 진행: {e}")
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


# ─── 페어 스키마 (§5.1) ────────────────────────────────────────────────────

@dataclass
class DPOPair:
    """학습용 레코드 1건. jsonl 한 줄에 대응한다."""

    pair_id:         str
    question:        str
    chosen:          str
    rejected:        str
    paper_id:        str = ""
    created_at:      str = ""
    lang:            str = "ko"
    question_source: str = "user"          # "auto" | "user"
    question_type:   str = "VQA"
    difficulty:      int = 1
    kg_provenance:   Dict[str, Any] = field(default_factory=dict)
    gold_answer:     str = ""
    image_paths:     List[str] = field(default_factory=list)
    candidates:      List[str] = field(default_factory=list)
    chosen_idx:      int = -1
    rejected_idx:    int = -1
    annotator:       str = "user"
    model_version:   str = "base"          # 후보를 생성한 모델
    notes:           str = ""
    status:          str = "active"        # "active" | "deleted"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DPOPair":
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        pair = cls(**{k: v for k, v in d.items() if k in known})
        if extra:
            # 스키마가 나중에 확장돼도 기존 레코드를 잃지 않도록 notes 에 보존
            pair.notes = (pair.notes + " " if pair.notes else "") + json.dumps(
                extra, ensure_ascii=False
            )
        return pair

    @property
    def is_active(self) -> bool:
        return self.status == "active"


# ─── 저장소 ────────────────────────────────────────────────────────────────

class DPOPairStore:
    """
    DPO 페어 append-only 저장소 + 공유 포맷 export.

        store = DPOPairStore("dpo_collector/outputs/dpo_pairs.jsonl", language="ko")
        pair  = store.add(question="...", candidates=[...], chosen_idx=1, rejected_idx=0, ...)
        store.export(fmt="rlaif_v")
    """

    def __init__(
        self,
        pairs_jsonl: str,
        export_dir: Optional[str] = None,
        language: str = "ko",
        export_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.path = Path(pairs_jsonl)
        self.export_dir = Path(export_dir) if export_dir else self.path.parent / "export"
        self.language = language
        self.export_cfg = export_cfg or {}

    # ── 읽기 ──────────────────────────────────────────────────────────────

    def _iter_raw(self) -> Iterable[Dict[str, Any]]:
        """파일의 모든 레코드를 기록 순서대로 읽는다 (tombstone 포함)."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"[Store] {self.path}:{lineno} 파싱 실패, 건너뜀: {e}")

    def all(self, include_deleted: bool = False) -> List[DPOPair]:
        """
        pair_id 별 **마지막** 레코드를 유효 상태로 보고 목록을 만든다.
        (append-only 이므로 뒤에 쓰인 레코드가 앞의 것을 덮어쓴다)
        """
        latest: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for rec in self._iter_raw():
            pid = rec.get("pair_id")
            if not pid:
                continue
            if pid not in latest:
                order.append(pid)
            latest[pid] = rec

        out: List[DPOPair] = []
        for pid in order:
            pair = DPOPair.from_dict(latest[pid])
            if pair.is_active or include_deleted:
                out.append(pair)
        return out

    def get(self, pair_id: str) -> Optional[DPOPair]:
        for p in self.all(include_deleted=True):
            if p.pair_id == pair_id:
                return p
        return None

    def count(self, include_deleted: bool = False) -> int:
        return len(self.all(include_deleted=include_deleted))

    def counts_by_type(self) -> Dict[str, int]:
        """질문 유형별 유효 페어 수 — 학습 트리거의 유형 편향 가드레일에 쓰인다."""
        counts = {t: 0 for t in QUESTION_TYPES}
        for p in self.all():
            counts[p.question_type] = counts.get(p.question_type, 0) + 1
        return counts

    def stats(self) -> Dict[str, Any]:
        pairs = self.all()
        by_source: Dict[str, int] = {}
        by_model: Dict[str, int] = {}
        by_difficulty: Dict[int, int] = {}
        n_with_image = 0
        for p in pairs:
            by_source[p.question_source] = by_source.get(p.question_source, 0) + 1
            by_model[p.model_version] = by_model.get(p.model_version, 0) + 1
            by_difficulty[p.difficulty] = by_difficulty.get(p.difficulty, 0) + 1
            if p.image_paths:
                n_with_image += 1
        return {
            "total_pairs":     len(pairs),
            "deleted_pairs":   self.count(include_deleted=True) - len(pairs),
            "with_image":      n_with_image,
            "by_question_type": self.counts_by_type(),
            "by_source":       by_source,
            "by_model_version": by_model,
            "by_difficulty":   dict(sorted(by_difficulty.items())),
            "path":            str(self.path),
        }

    # ── 쓰기 ──────────────────────────────────────────────────────────────

    def _next_pair_id(self) -> str:
        """기존 최대 번호 + 1. 락 안에서 호출해야 한다."""
        max_n = 0
        for rec in self._iter_raw():
            m = _PAIR_ID_RE.match(str(rec.get("pair_id", "")))
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"{_PAIR_ID_PREFIX}{max_n + 1:06d}"

    def _append_raw(self, record: Dict[str, Any]) -> None:
        """검증 없이 한 줄 추가 (락 안에서 호출)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add(
        self,
        question: str,
        candidates: List[str],
        chosen_idx: int,
        rejected_idx: int,
        paper_id: str = "",
        image_paths: Optional[List[str]] = None,
        question_source: str = "user",
        question_type: str = "VQA",
        difficulty: int = 1,
        kg_provenance: Optional[Dict[str, Any]] = None,
        gold_answer: str = "",
        annotator: str = "user",
        model_version: str = "base",
        notes: str = "",
        lang: Optional[str] = None,
        allow_duplicate: bool = False,
    ) -> DPOPair:
        """
        후보 답변 목록에서 chosen/rejected 를 골라 페어 1건을 저장한다.

        Args:
            allow_duplicate: False(기본)면 (질문, chosen, rejected)가 완전히 같은
                유효 페어가 이미 있을 때 거부한다. UI 에서 저장 버튼을 두 번 누르거나
                요청이 재시도될 때 같은 페어가 중복 적재되는 것을 막는다.

        Raises:
            PairValidationError: 검증 실패 시 (저장하지 않는다).
        """
        image_paths = list(image_paths or [])
        pair = DPOPair(
            pair_id="",                 # 락 안에서 발번
            question=(question or "").strip(),
            chosen="",
            rejected="",
            paper_id=paper_id,
            lang=lang or self.language,
            question_source=question_source,
            question_type=question_type,
            difficulty=int(difficulty),
            kg_provenance=kg_provenance or {},
            gold_answer=gold_answer,
            image_paths=image_paths,
            candidates=[c for c in candidates],
            chosen_idx=int(chosen_idx),
            rejected_idx=int(rejected_idx),
            annotator=annotator,
            model_version=model_version,
            notes=notes,
        )

        # 인덱스 → 실제 답변 텍스트
        n = len(pair.candidates)
        for label, idx in (("chosen_idx", pair.chosen_idx), ("rejected_idx", pair.rejected_idx)):
            if not (0 <= idx < n):
                raise PairValidationError(
                    f"{label}={idx} 가 후보 범위(0~{n - 1})를 벗어났습니다."
                )
        pair.chosen = (pair.candidates[pair.chosen_idx] or "").strip()
        pair.rejected = (pair.candidates[pair.rejected_idx] or "").strip()

        self.validate(pair)

        with _file_lock(self.path):
            if not allow_duplicate:
                dup = self._find_duplicate(pair)
                if dup:
                    raise PairValidationError(
                        f"동일한 페어가 이미 저장되어 있습니다 ({dup}). "
                        "다시 저장하려면 질문이나 선택을 바꾸세요."
                    )
            pair.pair_id = self._next_pair_id()
            pair.created_at = _now_iso()
            self._append_raw(pair.to_dict())

        logger.info(
            f"[Store] 페어 저장: {pair.pair_id} "
            f"(type={pair.question_type}, L{pair.difficulty}, "
            f"images={len(pair.image_paths)}, model={pair.model_version})"
        )
        return pair

    @staticmethod
    def _dedup_key(pair: DPOPair) -> Tuple[str, str, str]:
        """중복 판정 키 — 공백 차이는 무시한다."""
        norm = lambda s: " ".join((s or "").split())   # noqa: E731
        return norm(pair.question), norm(pair.chosen), norm(pair.rejected)

    def _find_duplicate(self, pair: DPOPair) -> Optional[str]:
        """같은 (질문, chosen, rejected)를 가진 유효 페어의 pair_id. 없으면 None."""
        key = self._dedup_key(pair)
        for existing in self.all():
            if self._dedup_key(existing) == key:
                return existing.pair_id
        return None

    def validate(self, pair: DPOPair) -> None:
        """§11 체크리스트에 대응하는 저장 전 검증."""
        if not pair.question:
            raise PairValidationError("질문이 비어 있습니다.")
        if not pair.chosen or not pair.rejected:
            raise PairValidationError("chosen/rejected 가 비어 있습니다.")
        if pair.chosen == pair.rejected:
            raise PairValidationError(
                "chosen 과 rejected 가 동일합니다 — 선호 신호가 없어 학습에 쓸 수 없습니다."
            )
        if pair.lang != self.language:
            raise PairValidationError(
                f"lang='{pair.lang}' 이 설정 언어 '{self.language}' 와 다릅니다."
            )
        if pair.question_type not in QUESTION_TYPES:
            raise PairValidationError(
                f"question_type='{pair.question_type}' 은 허용되지 않습니다 {QUESTION_TYPES}."
            )
        missing = [p for p in pair.image_paths if not Path(p).is_file()]
        if missing:
            raise PairValidationError(f"이미지 경로가 유효하지 않습니다: {missing}")

    def delete(self, pair_id: str, reason: str = "") -> bool:
        """
        tombstone 방식 삭제 — 원본 줄은 그대로 두고 `status="deleted"` 레코드를 덧붙인다.
        """
        pair = self.get(pair_id)
        if pair is None:
            logger.warning(f"[Store] 삭제 대상 없음: {pair_id}")
            return False
        if not pair.is_active:
            return True

        pair.status = "deleted"
        pair.notes = (pair.notes + " | " if pair.notes else "") + f"deleted: {reason}"
        with _file_lock(self.path):
            self._append_raw(pair.to_dict())
        logger.info(f"[Store] 페어 삭제(tombstone): {pair_id} ({reason})")
        return True

    # ── 공유용 export (§5.2) ──────────────────────────────────────────────

    def export(
        self,
        fmt: Optional[str] = None,
        field_map: Optional[Dict[str, str]] = None,
        out_dir: Optional[str] = None,
        copy_images: bool = True,
    ) -> Dict[str, Any]:
        """
        공유 포맷으로 내보낸다.

        Args:
            fmt:         "rlaif_v"(기본) | "hf_conversational". 미지정 시 export_cfg 값.
            field_map:   출력 키 이름 매핑 (타 기관 스키마 대응). 미지정 시 export_cfg 값.
            out_dir:     출력 디렉토리. 미지정 시 `self.export_dir`.
            copy_images: 이미지를 export 폴더로 복사하고 상대경로로 기록할지.

        Returns:
            {"path", "n_pairs", "n_images", "format", "image_dir"}
        """
        fmt = (fmt or self.export_cfg.get("format") or "rlaif_v").lower()
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"지원하지 않는 export 포맷: {fmt} (선택지: {EXPORT_FORMATS})")
        fmap = dict(field_map or self.export_cfg.get("field_map") or {})

        base = Path(out_dir) if out_dir else self.export_dir
        base.mkdir(parents=True, exist_ok=True)
        img_dir = base / "images"
        if copy_images:
            img_dir.mkdir(parents=True, exist_ok=True)

        out_path = base / "dpo_share.jsonl"
        pairs = self.all()
        n_images = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                rel_images = self._export_images(pair, img_dir, copy_images)
                n_images += len(rel_images)
                if fmt == "rlaif_v":
                    rec = self._to_rlaif_v(pair, rel_images)
                else:
                    rec = self._to_hf_conversational(pair, rel_images)
                if fmap:
                    rec = {fmap.get(k, k): v for k, v in rec.items()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            f"[Store] export 완료 → {out_path} "
            f"({len(pairs)} pairs, {n_images} images, format={fmt})"
        )
        return {
            "path": str(out_path),
            "n_pairs": len(pairs),
            "n_images": n_images,
            "format": fmt,
            "image_dir": str(img_dir) if copy_images else "",
        }

    def _export_images(
        self, pair: DPOPair, img_dir: Path, copy_images: bool
    ) -> List[str]:
        """이미지를 export 폴더로 복사하고 export 루트 기준 상대경로를 돌려준다."""
        if not copy_images:
            return list(pair.image_paths)
        rels: List[str] = []
        for i, src in enumerate(pair.image_paths):
            sp = Path(src)
            if not sp.is_file():
                logger.warning(f"[Store] export 이미지 누락: {src} ({pair.pair_id})")
                continue
            suffix = sp.suffix or ".jpg"
            name = f"{pair.pair_id}_{i}{suffix}" if len(pair.image_paths) > 1 \
                else f"{pair.pair_id}{suffix}"
            dst = img_dir / name
            if not dst.exists():
                shutil.copy2(sp, dst)
            rels.append(f"images/{name}")
        return rels

    @staticmethod
    def _to_rlaif_v(pair: DPOPair, images: List[str]) -> Dict[str, Any]:
        """
        RLAIF-V 스키마: `{"image", "question", "chosen", "rejected"}`.
        이 스키마의 `image` 는 단일 필드이므로 이미지가 여러 장이면 **첫 장만** 나간다.
        (전체를 보존하려면 `hf_conversational` 포맷을 쓸 것)
        """
        return {
            "image": images[0] if images else "",
            "question": pair.question,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }

    @staticmethod
    def _to_hf_conversational(pair: DPOPair, images: List[str]) -> Dict[str, Any]:
        """HF `datasets` conversational 포맷 — 이미지 여러 장을 그대로 보존한다."""
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": pair.question})
        return {
            "images": images,
            "prompt": [{"role": "user", "content": content}],
            "chosen": [
                {"role": "assistant", "content": [{"type": "text", "text": pair.chosen}]}
            ],
            "rejected": [
                {"role": "assistant", "content": [{"type": "text", "text": pair.rejected}]}
            ],
        }


# ─── 설정 파일에서 저장소 생성 ─────────────────────────────────────────────

def from_config(config_path: str = "dpo_collector/config_dpo.yaml") -> DPOPairStore:
    """`config_dpo.yaml` 의 `paths` / `language` / `export` 섹션으로 저장소를 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths", {})
    return DPOPairStore(
        pairs_jsonl=paths.get("pairs_jsonl", "dpo_collector/outputs/dpo_pairs.jsonl"),
        export_dir=paths.get("export_dir"),
        language=cfg.get("language", "ko"),
        export_cfg=cfg.get("export", {}),
    )
