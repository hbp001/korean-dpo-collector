"""
dpo_collector/state.py
-----------------------
수집·학습 루프의 지속 상태 관리.

두 파일로 나눈다 (CLAUDE_dpo.md §9):
  - `state.json`            : 활성 어댑터 포인터, 페어 카운터, 학습 횟수 (자주 갱신, 작음)
  - `training_history.json` : 체크포인트별 loss + 평가 지표 이력 (append-only, UI 곡선 소스)

`store.py` 와 마찬가지로 Gradio 동시 요청을 고려해 쓰기는 파일 락으로 직렬화하고,
**read-modify-write 를 락 안에서 한 번에** 수행한다(카운터 유실 방지).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import KST, _file_lock

logger = logging.getLogger(__name__)

#: 어댑터 디렉토리 명명 규칙 — 코어 `model_trainer._get_adapter_path` 와 일관
_ADAPTER_RE = re.compile(r"^adapter_epoch(\d+)")

#: base 모델(어댑터 없음)을 가리키는 model_version 값
BASE_VERSION = "base"


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


# ─── 학습 이력 레코드 (§8) ─────────────────────────────────────────────────

@dataclass
class TrainingRecord:
    """학습 1회 + 직후 고정 평가셋 평가 결과."""

    checkpoint:    str                                   # "adapter_epoch003"
    step:          int = 0
    loss:          float = 0.0
    eval:          Dict[str, float] = field(default_factory=dict)   # {anls, accuracy, f1}
    n_train_pairs: int = 0
    timestamp:     str = ""
    notes:         str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingRecord":
        known = {f for f in cls.__dataclass_fields__}     # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ─── 상태 저장소 ───────────────────────────────────────────────────────────

class StateStore:
    """
    활성 어댑터 / 카운터 / 학습 이력 관리.

        state = StateStore("outputs/state.json", "outputs/training_history.json",
                           adapter_dir="outputs/adapters")
        state.bump_pair_count()               # 페어 저장 시
        state.pairs_since_last_train          # 트리거 판정용
        state.set_active_adapter(path)        # 학습 후 / UI 에서 어댑터 활성화
        state.model_version                   # 후보 생성 모델 라벨 → dpo_pairs.jsonl 에 기록
    """

    #: state.json 기본값
    _DEFAULTS: Dict[str, Any] = {
        "active_adapter": None,          # 절대/상대 경로. None = base 모델
        "total_pairs": 0,
        "pairs_since_last_train": 0,
        "train_count": 0,
        "last_train_at": None,
        "updated_at": None,
    }

    def __init__(
        self,
        state_json: str,
        history_json: Optional[str] = None,
        adapter_dir: Optional[str] = None,
    ):
        self.path = Path(state_json)
        self.history_path = (
            Path(history_json) if history_json
            else self.path.parent / "training_history.json"
        )
        self.adapter_dir = Path(adapter_dir) if adapter_dir \
            else self.path.parent / "adapters"

    # ── 저수준 읽기/쓰기 ──────────────────────────────────────────────────

    def _read(self) -> Dict[str, Any]:
        """락 없이 읽는다 (조회 전용). 파일이 없거나 깨졌으면 기본값."""
        if not self.path.exists():
            return dict(self._DEFAULTS)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[State] {self.path} 파싱 실패, 기본값 사용: {e}")
            return dict(self._DEFAULTS)
        merged = dict(self._DEFAULTS)
        merged.update(data if isinstance(data, dict) else {})
        return merged

    def _write(self, data: Dict[str, Any]) -> None:
        """원자적 쓰기 — 임시 파일에 쓰고 rename (중단 시 파일 손상 방지)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = _now_iso()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    def _update(self, **changes: Any) -> Dict[str, Any]:
        """
        락 안에서 read-modify-write 를 한 번에 수행한다.
        값이 callable 이면 현재 값을 받아 새 값을 돌려주는 함수로 취급한다.
        """
        with _file_lock(self.path):
            data = self._read()
            for k, v in changes.items():
                data[k] = v(data.get(k)) if callable(v) else v
            self._write(data)
            return data

    def snapshot(self) -> Dict[str, Any]:
        """현재 상태 전체 (UI 표시용)."""
        data = self._read()
        data["model_version"] = self.model_version
        data["adapter_dir"] = str(self.adapter_dir)
        return data

    # ── 활성 어댑터 ───────────────────────────────────────────────────────

    @property
    def active_adapter(self) -> Optional[str]:
        """현재 활성 LoRA 어댑터 경로. None 이면 base 모델."""
        p = self._read().get("active_adapter")
        if not p:
            return None
        if not Path(p).exists():
            logger.warning(f"[State] 활성 어댑터 경로가 존재하지 않습니다: {p}")
            return None
        return p

    def set_active_adapter(self, adapter_path: Optional[str]) -> None:
        """
        어댑터를 활성화한다. `None` 이면 base 모델로 되돌린다.
        경로가 없으면 예외 — 잘못된 포인터가 저장되어 이후 로드가 조용히 실패하는 걸 막는다.
        """
        if adapter_path is not None:
            p = Path(adapter_path)
            if not p.exists():
                raise FileNotFoundError(f"어댑터 경로가 없습니다: {adapter_path}")
            adapter_path = str(p.resolve())
        self._update(active_adapter=adapter_path)
        logger.info(
            f"[State] 활성 어댑터 변경 → {adapter_path or 'base (어댑터 없음)'}"
        )

    @property
    def model_version(self) -> str:
        """
        후보 답변을 생성한 모델 라벨. `dpo_pairs.jsonl::model_version` 에 기록한다.
        base 모델이면 "base", 어댑터가 있으면 디렉토리명("adapter_epoch003").
        """
        p = self.active_adapter
        return Path(p).name if p else BASE_VERSION

    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        `adapter_dir` 아래의 어댑터 목록 (최신 epoch 순).
        각 항목: {name, path, epoch, is_active, created_at}
        """
        if not self.adapter_dir.exists():
            return []
        active = self.active_adapter
        out: List[Dict[str, Any]] = []
        for d in self.adapter_dir.iterdir():
            if not d.is_dir():
                continue
            # PEFT 어댑터 디렉토리인지 확인
            if not (d / "adapter_config.json").exists():
                continue
            m = _ADAPTER_RE.match(d.name)
            out.append({
                "name": d.name,
                "path": str(d.resolve()),
                "epoch": int(m.group(1)) if m else -1,
                "is_active": active is not None and Path(active) == d.resolve(),
                "created_at": datetime.fromtimestamp(
                    d.stat().st_mtime, KST
                ).isoformat(timespec="seconds"),
            })
        return sorted(out, key=lambda x: (x["epoch"], x["created_at"]), reverse=True)

    def next_adapter_path(self) -> Path:
        """
        다음 학습이 저장할 어댑터 경로. 코어 명명 규칙(`adapter_epochNNN`)을 따른다.
        기존 최대 epoch + 1.
        """
        max_epoch = 0
        for a in self.list_adapters():
            max_epoch = max(max_epoch, a["epoch"])
        return self.adapter_dir / f"adapter_epoch{max_epoch + 1:03d}"

    # ── 페어 카운터 ───────────────────────────────────────────────────────

    @property
    def total_pairs(self) -> int:
        return int(self._read().get("total_pairs", 0))

    @property
    def pairs_since_last_train(self) -> int:
        return int(self._read().get("pairs_since_last_train", 0))

    @property
    def train_count(self) -> int:
        return int(self._read().get("train_count", 0))

    @property
    def is_first_train(self) -> bool:
        return self.train_count == 0

    def bump_pair_count(self, n: int = 1) -> Dict[str, Any]:
        """페어 저장 시 호출. 총계와 '마지막 학습 이후' 카운터를 함께 올린다."""
        return self._update(
            total_pairs=lambda v: int(v or 0) + n,
            pairs_since_last_train=lambda v: int(v or 0) + n,
        )

    def sync_pair_count(self, total: int) -> Dict[str, Any]:
        """
        저장소 실제 페어 수로 총계를 맞춘다.
        (외부에서 jsonl 을 직접 편집했거나 카운터가 어긋났을 때 UI 에서 호출)
        `pairs_since_last_train` 은 학습 시점 기준이라 여기서 건드리지 않는다.
        """
        return self._update(total_pairs=int(total))

    # ── 학습 이력 ─────────────────────────────────────────────────────────

    def history(self) -> List[TrainingRecord]:
        """체크포인트별 학습/평가 이력 (기록 순)."""
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[State] {self.history_path} 파싱 실패: {e}")
            return []
        if not isinstance(data, list):
            return []
        return [TrainingRecord.from_dict(d) for d in data if isinstance(d, dict)]

    def add_training_record(self, record: TrainingRecord) -> TrainingRecord:
        """
        학습 이력을 append 하고, 학습 카운터를 갱신한다
        (`pairs_since_last_train` 을 0 으로 리셋 — 다음 트리거 기준점).
        """
        if not record.timestamp:
            record.timestamp = _now_iso()

        with _file_lock(self.history_path):
            records = self.history()
            records.append(record)
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    [r.to_dict() for r in records], ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            tmp.replace(self.history_path)

        self._update(
            train_count=lambda v: int(v or 0) + 1,
            pairs_since_last_train=0,
            last_train_at=record.timestamp,
        )
        logger.info(
            f"[State] 학습 이력 추가: {record.checkpoint} "
            f"(loss={record.loss:.4f}, eval={record.eval}, pairs={record.n_train_pairs})"
        )
        return record

    def metric_series(self) -> Dict[str, List[Any]]:
        """
        UI 곡선용 시계열. `{"checkpoint": [...], "loss": [...], "anls": [...], ...}`
        평가 지표가 없는 체크포인트는 None 으로 채워 x축 정렬을 유지한다.
        """
        records = self.history()
        metric_keys: List[str] = []
        for r in records:
            for k in r.eval:
                if k not in metric_keys:
                    metric_keys.append(k)

        series: Dict[str, List[Any]] = {
            "checkpoint": [r.checkpoint for r in records],
            "step": [r.step for r in records],
            "loss": [r.loss for r in records],
            "n_train_pairs": [r.n_train_pairs for r in records],
        }
        for k in metric_keys:
            series[k] = [r.eval.get(k) for r in records]
        return series

    # ── 초기화 ────────────────────────────────────────────────────────────

    def reset(self, keep_adapter: bool = True) -> None:
        """카운터/이력을 초기화한다 (어댑터 파일 자체는 지우지 않는다)."""
        active = self.active_adapter if keep_adapter else None
        with _file_lock(self.path):
            data = dict(self._DEFAULTS)
            data["active_adapter"] = active
            self._write(data)
        if self.history_path.exists():
            self.history_path.unlink()
        logger.info(f"[State] 상태 초기화 (활성 어댑터 유지={keep_adapter})")


# ─── 설정 파일에서 상태 저장소 생성 ────────────────────────────────────────

def from_config(config_path: str = "dpo_collector/config_dpo.yaml") -> StateStore:
    """`config_dpo.yaml` 의 `paths` 섹션으로 StateStore 를 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths", {})
    return StateStore(
        state_json=paths.get("state_json", "dpo_collector/outputs/state.json"),
        history_json=paths.get("history_json", "dpo_collector/outputs/training_history.json"),
        adapter_dir=paths.get("adapter_dir", "dpo_collector/outputs/adapters"),
    )
