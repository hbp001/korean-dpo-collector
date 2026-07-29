"""
dpo_collector/trigger.py
-------------------------
자동 학습 트리거 정책 (CLAUDE_dpo.md §7).

판정 규칙:
  1. `n_new_pairs_since_last_train >= (첫 학습 ? first_train_min_pairs : retrain_every_n_pairs)`
  2. **그리고** 질문 유형별로 `min_pairs_per_type` 이상 모였을 것 (유형 편향 가드레일)

유효 페어만 센다: `chosen != rejected` 이고 양쪽이 비어 있지 않은 것.
(`store.py` 가 저장 시점에 이미 검증하지만, 외부에서 jsonl 을 편집했을 수 있어 다시 확인한다)

임계치와 무관한 **"지금 학습" 수동 실행**은 항상 가능하다 — 이 모듈은 판정만 하고
학습을 직접 호출하지 않는다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state import StateStore
from .store import QUESTION_TYPES, DPOPair, DPOPairStore

logger = logging.getLogger(__name__)


@dataclass
class TriggerDecision:
    """트리거 판정 결과 — UI 진행바와 학습 버튼 상태를 모두 여기서 만든다."""

    should_train: bool
    reason: str
    #: 마지막 학습 이후 쌓인 유효 페어 수
    n_since_last_train: int = 0
    #: 이번 학습에 필요한 임계치
    threshold: int = 0
    #: 임계치까지 남은 수
    remaining: int = 0
    #: 전체 유효 페어 수
    n_valid_total: int = 0
    #: 유형별 부족분 {유형: 부족 개수}
    type_shortfall: Dict[str, int] = field(default_factory=dict)
    is_first_train: bool = True

    @property
    def progress(self) -> float:
        """0.0~1.0 진행률."""
        if self.threshold <= 0:
            return 1.0
        return min(1.0, self.n_since_last_train / self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_train": self.should_train,
            "reason": self.reason,
            "n_since_last_train": self.n_since_last_train,
            "threshold": self.threshold,
            "remaining": self.remaining,
            "n_valid_total": self.n_valid_total,
            "type_shortfall": self.type_shortfall,
            "is_first_train": self.is_first_train,
            "progress": round(self.progress, 4),
        }

    def markdown(self) -> str:
        """UI 표시용 진행바 + 상태 문구."""
        filled = int(self.progress * 20)
        bar = "█" * filled + "░" * (20 - filled)
        label = "첫 학습" if self.is_first_train else "재학습"
        head = "✅ **학습 조건 충족**" if self.should_train else "⏳ 수집 중"
        lines = [
            f"{head} — {self.reason}",
            "",
            f"{label}까지 `{bar}` {self.n_since_last_train}/{self.threshold}"
            f"  (**{self.remaining}건 남음**)",
        ]
        if self.type_shortfall:
            short = " · ".join(
                f"{k} {v}건 부족" for k, v in self.type_shortfall.items()
            )
            lines.append(f"\n유형 편향 가드레일: {short}")
        return "\n".join(lines)


def is_valid_pair(pair: DPOPair) -> bool:
    """학습에 실제로 쓸 수 있는 페어인지."""
    chosen = (pair.chosen or "").strip()
    rejected = (pair.rejected or "").strip()
    return bool(chosen and rejected and chosen != rejected)


def valid_pairs(store: DPOPairStore) -> List[DPOPair]:
    """유효 페어만 추린다 (tombstone 으로 삭제된 것은 store 가 이미 제외)."""
    return [p for p in store.all() if is_valid_pair(p)]


class TrainingTrigger:
    """
        trigger = TrainingTrigger(store, state, train_cfg)
        decision = trigger.evaluate()
        if decision.should_train:
            ...   # dpo_train.run_dpo_training(...)
    """

    def __init__(
        self,
        store: DPOPairStore,
        state: StateStore,
        train_cfg: Optional[Dict[str, Any]] = None,
    ):
        cfg = dict(train_cfg or {})
        self.store = store
        self.state = state
        self.first_train_min_pairs = int(cfg.get("first_train_min_pairs", 150))
        self.retrain_every_n_pairs = int(cfg.get("retrain_every_n_pairs", 100))
        self.min_pairs_per_type = int(cfg.get("min_pairs_per_type", 10))
        #: 유형 가드레일을 적용할 질문 유형. 비우면 실제로 수집된 유형만 본다.
        self.guarded_types: List[str] = list(
            cfg.get("guarded_question_types") or []
        )

    # ── 판정 ──────────────────────────────────────────────────────────────

    def evaluate(self) -> TriggerDecision:
        pairs = valid_pairs(self.store)
        n_valid = len(pairs)
        is_first = self.state.is_first_train
        threshold = (
            self.first_train_min_pairs if is_first else self.retrain_every_n_pairs
        )

        # 마지막 학습 이후 쌓인 수는 state 카운터를 쓰되, 유효 페어 총계를 넘지 않게 한다
        # (외부에서 jsonl 을 손댔거나 카운터가 어긋난 경우 방어)
        since = min(self.state.pairs_since_last_train, n_valid)
        remaining = max(0, threshold - since)

        shortfall = self._type_shortfall(pairs)

        if since < threshold:
            return TriggerDecision(
                should_train=False,
                reason=f"페어 {remaining}건 더 필요합니다.",
                n_since_last_train=since, threshold=threshold,
                remaining=remaining, n_valid_total=n_valid,
                type_shortfall=shortfall, is_first_train=is_first,
            )

        if shortfall:
            short_txt = ", ".join(f"{k} {v}건" for k, v in shortfall.items())
            return TriggerDecision(
                should_train=False,
                reason=(
                    f"수량은 충족했지만 유형 편향 가드레일에 걸렸습니다 "
                    f"(부족: {short_txt}). 해당 유형을 더 모으거나 '지금 학습'으로 강제 실행하세요."
                ),
                n_since_last_train=since, threshold=threshold,
                remaining=0, n_valid_total=n_valid,
                type_shortfall=shortfall, is_first_train=is_first,
            )

        return TriggerDecision(
            should_train=True,
            reason=f"유효 페어 {since}건 확보 (임계치 {threshold}).",
            n_since_last_train=since, threshold=threshold,
            remaining=0, n_valid_total=n_valid,
            type_shortfall={}, is_first_train=is_first,
        )

    def _type_shortfall(self, pairs: List[DPOPair]) -> Dict[str, int]:
        """
        유형별 최소 개수 미달분.

        `guarded_question_types` 를 지정하지 않았다면 **실제로 수집된 유형만** 검사한다.
        아직 한 건도 없는 유형까지 요구하면(예: MCQ) 초기 학습이 영원히 막힌다.
        """
        if self.min_pairs_per_type <= 0:
            return {}
        counts: Dict[str, int] = {}
        for p in pairs:
            counts[p.question_type] = counts.get(p.question_type, 0) + 1

        targets = self.guarded_types or [t for t in QUESTION_TYPES if counts.get(t)]
        return {
            t: self.min_pairs_per_type - counts.get(t, 0)
            for t in targets
            if counts.get(t, 0) < self.min_pairs_per_type
        }

    # ── 학습 대상 페어 ────────────────────────────────────────────────────

    def training_pairs(self, limit: Optional[int] = None) -> List[DPOPair]:
        """
        학습에 넣을 유효 페어.

        누적 전체를 쓴다 — 마지막 학습 이후 것만 쓰면 앞서 모은 선호가 잊히고,
        DPO 는 reference 가 base 로 고정돼 있어 전체 재학습이 자연스럽다.
        """
        pairs = valid_pairs(self.store)
        return pairs[-limit:] if limit else pairs


# ─── 설정 파일 연동 ────────────────────────────────────────────────────────

def from_config(
    config_path: str = "dpo_collector/config_dpo.yaml",
) -> TrainingTrigger:
    """`config_dpo.yaml` 의 `paths` / `train` 섹션으로 트리거를 만든다."""
    import yaml

    from .state import from_config as state_from_config
    from .store import from_config as store_from_config

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return TrainingTrigger(
        store=store_from_config(config_path),
        state=state_from_config(config_path),
        train_cfg=cfg.get("train", {}),
    )
