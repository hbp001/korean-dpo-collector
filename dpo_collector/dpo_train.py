"""
dpo_collector/dpo_train.py
---------------------------
수동 DPO LoRA fine-tune + 고정 평가셋 평가 로깅 (CLAUDE_dpo.md §8).

**수동 루프를 쓰는 이유**: TRL 의 `DPOTrainer` 는 표준 HF 모델을 전제로 하는데,
InternVL 계열 커스텀 체크포인트(`.chat()` 인터페이스)와는 맞물리지 않는다.
코어 `self_play/model_trainer.py::_train_grpo_manual` 을 템플릿으로, 백엔드 인터페이스
(`logprob` / `set_adapter_enabled`)만 알면 되는 형태로 작성한다.

DPO loss:
    Δθ   = logπθ(y_w|x) - logπθ(y_l|x)          # adapter ON  (policy)
    Δref = logπref(y_w|x) - logπref(y_l|x)      # adapter OFF (reference)
    loss = -log σ( β · (Δθ - Δref) )

**OOM 방지(§1-6)**: policy 와 reference 를 따로 로드하지 않는다. 같은 모델에서
LoRA 를 켜면 policy, 끄면 reference 다. 학습 시작 전에는 추론용 공유 백엔드를
언로드해 GPU 를 비운다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .state import StateStore, TrainingRecord
from .store import DPOPair

logger = logging.getLogger(__name__)

#: 학습 로그를 UI 로 흘려보내는 콜백 시그니처: (progress 0~1, 메시지)
ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class TrainResult:
    """학습 1회 결과."""

    checkpoint: str = ""
    adapter_path: str = ""
    steps: int = 0
    loss: float = 0.0
    #: chosen 이 rejected 보다 높은 보상을 받은 비율 (0.5 = 무학습, 1.0 = 완전 분리)
    reward_accuracy: float = 0.0
    n_pairs: int = 0
    skipped: int = 0
    seconds: float = 0.0
    eval: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    loss_curve: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error and self.steps > 0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["ok"] = self.ok
        return d


def run_dpo_training(
    pairs: List[DPOPair],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    state: StateStore,
    eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    progress_cb: ProgressCb = None,
    resume_from_adapter: Optional[str] = None,
) -> TrainResult:
    """
    DPO LoRA 학습 1회를 수행하고 어댑터를 저장한다.

    Args:
        pairs:      학습에 쓸 유효 DPO 페어 (`trigger.valid_pairs` 결과)
        model_cfg:  `config_dpo.yaml::model`
        train_cfg:  `config_dpo.yaml::train`
        state:      어댑터 경로 발번 / 이력 기록
        eval_fn:    학습 후 평가 함수. 백엔드를 받아 {anls, accuracy, f1} 반환.
                    None 이면 평가를 건너뛴다.
        resume_from_adapter: 이어서 학습할 어댑터 경로 (None = 새 LoRA)

    Returns:
        TrainResult (실패 시 `error` 채워짐)
    """
    result = TrainResult(n_pairs=len(pairs))
    if not pairs:
        result.error = "학습할 페어가 없습니다."
        return result

    t0 = time.time()
    backend = None
    try:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW

        from .backends import get_backend, unload_shared_backends

        # ── 추론용 모델을 먼저 내려 GPU 를 비운다 (§8 메모리 관리) ──────────
        n_unloaded = unload_shared_backends()
        if n_unloaded:
            logger.info(f"[DPOTrain] 추론용 공유 백엔드 {n_unloaded}개 언로드")

        beta = float(train_cfg.get("dpo_beta", 0.1))
        lr = float(train_cfg.get("learning_rate", 5.0e-6))
        epochs = int(train_cfg.get("epochs", 2))
        grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 8)))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        lora_cfg = dict(train_cfg.get("lora", {}))
        log_every = max(1, int(train_cfg.get("log_every", 10)))

        _report(progress_cb, 0.02, "학습용 모델 로드 중…")
        backend = get_backend(model_cfg)

        # 학습 전용 메모리 설정을 lora_cfg 로 함께 넘긴다
        # (백엔드가 gradient checkpointing / 타일 수를 여기서 읽는다)
        lora_cfg.setdefault(
            "gradient_checkpointing", train_cfg.get("gradient_checkpointing", True)
        )
        if train_cfg.get("max_num_tiles"):
            lora_cfg["max_num_tiles"] = train_cfg["max_num_tiles"]
        if train_cfg.get("max_answer_tokens"):
            backend.model_cfg["max_answer_tokens"] = train_cfg["max_answer_tokens"]

        if not backend.load_for_train(resume_from_adapter, lora_cfg):
            result.error = "학습용 모델 로드 실패"
            return result

        params = [p for p in backend.model.parameters() if p.requires_grad]
        if not params:
            result.error = "학습 가능한 파라미터가 없습니다 (LoRA 부착 확인 필요)."
            return result
        optimizer = AdamW(params, lr=lr)

        total_steps = len(pairs) * epochs
        step = 0
        acc_loss = 0.0
        n_correct = 0
        n_scored = 0

        logger.info(
            f"[DPOTrain] 시작 — pairs={len(pairs)} epochs={epochs} "
            f"beta={beta} lr={lr} grad_accum={grad_accum}"
        )

        for epoch in range(epochs):
            for pair in pairs:
                step += 1
                try:
                    images = list(pair.image_paths or [])

                    # ── policy: 어댑터 ON ─────────────────────────────────
                    backend.set_adapter_enabled(True)
                    pol_chosen = backend.logprob(images, pair.question, pair.chosen)
                    pol_rejected = backend.logprob(images, pair.question, pair.rejected)

                    # ── reference: 어댑터 OFF (grad 불필요) ───────────────
                    backend.set_adapter_enabled(False)
                    with torch.no_grad():
                        ref_chosen = backend.logprob(images, pair.question, pair.chosen)
                        ref_rejected = backend.logprob(
                            images, pair.question, pair.rejected
                        )
                    backend.set_adapter_enabled(True)

                    pi_logratio = pol_chosen - pol_rejected
                    ref_logratio = ref_chosen.detach() - ref_rejected.detach()
                    logits = beta * (pi_logratio - ref_logratio)
                    loss = -F.logsigmoid(logits)

                    (loss / grad_accum).backward()

                    if step % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    acc_loss += float(loss.item())
                    # implicit reward: β·(logπθ - logπref). chosen 쪽이 크면 학습이 되는 중.
                    n_scored += 1
                    if float(logits.item()) > 0:
                        n_correct += 1

                    if step % log_every == 0:
                        avg = acc_loss / max(step, 1)
                        result.loss_curve.append(round(avg, 4))
                        msg = (
                            f"epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                            f"loss={avg:.4f} reward_acc={n_correct / max(n_scored, 1):.3f}"
                        )
                        logger.info(f"[DPOTrain] {msg}")
                        _report(progress_cb, 0.05 + 0.75 * step / total_steps, msg)

                except torch.cuda.OutOfMemoryError as e:
                    # OOM 은 남은 그래프를 정리하고 캐시를 비워야 다음 페어가 살아남는다.
                    # 정리하지 않으면 한 번 터진 뒤 모든 페어가 연쇄적으로 실패한다.
                    result.skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    for _v in ("pol_chosen", "pol_rejected", "ref_chosen",
                               "ref_rejected", "loss", "logits"):
                        locals().pop(_v, None)
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"[DPOTrain] GPU 메모리 부족으로 페어 스킵 ({pair.pair_id}). "
                        f"train.max_num_tiles / max_answer_tokens 를 낮추면 개선됩니다. — {e}"
                    )
                    continue
                except Exception as e:
                    result.skipped += 1
                    logger.warning(
                        f"[DPOTrain] 페어 스킵 ({pair.pair_id}): {e}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

        # 남은 gradient 반영
        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        effective = step - result.skipped
        if effective <= 0:
            result.error = "모든 페어가 스킵되어 학습이 이뤄지지 않았습니다."
            return result

        result.steps = step
        result.loss = round(acc_loss / max(effective, 1), 4)
        result.reward_accuracy = round(n_correct / max(n_scored, 1), 4)

        # ── 어댑터 저장 ───────────────────────────────────────────────────
        adapter_path = state.next_adapter_path()
        _report(progress_cb, 0.85, f"어댑터 저장 중… ({adapter_path.name})")
        backend.save_adapter(str(adapter_path))
        result.adapter_path = str(adapter_path)
        result.checkpoint = adapter_path.name

        # ── 고정 평가셋 평가 ──────────────────────────────────────────────
        if eval_fn is not None:
            _report(progress_cb, 0.88, "고정 평가셋으로 평가 중…")
            try:
                backend.model.eval()
                backend.set_adapter_enabled(True)
                result.eval = eval_fn(backend) or {}
            except Exception as e:
                logger.error(f"[DPOTrain] 평가 실패(학습 결과는 유지): {e}")
                result.eval = {}

        result.seconds = round(time.time() - t0, 1)
        logger.info(
            f"[DPOTrain] 완료 — {result.checkpoint} loss={result.loss} "
            f"reward_acc={result.reward_accuracy} eval={result.eval} "
            f"({result.seconds}s, 스킵 {result.skipped})"
        )
        return result

    except Exception as e:
        logger.exception("[DPOTrain] 학습 실패")
        result.error = str(e)
        result.seconds = round(time.time() - t0, 1)
        return result

    finally:
        # 학습 모델을 반드시 내려 다음 추론이 GPU 를 쓸 수 있게 한다
        if backend is not None:
            try:
                backend.unload()
            except Exception as e:
                logger.debug(f"[DPOTrain] 언로드 경고: {e}")


def record_training(
    state: StateStore,
    result: TrainResult,
    activate: bool = True,
) -> Optional[TrainingRecord]:
    """
    학습 결과를 `training_history.json` 에 남기고, 필요하면 새 어댑터를 활성화한다.
    (카운터 리셋은 `StateStore.add_training_record` 가 처리한다)
    """
    if not result.ok:
        logger.warning(f"[DPOTrain] 실패한 학습은 이력에 남기지 않습니다: {result.error}")
        return None

    record = TrainingRecord(
        checkpoint=result.checkpoint,
        step=result.steps,
        loss=result.loss,
        eval=dict(result.eval),
        n_train_pairs=result.n_pairs,
        notes=(
            f"reward_acc={result.reward_accuracy}, "
            f"skipped={result.skipped}, {result.seconds}s"
        ),
    )
    state.add_training_record(record)

    if activate and result.adapter_path:
        try:
            state.set_active_adapter(result.adapter_path)
        except Exception as e:
            logger.error(f"[DPOTrain] 어댑터 활성화 실패: {e}")
    return record


def _report(cb: ProgressCb, frac: float, msg: str) -> None:
    if cb is None:
        return
    try:
        cb(frac, msg)
    except Exception:  # pragma: no cover - UI 콜백 오류가 학습을 막지 않게
        pass


# ─── 설정 파일 연동 ────────────────────────────────────────────────────────

def make_eval_fn(
    config_path: str = "dpo_collector/config_dpo.yaml",
) -> Optional[Callable[[Any], Dict[str, float]]]:
    """
    고정 평가셋이 준비돼 있으면 평가 함수를, 없으면 None 을 돌려준다.
    `run_dpo_training(eval_fn=...)` 에 그대로 넘긴다.
    """
    import yaml

    from .eval_ko import evaluate, from_config as eval_from_config, metrics_only

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ev_cfg = cfg.get("eval", {})
    evalset, _ = eval_from_config(config_path)

    items = evalset.load()
    if not items:
        logger.warning(
            "[DPOTrain] 고정 평가셋이 비어 있어 학습 후 평가를 건너뜁니다. "
            "`python -m dpo_collector.eval_ko draft` → 검토 → `confirm` 을 먼저 수행하세요."
        )
        return None

    def _eval(backend) -> Dict[str, float]:
        res = evaluate(
            backend, items,
            max_new_tokens=int(ev_cfg.get("max_new_tokens", 48)),
            temperature=float(ev_cfg.get("temperature", 0.0)),
        )
        return metrics_only(res)

    return _eval


def train_from_config(
    config_path: str = "dpo_collector/config_dpo.yaml",
    limit: Optional[int] = None,
    force: bool = False,
    progress_cb: ProgressCb = None,
) -> TrainResult:
    """
    설정 파일 하나로 트리거 판정 → 학습 → 이력 기록까지 수행한다.

    Args:
        limit: 학습에 쓸 페어 수 상한 (디버깅용)
        force: 트리거 판정을 무시하고 강제 학습 ("지금 학습" 버튼)
    """
    import yaml

    from .state import from_config as state_from_config
    from .trigger import from_config as trigger_from_config

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    trigger = trigger_from_config(config_path)
    state = state_from_config(config_path)

    decision = trigger.evaluate()
    if not decision.should_train and not force:
        return TrainResult(
            error=f"학습 조건 미충족: {decision.reason}",
            n_pairs=decision.n_valid_total,
        )

    pairs = trigger.training_pairs(limit=limit)
    result = run_dpo_training(
        pairs=pairs,
        model_cfg=cfg.get("model", {}),
        train_cfg=cfg.get("train", {}),
        state=state,
        eval_fn=make_eval_fn(config_path),
        progress_cb=progress_cb,
        resume_from_adapter=state.active_adapter,
    )
    record_training(state, result)
    return result


# ─── CLI ───────────────────────────────────────────────────────────────────

def _main() -> int:  # pragma: no cover - 수동 운영용
    import argparse
    import json

    ap = argparse.ArgumentParser(description="DPO LoRA 학습 (수동 루프)")
    ap.add_argument("--config", default="dpo_collector/config_dpo.yaml")
    ap.add_argument("--force", action="store_true", help="트리거 무시하고 지금 학습")
    ap.add_argument("--limit", type=int, default=None, help="학습 페어 수 상한")
    ap.add_argument("--status", action="store_true", help="트리거 상태만 출력")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    from .trigger import from_config as trigger_from_config

    if args.status:
        d = trigger_from_config(args.config).evaluate()
        print(json.dumps(d.to_dict(), ensure_ascii=False, indent=2))
        print()
        print(d.markdown())
        return 0

    result = train_from_config(args.config, limit=args.limit, force=args.force)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
