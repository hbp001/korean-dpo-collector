"""
dpo_collector/backends/base.py
-------------------------------
모델 무관 VLM 백엔드 추상 인터페이스.

목표: HuggingFace 모델을 `config_dpo.yaml::model.name` 한 줄로 자유 교체.
모델마다 로딩/추론 방식이 다르므로(표준 processor vs. InternVL 커스텀 `.chat()`)
공통 인터페이스 뒤에 계열(family)별 구현을 두고 팩토리로 선택한다.

구현 범위(현재 단계): **로드 / 추론만**.
학습용 메서드(`load_for_train`, `logprob`, `set_adapter_enabled`, `save_adapter`)는
시그니처만 정의하고 `NotImplementedError` 를 던진다 — DPO 학습 단계에서 구현.

주의(CLAUDE_dpo.md §1-5): InternVL 계열은 4bit 양자화 시 InternViT 출력이 깨지므로
bf16 로드를 강제한다. 이 파일의 `resolve_dtype()` 이 그 가드를 담당한다.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── 공용 유틸 ─────────────────────────────────────────────────────────────

def resolve_dtype(name: str) -> Any:
    """문자열 dtype → torch dtype. 미지정/미지원이면 bfloat16."""
    import torch

    table = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
        "auto":     "auto",
    }
    key = (name or "bfloat16").lower()
    if key not in table:
        logger.warning(f"[Backend] 알 수 없는 dtype '{name}' → bfloat16 사용")
        return torch.bfloat16
    return table[key]


def dtype_kwarg_name() -> str:
    """
    `from_pretrained` 의 dtype 인자 이름.
    transformers 4.56+ 는 `dtype`, 그 이전은 `torch_dtype` (현재 환경: 5.x → `dtype`).
    """
    try:
        import transformers

        parts = transformers.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "dtype"


def valid_image_paths(paths: Optional[List[str]]) -> List[str]:
    """실제로 열 수 있는 이미지 경로만 남긴다 (깨진 크롭 이미지 방어)."""
    if not paths:
        return []
    out: List[str] = []
    for p in paths:
        try:
            fp = Path(p)
            if not fp.is_file() or fp.stat().st_size < 100:
                logger.warning(f"[Backend] 이미지 스킵(없음/빈 파일): {p}")
                continue
            from PIL import Image as PILImage

            with PILImage.open(fp) as img:
                img.verify()
            out.append(str(fp))
        except Exception as e:
            logger.warning(f"[Backend] 이미지 스킵(손상): {p} ({e})")
    return out


def flash_attn_available() -> bool:
    """flash-attn 설치 여부. 미설치 환경에서 로드 실패를 막기 위한 사전 확인."""
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


# ─── 추상 백엔드 ───────────────────────────────────────────────────────────

class VLMBackend(ABC):
    """
    모든 백엔드가 따르는 공통 인터페이스.

    사용 흐름:
        backend = get_backend(model_cfg)      # backends/__init__.py 팩토리
        backend.load()                        # 또는 load(adapter_path)
        text = backend.infer(["/a.jpg"], "이 그림은 무엇을 보여주는가?")
        backend.unload()
    """

    #: 팩토리에서 사용하는 계열 식별자
    family: str = "base"

    def __init__(self, model_cfg: Dict[str, Any]):
        cfg = dict(model_cfg or {})
        self.model_cfg = cfg

        self.model_name: str = cfg.get("name") or ""
        if not self.model_name:
            raise ValueError("model_cfg['name'] 이 필요합니다 (HF 모델명 또는 로컬 경로).")

        self.dtype_name: str    = cfg.get("dtype", "bfloat16")
        self.device_map         = cfg.get("device_map", "auto")
        self.use_flash_attn: bool = bool(cfg.get("use_flash_attn", False))
        self.max_num_tiles: int   = int(cfg.get("max_num_tiles", 12))
        self.max_new_tokens: int  = int(cfg.get("max_new_tokens", 512))
        self.temperature: float   = float(cfg.get("temperature", 0.7))

        # 런타임 상태
        self.model: Any     = None
        self.processor: Any = None   # hf 계열
        self.tokenizer: Any = None   # internvl 계열
        self.adapter_path: Optional[str] = None

    # ── 추론 (수집 / 플레이그라운드용) ────────────────────────────────────

    @abstractmethod
    def load(self, adapter_path: Optional[str] = None) -> bool:
        """
        추론용 모델 로드. 이미 같은 어댑터로 로드돼 있으면 스킵하고 True 반환.

        adapter_path 가 주어지면 LoRA 어댑터를 얹는다. 어댑터는 **merge 하지 않는다** —
        `set_adapter_enabled()` 로 base/adapter 비교가 가능해야 하기 때문.
        """

    @abstractmethod
    def infer(
        self,
        image_paths: List[str],
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """이미지 + 질문 → 응답 텍스트 1개. 실패 시 빈 문자열."""

    def infer_n(
        self,
        image_paths: List[str],
        question: str,
        n: int = 3,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        후보 답변 N개 샘플링 (기본 구현: `infer()` N회 반복).
        다양성 확보를 위해 temperature 가 0 이하면 강제로 0.9 로 올린다.
        배치 샘플링이 가능한 백엔드는 이 메서드를 오버라이드하면 된다.
        """
        temp = temperature if temperature is not None else self.temperature
        if temp <= 0:
            temp = 0.9
        return [
            self.infer(image_paths, question, max_new_tokens, temp)
            for _ in range(max(1, n))
        ]

    @abstractmethod
    def unload(self) -> None:
        """GPU 메모리 해제. 학습 단계 진입 전 반드시 호출."""

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── 학습 (DPO 단계에서 구현) ──────────────────────────────────────────

    def load_for_train(
        self,
        adapter_path: Optional[str] = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """학습용 로드 (LoRA 부착, trainable)."""
        raise NotImplementedError(
            f"{type(self).__name__}.load_for_train — DPO 학습 단계에서 구현 예정"
        )

    def logprob(
        self,
        image_paths: List[str],
        question: str,
        answer: str,
    ) -> Any:
        """log P(answer | image, question). DPO loss 계산용 (torch.Tensor)."""
        raise NotImplementedError(
            f"{type(self).__name__}.logprob — DPO 학습 단계에서 구현 예정"
        )

    def set_adapter_enabled(self, enabled: bool) -> None:
        """
        LoRA on/off 토글 → policy(on) / reference(off) 전환.
        policy와 reference를 각각 로드하지 않기 위한 OOM 방지 장치(§1-6).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.set_adapter_enabled — DPO 학습 단계에서 구현 예정"
        )

    def save_adapter(self, path: str) -> None:
        """학습된 LoRA 어댑터 저장."""
        raise NotImplementedError(
            f"{type(self).__name__}.save_adapter — DPO 학습 단계에서 구현 예정"
        )

    # ── 모델 아키텍처별 기본값 ────────────────────────────────────────────

    @abstractmethod
    def default_lora_targets(self) -> List[str]:
        """config 에 `model.lora.target_modules` 미지정 시 사용할 LoRA 타깃 모듈."""

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _gen_kwargs(
        self,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        """generate() 공통 인자. temperature <= 0 이면 greedy."""
        temp = self.temperature if temperature is None else temperature
        kw: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            "do_sample": temp > 0,
        }
        if temp > 0:
            kw["temperature"] = temp
            kw["top_p"] = float(self.model_cfg.get("top_p", 0.9))
        return kw

    def _attn_implementation(self) -> Optional[str]:
        """flash-attn 요청 시 설치 여부를 확인하고, 없으면 None(기본 구현)."""
        if not self.use_flash_attn:
            return None
        if not flash_attn_available():
            logger.warning(
                "[Backend] use_flash_attn=true 이지만 flash-attn 미설치 → 기본 attention 사용"
            )
            return None
        return "flash_attention_2"

    def _free_cuda(self) -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover - 환경 의존
            logger.debug(f"[Backend] CUDA 캐시 정리 경고: {e}")

    def __repr__(self) -> str:  # pragma: no cover - 로깅용
        return (
            f"<{type(self).__name__} family={self.family} name={self.model_name} "
            f"dtype={self.dtype_name} loaded={self.is_loaded} adapter={self.adapter_path}>"
        )
