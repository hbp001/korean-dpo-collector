"""
dpo_collector/backends/__init__.py
-----------------------------------
백엔드 팩토리 — `config_dpo.yaml::model.name` 한 줄로 HF 모델을 자유 교체.

    from dpo_collector.backends import get_backend
    backend = get_backend({"name": "OpenGVLab/InternVL3_5-4B-HF", "backend": "auto"})
    backend.load()
    print(backend.infer(["/path/fig1.jpg"], "이 그림은 무엇을 보여주는가?"))

계열(family) 판별 규칙:
  - `internvl` : InternVL2 / InternVL2.5 커스텀 코드 체크포인트 (`InternVLChatModel`, `.chat()`)
  - `hf`       : 그 외 대부분 (Qwen2.5/3-VL, InternVL3.5-*-HF, LLaVA, Idefics, SmolVLM ...)

판별은 **config.json 우선**(architectures / auto_map) → 실패 시 모델명 휴리스틱 순.
둘 다 애매하면 예외를 던지고 사용자가 `model.backend` 를 명시하도록 안내한다
(추측 로드 금지 — CLAUDE_dpo.md §4.2).

스모크 테스트:
    python -m dpo_collector.backends --model OpenGVLab/InternVL3_5-4B-HF \
        --image /path/fig.jpg --question "이 그림을 한국어로 설명해줘."
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import VLMBackend
from .hf import HFBackend
from .internvl import InternVLBackend

logger = logging.getLogger(__name__)

__all__ = [
    "VLMBackend",
    "HFBackend",
    "InternVLBackend",
    "FAMILIES",
    "detect_family",
    "get_backend",
    "get_shared_backend",
    "unload_shared_backends",
    "shared_backend_info",
]

FAMILIES: Dict[str, Type[VLMBackend]] = {
    "hf": HFBackend,
    "internvl": InternVLBackend,
}

#: 커스텀 remote code 로 `.chat()` 을 노출하는 아키텍처
_CUSTOM_CHAT_ARCHS = ("InternVLChatModel",)

#: 이름만으로 `hf` 계열이 확실한 패턴
_HF_NAME_HINTS = (
    "qwen2-vl", "qwen2.5-vl", "qwen2_5_vl", "qwen3-vl", "qwen3vl",
    "llava", "idefics", "smolvlm", "paligemma", "llama-3.2-vision",
    "mllama", "gemma-3", "pixtral", "molmo", "phi-3-vision", "phi-4-multimodal",
)


# ─── 계열 판별 ─────────────────────────────────────────────────────────────

def _read_raw_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    모델의 config.json 원본을 읽는다 (로컬 경로 우선, 없으면 HF 허브 캐시/다운로드).
    `AutoConfig` 는 커스텀 코드 모델에서 trust_remote_code 없이 실패하므로 raw JSON 을 쓴다.
    """
    import json

    local = Path(model_name) / "config.json"
    if local.is_file():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"[backends] 로컬 config.json 파싱 실패: {e}")
            return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "config.json")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"[backends] 허브 config.json 조회 실패({model_name}): {e}")
        return None


def _family_from_config(cfg: Dict[str, Any]) -> Optional[str]:
    archs: List[str] = cfg.get("architectures") or []
    auto_map: Dict[str, Any] = cfg.get("auto_map") or {}

    if any(a in _CUSTOM_CHAT_ARCHS for a in archs):
        return "internvl"
    if archs and not auto_map:
        # transformers 네이티브 아키텍처 → 표준 프로세서 경로
        return "hf"
    if archs and auto_map:
        # 커스텀 remote code 인데 InternVL 이 아님 → 지원 여부 불확실
        return None
    return None


def _family_from_name(model_name: str) -> Optional[str]:
    m = model_name.lower()
    if "internvl" in m:
        # `-hf` 리비전(InternVL3_5-*-HF 등)은 transformers 네이티브
        if m.endswith("-hf") or "-hf/" in m or "_hf" in m:
            return "hf"
        # InternVL2 / InternVL2.5 → 커스텀 코드
        if "internvl2" in m or "internvl-chat" in m:
            return "internvl"
        return None
    if any(h in m for h in _HF_NAME_HINTS):
        return "hf"
    return None


def detect_family(model_name: str) -> str:
    """
    모델명/로컬 경로로부터 백엔드 계열을 판별한다.
    판별 불가 시 `ValueError` — 사용자가 `model.backend` 를 명시해야 한다.
    """
    cfg = _read_raw_config(model_name)
    if cfg is not None:
        fam = _family_from_config(cfg)
        if fam:
            logger.info(
                f"[backends] '{model_name}' → family='{fam}' "
                f"(config.json architectures={cfg.get('architectures')})"
            )
            return fam

    fam = _family_from_name(model_name)
    if fam:
        logger.info(f"[backends] '{model_name}' → family='{fam}' (모델명 휴리스틱)")
        return fam

    raise ValueError(
        f"백엔드 계열을 판별할 수 없습니다: '{model_name}'.\n"
        f"config_dpo.yaml 에 `backend` 를 명시하세요 — 선택지: {sorted(FAMILIES)}\n"
        f"  · hf       : AutoProcessor + AutoModelForImageTextToText 표준 경로 (대부분의 HF VLM)\n"
        f"  · internvl : InternVL2/2.5 커스텀 코드(.chat()) 경로"
    )


# ─── 팩토리 ────────────────────────────────────────────────────────────────

def get_backend(model_cfg: Dict[str, Any]) -> VLMBackend:
    """
    `config_dpo.yaml::model`(또는 `question_gen`) 딕셔너리로 백엔드 인스턴스를 만든다.
    로드는 하지 않는다 — 호출자가 `backend.load()` 를 명시적으로 호출해야 한다.
    """
    if not model_cfg or not model_cfg.get("name"):
        raise ValueError("model_cfg['name'] 이 필요합니다 (HF 모델명 또는 로컬 경로).")

    family = (model_cfg.get("backend") or "auto").lower()
    if family == "auto":
        family = detect_family(model_cfg["name"])

    if family not in FAMILIES:
        raise ValueError(
            f"알 수 없는 backend='{family}'. 선택지: {sorted(FAMILIES)} 또는 'auto'"
        )

    return FAMILIES[family](model_cfg)


# ─── 공유 백엔드 캐시 ──────────────────────────────────────────────────────
#
# 질문 생성기(Challenger)와 답변 생성기(Solver)는 서로 다른 모델일 수도, 같은 모델일 수도
# 있다(기본 config 는 동일 모델). 같은 모델을 두 번 로드하면 GPU 메모리가 두 배로 들고
# 4B 모델 두 개만 올려도 다른 작업과 충돌한다. (모델명, 계열, dtype, 어댑터)가 같으면
# 인스턴스를 공유한다.

_SHARED: Dict[str, VLMBackend] = {}


def _cache_key(model_cfg: Dict[str, Any], adapter_path: Optional[str]) -> str:
    return "|".join([
        str(model_cfg.get("name", "")),
        str(model_cfg.get("backend", "auto")),
        str(model_cfg.get("dtype", "bfloat16")),
        str(model_cfg.get("device_map", "auto")),
        str(model_cfg.get("max_num_tiles", 12)),
        str(adapter_path or ""),
    ])


def get_shared_backend(
    model_cfg: Dict[str, Any],
    adapter_path: Optional[str] = None,
    autoload: bool = True,
) -> VLMBackend:
    """
    캐시된 백엔드를 돌려준다. 없으면 만들고(옵션에 따라 로드하고) 캐시에 넣는다.

    Args:
        autoload: True 면 반환 전에 `load()` 까지 수행한다. 로드 실패 시 RuntimeError.
    """
    key = _cache_key(model_cfg, adapter_path)
    backend = _SHARED.get(key)
    if backend is None:
        backend = get_backend(model_cfg)
        _SHARED[key] = backend
        logger.info(
            f"[backends] 공유 백엔드 생성: {model_cfg.get('name')} "
            f"(adapter={adapter_path or 'none'}) — 캐시 {len(_SHARED)}개"
        )
    if autoload and not backend.is_loaded:
        if not backend.load(adapter_path):
            _SHARED.pop(key, None)
            raise RuntimeError(f"모델 로드 실패: {model_cfg.get('name')}")
    return backend


def unload_shared_backends(name: Optional[str] = None) -> int:
    """
    공유 백엔드를 언로드한다. `name` 을 주면 해당 모델만, 없으면 전부.
    학습 단계 진입 전 GPU 를 비우는 데 쓴다 (추론 모델과 학습 모델의 동시 점유 방지).

    Returns:
        언로드한 백엔드 수.
    """
    keys = [
        k for k in _SHARED
        if name is None or k.split("|")[0] == name
    ]
    for k in keys:
        try:
            _SHARED[k].unload()
        except Exception as e:
            logger.warning(f"[backends] 언로드 경고 ({k}): {e}")
        _SHARED.pop(k, None)
    if keys:
        logger.info(f"[backends] 공유 백엔드 {len(keys)}개 언로드")
    return len(keys)


def shared_backend_info() -> List[Dict[str, Any]]:
    """현재 캐시된 백엔드 목록 (UI 표시/디버깅용)."""
    return [
        {
            "name": b.model_name,
            "family": b.family,
            "adapter": b.adapter_path,
            "loaded": b.is_loaded,
        }
        for b in _SHARED.values()
    ]


# ─── 스모크 테스트 CLI ─────────────────────────────────────────────────────

def _main() -> int:  # pragma: no cover - 수동 검증용
    import argparse

    ap = argparse.ArgumentParser(
        description="백엔드 로드/추론 스모크 테스트 (계열 자동 판별 확인)"
    )
    ap.add_argument("--model", required=True, help="HF 모델명 또는 로컬 경로")
    ap.add_argument("--backend", default="auto", choices=["auto", "hf", "internvl"])
    ap.add_argument("--image", action="append", default=[], help="이미지 경로 (여러 번 지정 가능)")
    ap.add_argument("--question", default="이 이미지를 한국어로 자세히 설명해줘.")
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 경로")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-num-tiles", type=int, default=12)
    ap.add_argument("--n", type=int, default=1, help="후보 답변 개수")
    ap.add_argument("--detect-only", action="store_true", help="계열 판별만 하고 종료")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    if args.detect_only:
        print(f"family = {detect_family(args.model)}")
        return 0

    cfg = {
        "name": args.model,
        "backend": args.backend,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_num_tiles": args.max_num_tiles,
    }
    backend = get_backend(cfg)
    print(f"[smoke] backend = {backend!r}")

    if not backend.load(args.adapter):
        print("[smoke] 로드 실패")
        return 1
    print(f"[smoke] 로드 성공 — default_lora_targets={backend.default_lora_targets()}")

    answers = backend.infer_n(args.image, args.question, n=args.n)
    for i, a in enumerate(answers):
        print(f"\n--- 후보 {i + 1} ---\n{a}")

    backend.unload()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
