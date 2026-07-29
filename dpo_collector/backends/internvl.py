"""
dpo_collector/backends/internvl.py
-----------------------------------
계열 B — InternVL 커스텀 코드 백엔드.

OpenGVLab InternVL2 / InternVL2.5 체크포인트 전용 경로.
이 리비전들은 `trust_remote_code` 커스텀 모델(`InternVLChatModel`)이라
`AutoProcessor`/`AutoModelForImageTextToText` 를 쓰지 못하고,
`AutoModel` + `AutoTokenizer` + `model.chat()` 인터페이스를 쓴다.

  ※ `OpenGVLab/InternVL3_5-*-HF` 같은 `-HF` 리비전은 transformers 네이티브이므로
    이 백엔드가 아니라 `hf.py` 로 라우팅된다 (backends/__init__.py::detect_family).

이미지 전처리는 InternVL 공식 `load_image()`(dynamic tiling)를 그대로 옮겨왔다.
`max_num` 타일 수는 `config_dpo.yaml::model.max_num_tiles` 로 조정한다.

⚠ 양자화 금지(CLAUDE_dpo.md §1-5): InternVL 은 4bit 로드 시 InternViT 출력이 깨져
   이미지 이해가 무너진다. 이 백엔드는 bf16/fp16 만 허용하고 4bit/8bit 요청은 거부한다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import VLMBackend, dtype_kwarg_name, resolve_dtype, valid_image_paths

logger = logging.getLogger(__name__)

# InternVL 공식 전처리 상수 (ImageNet 정규화)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── InternVL 공식 이미지 전처리 (dynamic tiling) ──────────────────────────

def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """원본 종횡비에 가장 가까운 타일 배치(w_tiles, h_tiles)를 고른다."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # 동률이면 넓은 이미지 쪽을 선호 (공식 구현과 동일)
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    """고해상도 이미지를 448x448 타일 여러 장으로 분할 (InternVL 공식 로직)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12):
    """이미지 경로 → pixel_values 텐서 (N_tiles, 3, 448, 448)."""
    import torch
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ─── transformers 5.x 호환 shim ────────────────────────────────────────────
#
# InternVL2/2.5 의 remote code 는 transformers 4.x 시절에 작성되어
# `InternVLChatModel.__init__` 이 `post_init()` 을 호출하지 않는다.
# transformers 5.x 는 `post_init()` 에서 인스턴스 속성 `all_tied_weights_keys` 를
# 세팅한 뒤 가중치 로딩 중에 이를 참조하므로, 그대로 두면
#   AttributeError: 'InternVLChatModel' object has no attribute 'all_tied_weights_keys'
# 로 로드가 실패한다.
#
# 코어/외부 라이브러리를 수정하지 않고, 누락 시에만 인스턴스별 빈 dict 를 돌려주는
# **non-data 디스크립터**를 클래스 기본값으로 붙인다. `post_init()` 이 정상 호출되는
# 모델은 인스턴스 __dict__ 에 값을 직접 쓰므로 그쪽이 우선하고 이 shim 은 무시된다.

_SHIMS_APPLIED = False


class _LazyEmptyDict:
    """접근 시점에 인스턴스별 빈 dict 를 만들어 __dict__ 에 캐시하는 디스크립터."""

    def __init__(self, name: str):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return {}
        value: Dict[str, Any] = {}
        obj.__dict__[self._name] = value
        return value


def _apply_transformers_compat_shims() -> None:
    """구버전 remote code 를 transformers 5.x 에서 로드하기 위한 호환 패치."""
    global _SHIMS_APPLIED
    if _SHIMS_APPLIED:
        return
    try:
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = _LazyEmptyDict(
                "all_tied_weights_keys"
            )
            logger.info(
                "[InternVLBackend] transformers 호환 shim 적용: "
                "PreTrainedModel.all_tied_weights_keys 기본값"
            )
        _SHIMS_APPLIED = True
    except Exception as e:
        logger.warning(f"[InternVLBackend] 호환 shim 적용 실패(무시하고 진행): {e}")


# ─── 백엔드 ────────────────────────────────────────────────────────────────

class InternVLBackend(VLMBackend):
    """InternVL2 / InternVL2.5 커스텀 코드 백엔드 (`model.chat()` 경로)."""

    family = "internvl"

    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__(model_cfg)

        # 양자화 가드 — InternViT 출력이 깨지므로 허용하지 않는다
        for key in ("load_in_4bit", "load_in_8bit"):
            if self.model_cfg.get(key):
                raise ValueError(
                    f"InternVL 백엔드는 {key} 를 지원하지 않습니다. "
                    "4bit/8bit 로드 시 InternViT 이미지 이해가 깨집니다 → bfloat16 을 사용하세요."
                )
        if self.dtype_name.lower() in ("float32", "fp32", "auto"):
            logger.warning(
                f"[InternVLBackend] dtype='{self.dtype_name}' → 공식 권장값 bfloat16 으로 강제합니다."
            )
            self.dtype_name = "bfloat16"

        self.input_size: int = int(self.model_cfg.get("input_size", 448))

    # ── 로드 ──────────────────────────────────────────────────────────────

    def load(self, adapter_path: Optional[str] = None) -> bool:
        if self.is_loaded and self.adapter_path == adapter_path:
            return True
        if self.is_loaded:
            self.unload()

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            _apply_transformers_compat_shims()

            kw: Dict[str, Any] = {
                dtype_kwarg_name(): resolve_dtype(self.dtype_name),
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            # InternVL 커스텀 코드는 flash-attn 을 자체 플래그로 받는다
            kw["use_flash_attn"] = bool(
                self.use_flash_attn and self._attn_implementation() is not None
            )
            if self.device_map:
                kw["device_map"] = self.device_map

            logger.info(
                f"[InternVLBackend] 모델 로드: {self.model_name} "
                f"(dtype={self.dtype_name}, use_flash_attn={kw['use_flash_attn']})"
            )
            self.model = AutoModel.from_pretrained(self.model_name, **kw)

            if not self.device_map:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(dev)

            if adapter_path:
                if Path(adapter_path).exists():
                    from peft import PeftModel

                    logger.info(f"[InternVLBackend] LoRA 어댑터 적용: {adapter_path}")
                    # merge 하지 않음 — adapter on/off 토글 보존
                    self.model = PeftModel.from_pretrained(
                        self.model, adapter_path, is_trainable=False
                    )
                else:
                    logger.warning(
                        f"[InternVLBackend] 어댑터 경로 없음 → base 모델 사용: {adapter_path}"
                    )
                    adapter_path = None

            self.model.eval()

            # ※ AutoProcessor 가 아니라 AutoTokenizer (use_fast=False 필수)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )

            self.adapter_path = adapter_path
            logger.info("[InternVLBackend] 로드 완료")
            return True

        except Exception as e:
            logger.error(f"[InternVLBackend] 로드 실패: {e}")
            self.model = None
            self.tokenizer = None
            return False

    # ── 추론 ──────────────────────────────────────────────────────────────

    def _prepare_pixel_values(self, image_paths: List[str]):
        """
        이미지 경로 리스트 → (pixel_values, num_patches_list, image_prefix).
        다중 이미지는 타일을 concat 하고 질문에 `Image-N: <image>` 프리픽스를 붙인다.
        """
        import torch

        if not image_paths:
            return None, None, ""

        dtype = resolve_dtype(self.dtype_name)
        device = next(self.model.parameters()).device

        tensors = []
        num_patches_list: List[int] = []
        for p in image_paths:
            pv = load_image(p, input_size=self.input_size, max_num=self.max_num_tiles)
            tensors.append(pv)
            num_patches_list.append(pv.shape[0])

        pixel_values = torch.cat(tensors, dim=0).to(dtype=dtype, device=device)

        if len(image_paths) == 1:
            prefix = "<image>\n"
        else:
            prefix = "".join(
                f"Image-{i + 1}: <image>\n" for i in range(len(image_paths))
            )
        return pixel_values, num_patches_list, prefix

    def infer(
        self,
        image_paths: List[str],
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.is_loaded:
            logger.error("[InternVLBackend] 모델이 로드되지 않았습니다 — load() 먼저 호출하세요.")
            return ""

        try:
            imgs = valid_image_paths(image_paths)
            pixel_values, num_patches_list, prefix = self._prepare_pixel_values(imgs)

            gen_cfg = self._gen_kwargs(max_new_tokens, temperature)
            # .chat() 은 generation_config 로 dict 를 받는다
            generation_config: Dict[str, Any] = {
                "max_new_tokens": gen_cfg["max_new_tokens"],
                "do_sample": gen_cfg["do_sample"],
            }
            if gen_cfg["do_sample"]:
                generation_config["temperature"] = gen_cfg["temperature"]
                generation_config["top_p"] = gen_cfg["top_p"]

            chat_kwargs: Dict[str, Any] = {}
            if num_patches_list is not None and len(num_patches_list) > 1:
                chat_kwargs["num_patches_list"] = num_patches_list

            response = self.model.chat(
                self.tokenizer,
                pixel_values,            # 텍스트 전용이면 None
                prefix + question,
                generation_config,
                **chat_kwargs,
            )
            return (response or "").strip()

        except Exception as e:
            logger.error(f"[InternVLBackend] 추론 실패: {e}")
            return ""

    # ── 학습 (DPO) ────────────────────────────────────────────────────────

    def load_for_train(
        self,
        adapter_path: Optional[str] = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """학습용 로드 — LLM 쪽에만 LoRA 를 붙이고 InternViT 는 freeze 한다."""
        try:
            import torch
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import AutoModel, AutoTokenizer

            _apply_transformers_compat_shims()
            cfg = dict(lora_cfg or {})

            kw: Dict[str, Any] = {
                dtype_kwarg_name(): resolve_dtype(self.dtype_name),
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_flash_attn": bool(
                    self.use_flash_attn and self._attn_implementation() is not None
                ),
            }
            if self.device_map:
                kw["device_map"] = self.device_map

            logger.info(f"[InternVLBackend] 학습용 모델 로드: {self.model_name}")
            base = AutoModel.from_pretrained(self.model_name, **kw)
            if not self.device_map:
                base = base.to("cuda" if torch.cuda.is_available() else "cpu")

            # InternViT(비전 타워) freeze — LoRA 타깃에 없더라도 명시적으로 잠근다
            vision = getattr(base, "vision_model", None)
            if vision is not None:
                for p in vision.parameters():
                    p.requires_grad = False
                logger.info("[InternVLBackend] InternViT freeze 완료")

            if adapter_path and Path(adapter_path).exists():
                self.model = PeftModel.from_pretrained(
                    base, adapter_path, is_trainable=True
                )
            else:
                targets = cfg.get("target_modules") or self.default_lora_targets()
                self.model = get_peft_model(base, LoraConfig(
                    r=int(cfg.get("r", 16)),
                    lora_alpha=int(cfg.get("lora_alpha", 32)),
                    lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                    target_modules=list(targets),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ))

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )
            self.adapter_path = adapter_path
            self.model.train()
            return True

        except Exception as e:
            logger.error(f"[InternVLBackend] 학습용 로드 실패: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def set_adapter_enabled(self, enabled: bool) -> None:
        """LoRA on/off 토글 → policy(on) / reference(off)."""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        base = getattr(self.model, "base_model", None)
        if base is None or not hasattr(base, "enable_adapter_layers"):
            raise RuntimeError(
                "LoRA 어댑터가 붙어 있지 않습니다 — load_for_train() 을 먼저 호출하세요."
            )
        base.enable_adapter_layers() if enabled else base.disable_adapter_layers()

    def logprob(
        self,
        image_paths: List[str],
        question: str,
        answer: str,
    ) -> Any:
        """
        log P(answer | image, question) — 답변 토큰의 합 log-prob.

        `.chat()` 은 생성 전용이라 우회하고, InternVL 이 내부에서 쓰는 방식대로
        `<image>` 자리표시자를 이미지 토큰으로 확장한 뒤 forward 를 직접 호출한다.
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        if not answer:
            raise ValueError("답변이 비어 있어 log-prob 을 계산할 수 없습니다.")

        imgs = valid_image_paths(image_paths)
        pixel_values, num_patches_list, prefix = self._prepare_pixel_values(imgs)

        # PeftModel 로 감싸여 있어도 InternVL 고유 속성은 base 쪽에 있다
        core = self.model
        while hasattr(core, "base_model") and not hasattr(core, "img_context_token_id"):
            core = core.base_model
            if hasattr(core, "model") and not hasattr(core, "img_context_token_id"):
                core = core.model

        img_token = "<IMG_CONTEXT>"
        img_ctx_id = self.tokenizer.convert_tokens_to_ids(img_token)
        if hasattr(core, "img_context_token_id"):
            core.img_context_token_id = img_ctx_id

        # InternVL 대화 템플릿 (내부 conversation 템플릿과 동일한 형태)
        prompt = f"<|im_start|>user\n{prefix}{question}<|im_end|><|im_start|>assistant\n"
        if pixel_values is not None:
            n_tokens = getattr(core, "num_image_token", 256)
            for n_patch in (num_patches_list or [pixel_values.shape[0]]):
                prompt = prompt.replace(
                    "<image>", img_token * (n_tokens * n_patch), 1
                )

        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        answer_ids = self.tokenizer(
            answer, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if num_patches_list:
            kwargs["image_flags"] = torch.ones(
                (pixel_values.shape[0], 1), dtype=torch.long, device=device
            )

        out = self.model(**kwargs)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        mask = labels[:, 1:] != -100
        logp = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        return (logp * mask).sum()

    def save_adapter(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"[InternVLBackend] 어댑터 저장 → {path}")

    # ── 해제 / 기본값 ─────────────────────────────────────────────────────

    def unload(self) -> None:
        logger.info("[InternVLBackend] 모델 언로드")
        self.model = None
        self.tokenizer = None
        self.adapter_path = None
        self._free_cuda()

    def default_lora_targets(self) -> List[str]:
        """LLM 쪽 attention projection 만. InternViT(비전 타워)는 freeze."""
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
