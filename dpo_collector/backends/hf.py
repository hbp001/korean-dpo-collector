"""
dpo_collector/backends/hf.py
-----------------------------
계열 A — HuggingFace 표준 프로세서 백엔드 (**기본 경로**).

`AutoProcessor.apply_chat_template` + `AutoModelForImageTextToText` 를 쓰는
최신 HF VLM 대부분이 여기에 해당한다:
  - Qwen/Qwen2.5-VL-*, Qwen/Qwen3-VL-*
  - OpenGVLab/InternVL3_5-*-HF  (transformers 네이티브 리비전, 커스텀 코드 불필요)
  - LLaVA / Idefics / SmolVLM / PaliGemma / Llama-Vision / Gemma-3 등

코어 `models/model_registry.py::HFChatBackend` 의 로드·추론 패턴을 그대로 따르되,
DPO 수집/학습에 필요한 부분만 다르게 한다:
  - LoRA 어댑터를 **merge 하지 않는다** (registry 는 merge_and_unload 사용).
    → 나중에 `set_adapter_enabled()` 로 policy(on)/reference(off) 토글이 가능해야 함.
  - 단건 추론(`infer`) 중심 — 수집 UI 는 한 번에 한 질문을 다룬다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import VLMBackend, dtype_kwarg_name, resolve_dtype, valid_image_paths

logger = logging.getLogger(__name__)


def _load_auto_vlm_class():
    """transformers 버전에 따른 VLM AutoClass. 5.x 는 Vision2Seq 가 제거됨."""
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


class HFBackend(VLMBackend):
    """HF 표준 프로세서 경로 백엔드."""

    family = "hf"

    # ── 로드 ──────────────────────────────────────────────────────────────

    def load(self, adapter_path: Optional[str] = None) -> bool:
        # 같은 어댑터로 이미 로드돼 있으면 재로드 스킵
        if self.is_loaded and self.adapter_path == adapter_path:
            return True
        if self.is_loaded:
            self.unload()

        try:
            from transformers import AutoProcessor

            auto_cls = _load_auto_vlm_class()

            kw: Dict[str, Any] = {dtype_kwarg_name(): resolve_dtype(self.dtype_name)}
            if self.device_map:
                kw["device_map"] = self.device_map
            attn = self._attn_implementation()
            if attn:
                kw["attn_implementation"] = attn
            if self.model_cfg.get("trust_remote_code", False):
                kw["trust_remote_code"] = True

            logger.info(
                f"[HFBackend] 모델 로드: {self.model_name} "
                f"(dtype={self.dtype_name}, device_map={self.device_map}, attn={attn or 'default'})"
            )
            self.model = auto_cls.from_pretrained(self.model_name, **kw)

            # device_map 미사용 시에만 수동 배치 (device_map 사용 시 .to() 는 오류)
            if not self.device_map:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(dev)

            # LoRA 어댑터 — merge 하지 않음 (adapter on/off 토글 보존)
            if adapter_path:
                if Path(adapter_path).exists():
                    from peft import PeftModel

                    logger.info(f"[HFBackend] LoRA 어댑터 적용: {adapter_path}")
                    self.model = PeftModel.from_pretrained(
                        self.model, adapter_path, is_trainable=False
                    )
                else:
                    logger.warning(
                        f"[HFBackend] 어댑터 경로 없음 → base 모델 사용: {adapter_path}"
                    )
                    adapter_path = None

            self.model.eval()

            proc_kw: Dict[str, Any] = {}
            if self.model_cfg.get("trust_remote_code", False):
                proc_kw["trust_remote_code"] = True
            self.processor = AutoProcessor.from_pretrained(self.model_name, **proc_kw)

            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None and tok.pad_token is None:
                tok.pad_token = tok.eos_token

            self._apply_tiling_cfg()

            self.adapter_path = adapter_path
            logger.info("[HFBackend] 로드 완료")
            return True

        except Exception as e:
            logger.error(f"[HFBackend] 로드 실패: {e}")
            self.model = None
            self.processor = None
            return False

    def _apply_tiling_cfg(self) -> None:
        """
        InternVL 계열(-HF)의 dynamic tiling 상한 설정.
        문서/차트처럼 고해상도 입력에서 성능에 직접 영향을 준다.
        해당 속성이 없는 processor(Qwen 등)에는 아무 영향 없음.
        """
        ip = getattr(self.processor, "image_processor", None)
        if ip is None or not hasattr(ip, "max_patches"):
            return
        ip.crop_to_patches = True
        ip.max_patches = int(self.max_num_tiles)
        logger.info(
            f"[HFBackend] tiling 설정: crop_to_patches=True, max_patches={self.max_num_tiles}"
        )

    # ── 추론 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(question: str, image_paths: List[str]) -> List[Dict[str, Any]]:
        """통일 메시지 포맷 — 이미지 먼저, 텍스트 마지막."""
        content: List[Dict[str, Any]] = [
            {"type": "image", "path": p} for p in image_paths
        ]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def infer(
        self,
        image_paths: List[str],
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.is_loaded:
            logger.error("[HFBackend] 모델이 로드되지 않았습니다 — load() 먼저 호출하세요.")
            return ""

        try:
            import torch

            imgs = valid_image_paths(image_paths)
            convs = [self._build_messages(question, imgs)]

            inputs = self.processor.apply_chat_template(
                convs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, **self._gen_kwargs(max_new_tokens, temperature)
                )

            in_len = inputs["input_ids"].shape[1]
            text = self.processor.batch_decode(
                out[:, in_len:], skip_special_tokens=True
            )[0]
            return text.strip()

        except Exception as e:
            logger.error(f"[HFBackend] 추론 실패: {e}")
            return ""

    def infer_n(
        self,
        image_paths: List[str],
        question: str,
        n: int = 3,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        후보 n개를 **한 번의 generate 호출**로 뽑는다 (`num_return_sequences`).

        기본 구현(`infer()` n회 반복)은 프롬프트 인코딩과 이미지 전처리를 매번 반복해
        후보 3개에 70초가 걸렸다. 이미지 인코딩을 한 번만 하고 디코딩만 n갈래로
        나누면 같은 작업이 크게 빨라진다. 실패하면 순차 방식으로 폴백한다.
        """
        n = max(1, n)
        if n == 1:
            return [self.infer(image_paths, question, max_new_tokens, temperature)]
        if not self.is_loaded:
            logger.error("[HFBackend] 모델이 로드되지 않았습니다 — load() 먼저 호출하세요.")
            return []

        try:
            import torch

            imgs = valid_image_paths(image_paths)
            convs = [self._build_messages(question, imgs)]
            inputs = self.processor.apply_chat_template(
                convs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }

            gen_kw = self._gen_kwargs(max_new_tokens, temperature)
            # 후보를 여러 개 뽑으려면 샘플링이 켜져 있어야 한다
            if not gen_kw.get("do_sample"):
                gen_kw["do_sample"] = True
                gen_kw.setdefault("temperature", 0.9)
                gen_kw.setdefault("top_p", float(self.model_cfg.get("top_p", 0.9)))
            gen_kw["num_return_sequences"] = n

            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kw)

            in_len = inputs["input_ids"].shape[1]
            texts = self.processor.batch_decode(
                out[:, in_len:], skip_special_tokens=True
            )
            return [t.strip() for t in texts]

        except Exception as e:
            logger.warning(
                f"[HFBackend] 배치 샘플링 실패 ({e}) — 순차 생성으로 폴백"
            )
            return super().infer_n(
                image_paths, question, n, max_new_tokens, temperature
            )

    # ── 학습 (DPO) ────────────────────────────────────────────────────────

    def load_for_train(
        self,
        adapter_path: Optional[str] = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        학습용 로드 — base 모델에 **학습 가능한** LoRA 를 붙인다.

        policy 와 reference 를 따로 로드하지 않는다(§1-6 OOM 방지).
        같은 모델에서 어댑터를 켜면 policy, 끄면 reference 가 된다.
        비전 타워는 freeze — LoRA 타깃이 LLM 쪽 projection 뿐이라 자동으로 그렇게 된다.
        """
        try:
            import torch
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import AutoProcessor

            auto_cls = _load_auto_vlm_class()
            cfg = dict(lora_cfg or {})

            kw: Dict[str, Any] = {dtype_kwarg_name(): resolve_dtype(self.dtype_name)}
            if self.device_map:
                kw["device_map"] = self.device_map
            attn = self._attn_implementation()
            if attn:
                kw["attn_implementation"] = attn

            logger.info(f"[HFBackend] 학습용 모델 로드: {self.model_name}")
            base = auto_cls.from_pretrained(self.model_name, **kw)
            if not self.device_map:
                base = base.to("cuda" if torch.cuda.is_available() else "cpu")

            if adapter_path and Path(adapter_path).exists():
                logger.info(f"[HFBackend] 기존 어댑터 이어서 학습: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    base, adapter_path, is_trainable=True
                )
            else:
                targets = cfg.get("target_modules") or self.default_lora_targets()
                lconf = LoraConfig(
                    r=int(cfg.get("r", 16)),
                    lora_alpha=int(cfg.get("lora_alpha", 32)),
                    lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                    target_modules=list(targets),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                logger.info(f"[HFBackend] 새 LoRA 부착 (targets={targets})")
                self.model = get_peft_model(base, lconf)

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"[HFBackend] 학습 파라미터 {trainable:,} / 전체 {total:,} "
                f"({trainable / max(total, 1):.4%})"
            )

            # ── 메모리 대책 ───────────────────────────────────────────────
            # DPO 는 페어당 forward 4회(policy 2 + reference 2)를 돌리고, policy 쪽
            # 두 그래프를 동시에 들고 있어야 loss 를 만들 수 있다. 고해상도 타일까지
            # 겹치면 활성화 메모리가 그대로 터진다(실측: 이미지 2장×12타일에서 OOM).
            if cfg.get("gradient_checkpointing", True):
                try:
                    self.model.gradient_checkpointing_enable()
                    # LoRA 만 학습할 때는 입력 임베딩에 grad 를 흘려줘야
                    # checkpointing 이 실제로 동작한다
                    if hasattr(self.model, "enable_input_require_grads"):
                        self.model.enable_input_require_grads()
                    if hasattr(self.model, "config"):
                        self.model.config.use_cache = False
                    logger.info("[HFBackend] gradient checkpointing 활성화")
                except Exception as e:
                    logger.warning(f"[HFBackend] gradient checkpointing 실패(무시): {e}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None and tok.pad_token is None:
                tok.pad_token = tok.eos_token

            # 학습 중에는 타일 수를 낮춰 시퀀스 길이를 줄인다 (추론 때 값과 분리)
            train_tiles = cfg.get("max_num_tiles")
            if train_tiles:
                original = self.max_num_tiles
                self.max_num_tiles = int(train_tiles)
                logger.info(
                    f"[HFBackend] 학습용 타일 수 {original} → {self.max_num_tiles}"
                )
            self._apply_tiling_cfg()

            self.adapter_path = adapter_path
            self.model.train()
            return True

        except Exception as e:
            logger.error(f"[HFBackend] 학습용 로드 실패: {e}")
            self.model = None
            self.processor = None
            return False

    def set_adapter_enabled(self, enabled: bool) -> None:
        """
        LoRA on/off 토글 → policy(on) / reference(off).
        `PeftModel.base_model`(LoraModel)의 레이어 토글을 쓴다.
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        base = getattr(self.model, "base_model", None)
        if base is None or not hasattr(base, "enable_adapter_layers"):
            raise RuntimeError(
                "LoRA 어댑터가 붙어 있지 않습니다 — load_for_train() 을 먼저 호출하세요."
            )
        if enabled:
            base.enable_adapter_layers()
        else:
            base.disable_adapter_layers()

    def logprob(
        self,
        image_paths: List[str],
        question: str,
        answer: str,
    ) -> Any:
        """
        log P(answer | image, question) — 답변 토큰의 **합** log-prob.

        DPO 는 시퀀스 단위 log-prob 을 쓴다. 코어 `_train_grpo_manual` 처럼
        `-out.loss`(토큰 평균)를 쓰면 길이로 정규화되어 chosen/rejected 의 길이 차이가
        선호 신호를 덮어쓸 수 있으므로, 여기서는 log_softmax 로 직접 합을 구한다.

        prompt 부분은 라벨에서 -100 으로 마스킹해 답변 토큰만 센다.
        """
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        imgs = valid_image_paths(image_paths)
        convs = [self._build_messages(question, imgs)]

        # 프롬프트(이미지 토큰 확장 포함)를 먼저 인코딩해 경계를 잡는다
        prompt_inputs = self.processor.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        prompt_inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in prompt_inputs.items()
        }
        prompt_ids = prompt_inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]

        tok = self.processor.tokenizer
        answer_ids = tok(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        if answer_ids.shape[1] == 0:
            raise ValueError("답변이 비어 있어 log-prob 을 계산할 수 없습니다.")

        # 긴 답변은 잘라 활성화 메모리를 제한한다.
        # chosen/rejected 에 같은 상한이 적용되므로 선호 비교 자체는 공정하게 유지된다.
        max_ans = int(self.model_cfg.get("max_answer_tokens", 0) or 0)
        if max_ans and answer_ids.shape[1] > max_ans:
            answer_ids = answer_ids[:, :max_ans]

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # 이미지 관련 키(pixel_values, image_grid_thw 등)는 모델마다 달라 그대로 넘긴다
        extra = {
            k: v for k, v in prompt_inputs.items()
            if k not in ("input_ids", "attention_mask")
        }
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **extra
        )

        # next-token 예측이므로 한 칸 밀어서 맞춘다
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
        logger.info(f"[HFBackend] 어댑터 저장 → {path}")

    # ── 해제 / 기본값 ─────────────────────────────────────────────────────

    def unload(self) -> None:
        logger.info("[HFBackend] 모델 언로드")
        self.model = None
        self.processor = None
        self.adapter_path = None
        self._free_cuda()

    def default_lora_targets(self) -> List[str]:
        """
        Qwen / InternVL3.5-HF 모두 LLM 이 Qwen 계열이라 attention projection 이름이 동일.
        비전 타워는 freeze (타깃에 포함하지 않음).
        """
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
