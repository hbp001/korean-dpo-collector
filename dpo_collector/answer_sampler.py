"""
dpo_collector/answer_sampler.py
--------------------------------
활성 모델(Solver)로 **한국어** 후보 답변 N개를 생성한다.

DPO 페어를 만들려면 후보들이 서로 달라야 한다(`chosen != rejected`). 따라서:
  - 다양성 확보를 위해 `answer_gen.temperature`(기본 0.9)로 샘플링하고,
  - 완전히 같은 답변은 걸러내며,
  - 목표 개수를 못 채우면 온도를 올려 재시도한다.

답변 생성 모델은 `config_dpo.yaml::model` (= 학습 대상)이며, 활성 LoRA 어댑터가 있으면
그것을 얹어 로드한다. 어떤 모델이 후보를 만들었는지는 `state.model_version` 으로
`dpo_pairs.jsonl::model_version` 에 기록된다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends import get_shared_backend

logger = logging.getLogger(__name__)

#: 언어 코드 → 프롬프트에 넣을 언어 이름
_LANG_NAMES = {"ko": "한국어", "en": "English", "ja": "日本語", "zh": "中文"}

#: 기본 답변 프롬프트 — `{language}` / `{question}` 플레이스홀더 사용
_DEFAULT_ANSWER_PROMPT = """다음 질문에 {language}로 답변하세요.
이미지가 함께 제공되면 이미지를 근거로 구체적으로 답하고, 추측한 내용은 단정하지 마세요.

질문: {question}

답변:"""

#: 모델이 붙이는 군더더기 접두사
_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:답변|답|Answer|A)\s*[:：.)\]]\s*", re.IGNORECASE
)


# ─── 후보 묶음 ─────────────────────────────────────────────────────────────

@dataclass
class AnswerBundle:
    """질문 1개에 대한 후보 답변 묶음."""

    question:      str
    candidates:    List[str] = field(default_factory=list)
    image_paths:   List[str] = field(default_factory=list)
    model_version: str = "base"
    temperature:   float = 0.9
    metadata:      Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.candidates)

    @property
    def is_usable(self) -> bool:
        """DPO 페어를 만들려면 서로 다른 후보가 최소 2개 필요하다."""
        return len({c.strip() for c in self.candidates if c.strip()}) >= 2


# ─── 샘플러 ────────────────────────────────────────────────────────────────

class AnswerSampler:
    """
    활성 모델로 후보 답변을 뽑는다.

        sampler = AnswerSampler(model_cfg, answer_cfg, language="ko")
        bundle  = sampler.sample("이 그림은 무엇을 보여주는가?", ["/abs/fig.jpg"], n=3)
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        answer_cfg: Optional[Dict[str, Any]] = None,
        language: str = "ko",
        adapter_path: Optional[str] = None,
    ):
        self.model_cfg = dict(model_cfg or {})
        self.cfg = dict(answer_cfg or {})
        self.language = language
        self.adapter_path = adapter_path

        self.n_candidates = int(self.cfg.get("n_candidates", 3))
        self.temperature = float(self.cfg.get("temperature", 0.9))
        self.max_new_tokens = int(
            self.cfg.get("max_new_tokens", self.model_cfg.get("max_new_tokens", 512))
        )
        self.prompt_template: str = (
            self.cfg.get("prompt_template") or _DEFAULT_ANSWER_PROMPT
        )
        self._backend = None

    # ── 모델 ──────────────────────────────────────────────────────────────

    @property
    def language_name(self) -> str:
        return _LANG_NAMES.get(self.language, self.language)

    def load(self) -> bool:
        """Solver 모델 로드 (활성 어댑터 포함)."""
        if self._backend is not None and self._backend.is_loaded:
            return True
        try:
            self._backend = get_shared_backend(
                self.model_cfg, adapter_path=self.adapter_path, autoload=True
            )
            return True
        except Exception as e:
            logger.error(f"[Sampler] 답변 생성 모델 로드 실패: {e}")
            self._backend = None
            return False

    def unload(self) -> None:
        self._backend = None

    def set_adapter(self, adapter_path: Optional[str]) -> None:
        """
        활성 어댑터를 바꾼다. 다음 `load()` 에서 해당 어댑터로 로드된다
        (공유 캐시 키에 어댑터 경로가 포함되므로 별도 인스턴스가 만들어진다).
        """
        if adapter_path == self.adapter_path:
            return
        self.adapter_path = adapter_path
        self._backend = None
        logger.info(f"[Sampler] 어댑터 변경 → {adapter_path or 'base'}")

    # ── 샘플링 ────────────────────────────────────────────────────────────

    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(
                language=self.language_name, question=question
            )
        except KeyError as e:
            logger.warning(
                f"[Sampler] 답변 프롬프트에 알 수 없는 플레이스홀더 {e} — "
                "사용 가능: {language}, {question}"
            )
            return _DEFAULT_ANSWER_PROMPT.format(
                language=self.language_name, question=question
            )

    def sample(
        self,
        question: str,
        image_paths: Optional[List[str]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        model_version: str = "base",
    ) -> AnswerBundle:
        """
        후보 답변 n개를 생성한다.

        서로 다른 후보를 n개 채우려 시도하고, 부족하면 온도를 단계적으로 올려
        추가 샘플링한다. 그래도 못 채우면 확보된 만큼만 돌려준다
        (`AnswerBundle.is_usable` 로 페어 생성 가능 여부를 확인할 것).
        """
        n = n or self.n_candidates
        base_temp = temperature if temperature is not None else self.temperature
        images = list(image_paths or [])

        bundle = AnswerBundle(
            question=question,
            image_paths=images,
            model_version=model_version,
            temperature=base_temp,
        )
        if not self.load():
            bundle.metadata["error"] = "모델 로드 실패"
            return bundle

        prompt = self.build_prompt(question)
        seen: set = set()

        # 1차: 목표 온도로 n개, 2·3차: 온도를 올려 부족분 보충.
        # `infer_n` 은 백엔드가 배치 샘플링을 지원하면 한 번의 generate 로 n개를 뽑는다.
        for round_idx, temp in enumerate(
            (base_temp, min(1.2, base_temp + 0.2), min(1.5, base_temp + 0.5))
        ):
            if len(bundle.candidates) >= n:
                break
            need = n - len(bundle.candidates)
            raws = self._backend.infer_n(
                images, prompt,
                n=need,
                max_new_tokens=self.max_new_tokens,
                temperature=temp,
            )
            for raw in raws:
                if len(bundle.candidates) >= n:
                    break
                text = self.clean_answer(raw)
                if not text:
                    continue
                key = re.sub(r"\s+", "", text)
                if key in seen:
                    continue
                seen.add(key)
                bundle.candidates.append(text)
            if round_idx > 0 and len(bundle.candidates) < n:
                logger.debug(
                    f"[Sampler] 온도 {temp:.1f} 재시도 후 {len(bundle.candidates)}/{n}"
                )

        bundle.metadata["n_requested"] = n
        if len(bundle.candidates) < n:
            logger.warning(
                f"[Sampler] 서로 다른 후보 {n}개를 채우지 못했습니다 "
                f"({len(bundle.candidates)}개) — 모델이 같은 답변만 내놓는 상태일 수 있습니다."
            )
        else:
            logger.info(
                f"[Sampler] 후보 답변 {len(bundle.candidates)}개 생성 "
                f"(temp={base_temp}, images={len(images)}, model={model_version})"
            )
        return bundle

    # ── 후처리 ────────────────────────────────────────────────────────────

    @staticmethod
    def clean_answer(raw: str) -> str:
        """모델 출력에서 답변 본문만 남긴다."""
        if not raw:
            return ""
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        text = _PREFIX_RE.sub("", text).strip()
        text = text.strip('"').strip("'").strip()
        # 답변이 통째로 잘려 문장이 시작도 못 한 경우
        if len(text) < 2:
            return ""
        return text


# ─── 설정 파일에서 샘플러 만들기 ───────────────────────────────────────────

def from_config(
    config_path: str = "dpo_collector/config_dpo.yaml",
    adapter_path: Optional[str] = None,
) -> AnswerSampler:
    """`config_dpo.yaml` 의 `model` / `answer_gen` / `language` 섹션으로 샘플러를 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return AnswerSampler(
        model_cfg=cfg.get("model", {}),
        answer_cfg=cfg.get("answer_gen", {}),
        language=cfg.get("language", "ko"),
        adapter_path=adapter_path,
    )
