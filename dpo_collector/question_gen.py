"""
dpo_collector/question_gen.py
------------------------------
KG 경로를 근거로 **한국어** 후보 질문 2~3개를 생성한다 (Challenger).

코어 `self_play/question_generator.py` 의 설계를 참고하되:
  - 프롬프트가 전부 영어라 그대로 쓸 수 없어 **한국어로 재구성**했다.
  - 엣지 타입 → 난이도/유형 매핑(`EDGE_TO_DIFFICULTY` / `EDGE_TO_QTYPE`)은
    `kg_bridge.py` 를 통해 코어 것을 그대로 재사용한다.

프롬프트는 `config_dpo.yaml::question_gen.prompt_template` 로 통째로 덮어쓸 수 있고,
미지정이면 엣지 타입별 기본 템플릿을 쓴다. 언어는 `language` 설정을 따른다
(하드코딩 금지 — §1-3).

생성 실패(모델 미로드/빈 출력) 시 **규칙 기반 템플릿 질문**으로 폴백해 수집 UI 가
멈추지 않게 한다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends import get_shared_backend
from .kg_bridge import KGPath, clean_content

logger = logging.getLogger(__name__)

#: 언어 코드 → 프롬프트에 넣을 언어 이름
_LANG_NAMES = {"ko": "한국어", "en": "English", "ja": "日本語", "zh": "中文"}

#: 모델이 붙이는 군더더기 접두사 (한/영 모두)
_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:질문|문제|Question|Q)\s*(?:\d+)?\s*[:：.)\]]\s*",
    re.IGNORECASE,
)
#: "1. ", "1) ", "- ", "* " 같은 목록 기호
_BULLET_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")

#: 질문 후보로 인정할 길이 범위 (너무 짧으면 무의미, 너무 길면 설명이 섞인 것)
_MIN_LEN, _MAX_LEN = 6, 400


# ─── 엣지 타입별 한국어 프롬프트 ───────────────────────────────────────────
#
# 공통 규칙:
#   · KG 경로를 모르는 사람도 답할 수 있는 self-contained 질문이어야 한다.
#   · 이미지가 있으면 이미지를 봐야 답할 수 있는 질문을 우선한다.
#   · 질문 문장 하나만 출력한다 (설명/번호/따옴표 금지).

_COMMON_RULES = """규칙:
- 반드시 {language}로 질문 1개만 작성하세요.
- 지식 그래프나 아래 자료를 보지 못한 사람도 이해할 수 있는 완결된 질문이어야 합니다.
- "위 내용에 따르면", "주어진 그래프에서" 같은 표현을 쓰지 마세요.
- 설명, 번호, 따옴표 없이 질문 문장만 출력하세요."""

_DEFAULT_PROMPTS: Dict[str, str] = {
    # 단일 노드 (난이도 1) — 그림/표 자체를 묻는다
    "SINGLE": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

자료 유형: {src_type}
자료 내용: {src_content}
{image_hint}

이 자료가 무엇을 보여주는지, 또는 어떤 정보를 담고 있는지 묻는 질문을 만드세요.
{common_rules}

질문:""",

    "HAS_CAPTION": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

[{src_type}] 자료와 그 설명(캡션)이 주어졌습니다.
캡션: {dst_content}
{image_hint}

이미지를 보고 답할 수 있으면서, 캡션이 설명하는 대상의 내용·구조·목적을 묻는 질문을 만드세요.
{common_rules}

질문:""",

    "REFERENCES": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

본문 내용: {src_content}
본문이 참조하는 [{dst_type}]: {dst_content}
{image_hint}

본문과 참조 대상의 관계를 이해해야 답할 수 있는 질문을 만드세요.
(예: 본문의 주장을 그림/표가 어떻게 뒷받침하는가)
{common_rules}

질문:""",

    "QUANTIFIES": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

[{src_type}]: {src_content}
수치가 담긴 [{dst_type}]: {dst_content}
{image_hint}

구체적인 수치나 측정값을 답으로 요구하는 질문을 만드세요.
{common_rules}

질문:""",

    "DEFINES": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

[{src_type}]: {src_content}
정의되는 개념 [{dst_type}]: {dst_content}
{image_hint}

해당 개념이 이 문서에서 어떻게 정의·설명되는지 묻는 질문을 만드세요.
{common_rules}

질문:""",

    "COMPARES": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

비교 대상 A [{src_type}]: {src_content}
비교 대상 B [{dst_type}]: {dst_content}
{image_hint}

A와 B를 비교해야 답할 수 있는 객관식 질문을 만드세요.
질문 문장 다음 줄에 선택지 4개를 "A) ~", "B) ~", "C) ~", "D) ~" 형식으로 제시하세요.
- 반드시 {language}로 작성하세요.
- 정답이나 해설은 출력하지 마세요.

질문:""",

    "SUPPORTS": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

근거 [{src_type}]: {src_content}
뒷받침되는 주장 [{dst_type}]: {dst_content}
{image_hint}

근거가 주장을 **왜** 뒷받침하는지 이해해야 답할 수 있는 질문을 만드세요.
{common_rules}

질문:""",

    "CONTRADICTS": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

내용 A [{src_type}]: {src_content}
상충하는 내용 B [{dst_type}]: {dst_content}
{image_hint}

A와 B 사이의 차이 또는 상충 지점을 짚어야 답할 수 있는 질문을 만드세요.
{common_rules}

질문:""",

    "SAME_CONCEPT": """당신은 여러 논문을 함께 읽고 질문을 만드는 출제자입니다.

논문 1의 내용 [{src_type}]: {src_content}
논문 2의 내용 [{dst_type}]: {dst_content}
{image_hint}

두 논문이 같은 개념을 어떻게 다르게 다루는지 비교해야 답할 수 있는 질문을 만드세요.
{common_rules}

질문:""",

    "DEFAULT": """당신은 과학기술 문서를 기반으로 질문을 만드는 출제자입니다.

[{src_type}]: {src_content}
--{edge_type}--> [{dst_type}]: {dst_content}
{image_hint}

두 자료의 관계를 이해해야 답할 수 있는 질문을 만드세요.
{common_rules}

질문:""",
}

def josa(word: str, with_final: str, without_final: str) -> str:
    """
    한국어 조사 선택 — 마지막 글자의 받침 유무로 고른다.
    예: josa("그림", "은", "는") → "은",  josa("표", "은", "는") → "는"
    한글이 아니면(영문/숫자) 받침 없는 형태를 쓴다.
    """
    if not word:
        return without_final
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_final if (ord(ch) - 0xAC00) % 28 else without_final
    return without_final


#: 모델 없이도 수집을 이어갈 수 있게 하는 규칙 기반 폴백 질문
#: `{src_type_ko}` 등은 조사까지 포함해 미리 조립된 문자열이 들어간다.
_FALLBACK_TEMPLATES: Dict[str, str] = {
    "HAS_CAPTION":  "이 {src_eun} 무엇을 보여주는지 설명해 주세요.",
    "REFERENCES":   "본문에서 언급한 내용을 이 {dst_i} 어떻게 뒷받침하는지 설명해 주세요.",
    "QUANTIFIES":   "이 자료에 제시된 주요 수치와 그 의미를 설명해 주세요.",
    "DEFINES":      "이 문서에서 해당 개념을 어떻게 정의하고 있는지 설명해 주세요.",
    "COMPARES":     "이 자료에서 비교되는 두 대상의 차이를 설명해 주세요.",
    "SUPPORTS":     "이 근거가 주장을 어떻게 뒷받침하는지 설명해 주세요.",
    "CONTRADICTS":  "이 두 내용 사이의 차이점은 무엇인지 설명해 주세요.",
    "SAME_CONCEPT": "두 논문이 이 개념을 각각 어떻게 다루는지 비교해 주세요.",
    "SINGLE":       "이 {src_eun} 무엇을 보여주는지 설명해 주세요.",
    "DEFAULT":      "이 자료들이 어떤 관계인지 설명해 주세요.",
}

#: 노드 타입 → 한국어 표기 (폴백 질문용)
_NODE_TYPE_KO = {
    "Figure": "그림", "Table": "표", "Equation": "수식",
    "TextBlock": "본문", "Concept": "개념", "Claim": "주장", "Paper": "논문",
}


# ─── 질문 후보 ─────────────────────────────────────────────────────────────

@dataclass
class QuestionCandidate:
    """생성된 후보 질문 1개 + 근거."""

    text:          str
    question_type: str = "VQA"
    difficulty:    int = 1
    source:        str = "auto"        # "auto" (모델 생성) | "fallback" (규칙 기반)
    image_paths:   List[str] = field(default_factory=list)
    gold_answer:   str = ""
    paper_id:      str = ""
    kg_provenance: Dict[str, Any] = field(default_factory=dict)
    context:       str = ""

    def to_store_kwargs(self) -> Dict[str, Any]:
        """`DPOPairStore.add()` 에 그대로 넘길 수 있는 인자 묶음."""
        return {
            "question": self.text,
            "paper_id": self.paper_id,
            "image_paths": list(self.image_paths),
            "question_source": "auto",
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "kg_provenance": self.kg_provenance,
            "gold_answer": self.gold_answer,
        }


# ─── 생성기 ────────────────────────────────────────────────────────────────

class QuestionGenerator:
    """
    KG 경로 → 한국어 후보 질문.

        qg = QuestionGenerator(qg_cfg, language="ko")
        cands = qg.generate(kg_path, n=3)
    """

    def __init__(
        self,
        qg_cfg: Dict[str, Any],
        language: str = "ko",
    ):
        self.cfg = dict(qg_cfg or {})
        self.language = language
        self.n_candidates = int(self.cfg.get("n_candidates", 3))
        self.use_kg = bool(self.cfg.get("use_kg", True))
        self.prompt_template: Optional[str] = self.cfg.get("prompt_template") or None
        self._backend = None

    # ── 모델 ──────────────────────────────────────────────────────────────

    @property
    def language_name(self) -> str:
        return _LANG_NAMES.get(self.language, self.language)

    def load(self) -> bool:
        """Challenger 모델 로드. 실패해도 예외를 던지지 않고 폴백 경로를 쓴다."""
        if self._backend is not None and self._backend.is_loaded:
            return True
        try:
            self._backend = get_shared_backend(self.cfg, autoload=True)
            return True
        except Exception as e:
            logger.warning(f"[QGen] 질문 생성 모델 로드 실패 → 템플릿 폴백 사용: {e}")
            self._backend = None
            return False

    def unload(self) -> None:
        self._backend = None

    # ── 생성 ──────────────────────────────────────────────────────────────

    def generate(
        self,
        path: KGPath,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[QuestionCandidate]:
        """
        KG 경로 하나에서 후보 질문 n개를 만든다.

        후보는 개별 호출로 뽑는다 — 한 번에 N개를 요구하면 출력 형식이 자주 무너져
        파싱이 불안정해진다. 중복/빈 출력은 걸러내고, 모두 실패하면 폴백 1개를 돌려준다.
        """
        n = n or self.n_candidates
        if not self.load():
            return [self._fallback(path)]

        prompt = self.build_prompt(path)
        temp = temperature if temperature is not None \
            else float(self.cfg.get("temperature", 0.8))
        max_new = int(self.cfg.get("max_new_tokens", 256))

        seen: set = set()
        out: List[QuestionCandidate] = []
        # 중복/빈 출력을 감안해 여유 있게 시도한다
        for attempt in range(n * 3):
            if len(out) >= n:
                break
            raw = self._backend.infer(
                path.image_paths, prompt,
                max_new_tokens=max_new,
                # 첫 후보는 낮은 온도로 안정적으로, 이후엔 다양성을 위해 올린다
                temperature=temp if attempt else max(0.3, temp - 0.3),
            )
            text = self.clean_question(raw)
            if not text:
                continue
            key = re.sub(r"\s+", "", text)
            if key in seen:
                continue
            seen.add(key)
            out.append(self._to_candidate(path, text, source="auto"))

        if not out:
            logger.warning("[QGen] 모델이 유효한 질문을 만들지 못했습니다 — 폴백 사용")
            return [self._fallback(path)]

        logger.info(
            f"[QGen] 후보 질문 {len(out)}개 생성 "
            f"(요청 {n}, type={out[0].question_type}, L{out[0].difficulty})"
        )
        return out

    # ── 프롬프트 ──────────────────────────────────────────────────────────

    def build_prompt(self, path: KGPath) -> str:
        """
        경로에서 프롬프트를 만든다.
        `config_dpo.yaml::question_gen.prompt_template` 이 있으면 그것을 쓰고,
        없으면 엣지 타입별 기본 템플릿을 쓴다.
        """
        src = path.nodes[0]
        dst = path.nodes[-1]
        primary = path.edge_types[0] if path.edge_types else ""

        fields = {
            "language":     self.language_name,
            "common_rules": _COMMON_RULES.format(language=self.language_name),
            "src_type":     src.node_type.value,
            "dst_type":     dst.node_type.value,
            "src_content":  clean_content(src, max_len=500) or "(내용 없음)",
            "dst_content":  clean_content(dst, max_len=500) or "(내용 없음)",
            "edge_type":    primary or "SINGLE",
            "difficulty":   path.difficulty,
            "gold_answer":  path.gold_answer[:200],
            "image_hint":   "[이미지가 함께 제공됩니다]" if path.image_paths else "",
            "kg_context":   path.context if self.use_kg else "",
        }

        if self.prompt_template:
            template = self.prompt_template
        elif not path.edge_types:
            template = _DEFAULT_PROMPTS["SINGLE"]
        else:
            template = _DEFAULT_PROMPTS.get(primary, _DEFAULT_PROMPTS["DEFAULT"])

        try:
            return template.format(**fields)
        except KeyError as e:
            # 사용자가 커스텀 템플릿에 미지원 플레이스홀더를 쓴 경우
            logger.warning(
                f"[QGen] 프롬프트 템플릿에 알 수 없는 플레이스홀더 {e} — "
                f"사용 가능: {sorted(fields)}"
            )
            return _DEFAULT_PROMPTS["DEFAULT"].format(**fields)

    # ── 후처리 ────────────────────────────────────────────────────────────

    @staticmethod
    def clean_question(raw: str) -> str:
        """
        모델 출력에서 질문 문장만 남긴다.
        접두사("질문:"), 목록 기호, 감싼 따옴표, 뒤에 붙은 해설을 제거한다.
        """
        if not raw:
            return ""
        text = raw.strip()

        # 코드펜스로 감싸는 경우
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        # 마크다운 강조는 접두사 제거보다 **먼저** 걷어낸다.
        # ("**질문:** 본문" 에서 뒤쪽 `**` 가 남아 목록 기호로 오인되는 것을 막는다)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""

        # 객관식이면 선택지까지 살린다
        is_mcq = any(re.match(r"^[A-D]\)", l) for l in lines)
        if is_mcq:
            kept = []
            for l in lines:
                if re.match(r"^(?:정답|Answer)\s*[:：]", l, re.IGNORECASE):
                    break
                kept.append(_BULLET_RE.sub("", _PREFIX_RE.sub("", l)))
            text = "\n".join(kept).strip()
        else:
            first = _PREFIX_RE.sub("", lines[0])
            first = _BULLET_RE.sub("", first).strip()
            text = first

        text = text.strip().strip('"').strip("'").strip("「」").strip()

        if len(text) < _MIN_LEN or len(text) > _MAX_LEN:
            return ""
        return text

    # ── 후보 조립 / 폴백 ──────────────────────────────────────────────────

    def _to_candidate(
        self, path: KGPath, text: str, source: str
    ) -> QuestionCandidate:
        return QuestionCandidate(
            text=text,
            question_type=path.question_type,
            difficulty=path.difficulty,
            source=source,
            image_paths=list(path.image_paths),
            gold_answer=path.gold_answer,
            paper_id=path.paper_id,
            kg_provenance=path.provenance(),
            context=path.context,
        )

    def _fallback(self, path: KGPath) -> QuestionCandidate:
        """모델이 없거나 실패했을 때 쓰는 규칙 기반 질문."""
        primary = path.edge_types[0] if path.edge_types else "SINGLE"
        template = _FALLBACK_TEMPLATES.get(primary, _FALLBACK_TEMPLATES["DEFAULT"])
        src_ko = _NODE_TYPE_KO.get(path.nodes[0].node_type.value, "자료")
        dst_ko = _NODE_TYPE_KO.get(path.nodes[-1].node_type.value, "자료")
        text = template.format(
            src_eun=src_ko + josa(src_ko, "은", "는"),
            dst_eun=dst_ko + josa(dst_ko, "은", "는"),
            src_i=src_ko + josa(src_ko, "이", "가"),
            dst_i=dst_ko + josa(dst_ko, "이", "가"),
        )
        return self._to_candidate(path, text, source="fallback")


# ─── 설정 파일에서 생성기 만들기 ───────────────────────────────────────────

def from_config(config_path: str = "dpo_collector/config_dpo.yaml") -> QuestionGenerator:
    """`config_dpo.yaml` 의 `question_gen` / `language` 섹션으로 생성기를 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return QuestionGenerator(
        qg_cfg=cfg.get("question_gen", {}),
        language=cfg.get("language", "ko"),
    )
