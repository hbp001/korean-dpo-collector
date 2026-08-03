"""
dpo_collector/app.py
---------------------
Gradio 수집 UI.

이번 단계 구현 범위 (CLAUDE_dpo.md §10-5): **KG 구축 탭 + 수집 탭**.
  - 🕸️ KG 구축 : data_ko → KG 구축/로드, 노드·엣지 통계
  - ✍️ 수집    : 모드 A(사용자 질문) / 모드 B(KG 기반 자동 질문)
                → 후보 답변 N개 → chosen/rejected 선택 → 저장

학습 / 추론 / 내보내기 / 설정 탭은 이후 단계에서 추가한다.

실행:
    python scripts/run_dpo_collector.py
    python -m dpo_collector.app --port 7860
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml

from . import config_io
from .answer_sampler import AnswerSampler
from .eval_ko import EvalSetKo, PaperSplit
from .kg_bridge import KGBridge, KGPath, papers_for_collection
from .question_gen import QuestionCandidate, QuestionGenerator
from .state import StateStore
from .store import DPOPairStore, PairValidationError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "dpo_collector/config_dpo.yaml"

#: 후보 선택 라디오의 **고정** 선택지 상한.
#
# ⚠ Gradio 의 `Radio.preprocess` 는 컴포넌트를 정의할 때의 정적 `self.choices` 로
#   입력값을 검증한다. `gr.update(choices=...)` 는 클라이언트 표시만 바꾸고 이 값을
#   갱신하지 않으므로, 동적으로 채운 Radio 를 **입력**으로 쓰면
#   "Value: 0 is not in the list of choices: []" 오류가 난다.
#   따라서 선택지는 상한까지 미리 정의해 두고, 실제 후보 수만큼만 표시를 줄인다.
MAX_QUESTION_CANDIDATES = 5
MAX_ANSWER_CANDIDATES = 6


def _fixed_choices(n: int, prefix: str) -> List[Tuple[str, int]]:
    """`[("후보 1", 0), ("후보 2", 1), …]` 형태의 고정 선택지."""
    return [(f"{prefix} {i + 1}", i) for i in range(n)]


def _visible_choices(n: int, prefix: str) -> Any:
    """생성된 후보 수만큼만 선택지를 노출하는 update."""
    return gr.update(choices=_fixed_choices(n, prefix), value=None)


def _numbered(items: List[str], heading: str) -> str:
    """후보 전문을 번호와 함께 마크다운으로 펼친다 (라디오에는 번호만 보이므로)."""
    if not items:
        return ""
    blocks = [f"**{heading} {i + 1}**\n\n{t}" for i, t in enumerate(items)]
    return "\n\n---\n\n".join(blocks)


# ─── 앱 상태 (프로세스 단위로 공유되는 무거운 객체) ────────────────────────

@dataclass
class AppContext:
    """
    설정 + 모델/저장소 핸들.

    KG·모델은 로드가 무거우므로 프로세스 단위로 공유한다. 반면 **수집 중 화면 상태**
    (현재 후보 질문/답변 등)는 사용자마다 달라야 하므로 `gr.State` 로 세션마다 따로 들고 간다.
    """

    config_path: str = DEFAULT_CONFIG
    cfg: Dict[str, Any] = field(default_factory=dict)
    bridge: Optional[KGBridge] = None
    qgen: Optional[QuestionGenerator] = None
    sampler: Optional[AnswerSampler] = None
    store: Optional[DPOPairStore] = None
    state: Optional[StateStore] = None
    split: Optional[PaperSplit] = None
    evalset: Optional[EvalSetKo] = None

    @classmethod
    def load(cls, config_path: str = DEFAULT_CONFIG) -> "AppContext":
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        paths = cfg.get("paths", {})
        kg_cfg = cfg.get("kg", {})
        language = cfg.get("language", "ko")

        ctx = cls(config_path=config_path, cfg=cfg)
        ctx.bridge = KGBridge(
            data_root=paths.get("data_ko_root", "data_ko"),
            kg_json=paths.get("kg_json"),
            kg_cfg=kg_cfg.get("overrides"),
            augment_korean_refs=bool(kg_cfg.get("augment_korean_refs", True)),
        )
        ctx.store = DPOPairStore(
            pairs_jsonl=paths.get("pairs_jsonl", "dpo_collector/outputs/dpo_pairs.jsonl"),
            export_dir=paths.get("export_dir"),
            language=language,
            export_cfg=cfg.get("export", {}),
        )
        ctx.state = StateStore(
            state_json=paths.get("state_json", "dpo_collector/outputs/state.json"),
            history_json=paths.get("history_json"),
            adapter_dir=paths.get("adapter_dir"),
        )
        # 평가셋 split — 수집 샘플링에서 held-out 논문을 배제하기 위해 필요하다 (§1-4)
        ctx.split = PaperSplit(
            paths.get("splits_json", "dpo_collector/outputs/splits.json")
        )
        ctx.evalset = EvalSetKo(
            eval_jsonl=paths.get("eval_jsonl", "dpo_collector/outputs/eval_ko.jsonl"),
            draft_jsonl=paths.get("eval_draft_jsonl"),
            split=ctx.split,
            require_hangul_gold=bool(
                cfg.get("eval", {}).get("require_hangul_gold", True)
            ),
        )
        ctx.apply_eval_exclusion()

        ctx.qgen = QuestionGenerator(cfg.get("question_gen", {}), language=language)
        ctx.sampler = AnswerSampler(
            model_cfg=cfg.get("model", {}),
            answer_cfg=cfg.get("answer_gen", {}),
            language=language,
            adapter_path=ctx.state.active_adapter,
        )
        return ctx

    def reload(self, config_path: Optional[str] = None) -> str:
        """
        설정 파일을 다시 읽어 내부 핸들을 통째로 교체한다.

        UI 이벤트 핸들러들이 이 `ctx` 객체를 클로저로 붙잡고 있으므로,
        새 인스턴스를 만들어 반환하는 대신 **자기 자신의 필드를 갈아끼운다**.
        그래야 이미 배선된 버튼들이 새 설정을 쓴다.
        """
        from .backends import unload_shared_backends

        path = config_path or self.config_path
        old_model = (self.cfg.get("model", {}) or {}).get("name")
        old_kg = (self.cfg.get("paths", {}) or {}).get("kg_json")

        fresh = AppContext.load(path)
        # bridge 는 KG 를 다시 읽는 비용이 크므로, 경로가 그대로면 기존 것을 유지한다
        keep_kg = (
            old_kg == (fresh.cfg.get("paths", {}) or {}).get("kg_json")
            and self.bridge is not None
            and self.bridge.is_ready
        )
        if keep_kg:
            fresh.bridge = self.bridge
            fresh.apply_eval_exclusion()

        self.__dict__.update(fresh.__dict__)

        notes: List[str] = []
        new_model = (self.cfg.get("model", {}) or {}).get("name")
        if new_model != old_model:
            n = unload_shared_backends()
            notes.append(
                f"모델이 `{old_model}` → `{new_model}` 로 바뀌어 "
                f"로드된 백엔드 {n}개를 내렸습니다 (다음 추론에서 새로 로드)."
            )
        if not keep_kg:
            notes.append("KG 경로가 바뀌었습니다 — **KG 구축 탭에서 다시 로드**하세요.")

        return "\n".join(f"- {n}" for n in notes)

    def apply_eval_exclusion(self) -> int:
        """
        평가 전용(held-out) 논문을 수집 샘플링에서 배제한다.

        KG 는 평가 문항 생성을 위해 held-out 논문까지 포함해 구축하므로,
        배제는 샘플링 단계에서 걸어야 한다. split 이 아직 없으면 아무것도 하지 않는다.
        """
        if self.split is None or self.bridge is None:
            return 0
        try:
            if not self.split.is_initialized:
                return 0
            held = self.split.held_out_papers
            self.bridge.set_excluded_papers(held)
            return len(held)
        except Exception as e:
            logger.warning(f"[UI] 평가셋 배제 적용 실패: {e}")
            return 0

    # ── 진행 상태 요약 ────────────────────────────────────────────────────

    def progress_markdown(self) -> str:
        """수집 카운터 + 학습 트리거 상태 (유형 편향 가드레일 포함)."""
        by_type = self.store.counts_by_type()
        type_line = " · ".join(f"{k} {v}" for k, v in by_type.items() if v)
        head = (
            f"**수집된 페어: {self.store.count()}건**"
            + (f"  ({type_line})" if type_line else "")
        )
        try:
            from .trigger import TrainingTrigger

            decision = TrainingTrigger(
                self.store, self.state, self.cfg.get("train", {})
            ).evaluate()
            body = decision.markdown()
        except Exception as e:
            logger.warning(f"[UI] 트리거 판정 실패: {e}")
            body = f"_트리거 상태를 계산하지 못했습니다: {e}_"

        return f"{head}\n\n{body}\n\n현재 모델: `{self.state.model_version}`"


# ─── KG 구축 탭 핸들러 ─────────────────────────────────────────────────────

def kg_build_or_load(
    ctx: AppContext, max_papers: int, rebuild: bool, progress=gr.Progress()
) -> Tuple[str, Any]:
    """KG 를 구축하거나 저장된 것을 로드하고, 통계 + 논문 드롭다운을 갱신한다."""
    try:
        progress(0.1, desc="논문 목록 확인 중… (언어 판정은 첫 실행에만 시간이 걸립니다)")
        lang = ctx.cfg.get("kg", {}).get("language_filter") or None
        papers = ctx.bridge.list_papers(language=lang)
        if not papers:
            return (
                f"❌ `{ctx.bridge.data_root}` 에서 조건에 맞는 논문을 찾지 못했습니다."
                + (f" (언어 필터: {lang})" if lang else ""),
                gr.update(choices=[], value=None),
            )
        # KG 는 평가 문항 생성을 위해 held-out 논문까지 포함해 구축한다.
        # 수집 샘플링에서의 배제는 아래 apply_eval_exclusion() 이 담당한다.
        selected = papers[: int(max_papers)] if int(max_papers) > 0 else papers

        progress(0.3, desc=f"KG {'구축' if rebuild else '로드/구축'} 중… ({len(selected)}편)")
        if rebuild:
            ctx.bridge.build(paper_ids=selected, save=True)
        else:
            ctx.bridge.build_or_load(paper_ids=selected)

        progress(0.9, desc="통계 집계 중…")
        n_excluded = ctx.apply_eval_exclusion()
        s = ctx.bridge.stats()
        in_graph = ctx.bridge.papers_in_graph()
        # 논문 드롭다운에는 수집 가능한 논문만 노출한다
        held = ctx.bridge.excluded_papers
        selectable = [p for p in in_graph if p not in held]

        node_lines = "\n".join(f"| {k} | {v:,} |" for k, v in s["node_types"].items())
        edge_lines = "\n".join(f"| {k} | {v:,} |" for k, v in s["edge_types"].items())
        lang_line = ""
        try:
            lang_line = "언어별 논문 수: " + ", ".join(
                f"{k} {v:,}" for k, v in ctx.bridge.language_stats().items()
            )
        except Exception:
            pass

        md = (
            f"### ✅ KG 준비 완료 — 논문 {s['total_papers']:,}편\n\n"
            f"**노드 {s['total_nodes']:,}개 / 엣지 {s['total_edges']:,}개** "
            f"(이미지 보유 노드 {s['visual_nodes']:,}개)\n\n"
            f"| 노드 타입 | 개수 |\n|---|---:|\n{node_lines}\n\n"
            f"| 엣지 타입 | 개수 |\n|---|---:|\n{edge_lines}\n\n"
            f"- 한국어 참조 보강 엣지: **{s['korean_ref_edges']:,}개**\n"
            + (f"- {lang_line}\n" if lang_line else "")
            + (
                f"- 🔒 평가 전용 논문 **{n_excluded}편**은 수집 샘플링에서 제외됩니다 "
                f"(수집 가능 {len(selectable):,}편)\n"
                if n_excluded else
                "- ⚠️ 평가셋 split 이 없습니다. `python -m dpo_collector.eval_ko split` 으로 "
                "평가 전용 논문을 먼저 떼어내세요.\n"
            )
        )
        return md, gr.update(
            choices=selectable, value=selectable[0] if selectable else None
        )

    except Exception as e:
        logger.exception("[UI] KG 준비 실패")
        return f"❌ KG 준비 실패: {e}", gr.update()


def load_paper_images(ctx: AppContext, paper_id: str) -> Any:
    """선택한 논문의 Figure/Table 이미지 갤러리."""
    if not paper_id or not ctx.bridge.is_ready:
        return gr.update(value=[])
    items = ctx.bridge.paper_images(paper_id, limit=60)
    return gr.update(value=[(path, node_id.split("__")[-1]) for node_id, path in items])


# ─── 수집 탭 핸들러 ────────────────────────────────────────────────────────

def generate_questions(
    ctx: AppContext,
    paper_id: Optional[str],
    n_candidates: int,
    max_hops: int,
    require_image: bool,
    progress=gr.Progress(),
) -> Tuple[Any, str, Any, Dict[str, Any]]:
    """
    모드 B — KG 경로를 샘플링해 후보 질문을 만든다.

    Returns: (질문 라디오 update, 근거 마크다운, 이미지 갤러리 update, 세션 상태)
    """
    empty = {"candidates": [], "path": None}
    if not ctx.bridge.is_ready:
        return gr.update(choices=[], value=None), "⚠️ 먼저 **KG 구축** 탭에서 KG를 준비하세요.", gr.update(value=[]), empty

    try:
        progress(0.1, desc="KG 경로 샘플링 중…")
        sampling = ctx.cfg.get("kg", {}).get("sampling", {})
        paths = ctx.bridge.sample_paths(
            n=1,
            paper_id=paper_id or None,
            max_hops=int(max_hops),
            prefer_visual=bool(sampling.get("prefer_visual", True)),
            require_image=bool(require_image),
        )
        if not paths:
            return (
                gr.update(choices=[], value=None),
                "⚠️ 조건에 맞는 KG 경로를 찾지 못했습니다. "
                "'이미지 필수'를 끄거나 다른 논문을 선택해 보세요.",
                gr.update(value=[]),
                empty,
            )
        path = paths[0]

        progress(0.4, desc="후보 질문 생성 중… (모델 첫 호출은 로드 시간이 걸립니다)")
        n_want = min(int(n_candidates), MAX_QUESTION_CANDIDATES)
        cands = ctx.qgen.generate(path, n=n_want)[:MAX_QUESTION_CANDIDATES]

        evidence = (
            _numbered([c.text for c in cands], "후보 질문")
            + f"\n\n---\n\n**KG 근거** — {path.summary()}\n\n"
            f"- 논문: `{path.paper_id}`\n"
            f"- 경로: `{' → '.join(path.node_ids)}`\n"
            f"- gold answer(참고용): {path.gold_answer[:200] or '_(없음)_'}\n\n"
            f"```\n{path.context[:1200]}\n```"
        )
        if cands and cands[0].source == "fallback":
            evidence = (
                "⚠️ 질문 생성 모델을 쓰지 못해 **템플릿 질문**으로 대체했습니다.\n\n"
                + evidence
            )

        session = {
            "candidates": [c.__dict__ for c in cands],
            "path_images": list(path.image_paths),
        }
        return (
            _visible_choices(len(cands), "후보 질문"),
            evidence,
            gr.update(value=list(path.image_paths)),
            session,
        )

    except Exception as e:
        logger.exception("[UI] 질문 생성 실패")
        return (
            _visible_choices(0, "후보 질문"),
            f"❌ 질문 생성 실패: {e}",
            gr.update(value=[]),
            empty,
        )


def pick_question(q_session: Dict[str, Any], idx: Optional[int]) -> str:
    """선택한 후보 질문의 전문을 질문 입력창에 넣는다."""
    cands = (q_session or {}).get("candidates", [])
    if idx is None or not cands or not (0 <= int(idx) < len(cands)):
        return ""
    return cands[int(idx)]["text"]


def generate_answers(
    ctx: AppContext,
    question: str,
    images: Optional[List[Any]],
    q_session: Dict[str, Any],
    n_candidates: int,
    temperature: float,
    progress=gr.Progress(),
) -> Tuple[Any, str, Dict[str, Any]]:
    """후보 답변 N개를 생성한다 (모드 A/B 공통)."""
    empty = {"candidates": [], "images": []}
    question = (question or "").strip()
    if not question:
        return _visible_choices(0, "후보"), "⚠️ 질문을 입력하거나 생성하세요.", empty

    image_paths = _normalize_images(images)
    if not image_paths:
        # 갤러리 값이 비어 오는 경우가 있어(컴포넌트 왕복 중 유실) 모드 B 는
        # 질문 생성 때 확정된 KG 경로 이미지를 폴백으로 쓴다.
        fallback = (q_session or {}).get("path_images") or []
        image_paths = [p for p in fallback if Path(p).is_file()]
        if image_paths:
            logger.info(
                f"[UI] 갤러리가 비어 KG 경로 이미지 {len(image_paths)}장을 사용합니다."
            )
    try:
        progress(0.2, desc="후보 답변 생성 중… (모델 첫 호출은 로드 시간이 걸립니다)")
        bundle = ctx.sampler.sample(
            question,
            image_paths,
            n=min(int(n_candidates), MAX_ANSWER_CANDIDATES),
            temperature=float(temperature),
            model_version=ctx.state.model_version,
        )
        bundle.candidates = bundle.candidates[:MAX_ANSWER_CANDIDATES]
        if bundle.n == 0:
            return (
                _visible_choices(0, "후보"),
                f"❌ 답변을 생성하지 못했습니다. {bundle.metadata.get('error', '')}",
                empty,
            )

        detail = _numbered(bundle.candidates, "후보")
        if not bundle.is_usable:
            detail = (
                "⚠️ 서로 다른 후보가 2개 미만입니다 — 이대로는 DPO 페어를 만들 수 없습니다. "
                "temperature 를 올리고 다시 생성해 보세요.\n\n---\n\n" + detail
            )
        session = {
            "candidates": list(bundle.candidates),
            "images": image_paths,
            "model_version": bundle.model_version,
        }
        return _visible_choices(bundle.n, "후보"), detail, session

    except Exception as e:
        logger.exception("[UI] 답변 생성 실패")
        return _visible_choices(0, "후보"), f"❌ 답변 생성 실패: {e}", empty


def _normalize_images(images: Optional[List[Any]]) -> List[str]:
    """
    Gradio 갤러리/파일 컴포넌트가 돌려주는 값에서 이미지 경로 리스트를 뽑는다.
    (버전에 따라 str / (path, caption) 튜플 / dict 형태가 섞여 나온다)
    """
    out: List[str] = []
    for item in images or []:
        path: Optional[str] = None
        if isinstance(item, str):
            path = item
        elif isinstance(item, (tuple, list)) and item:
            first = item[0]
            path = first if isinstance(first, str) else getattr(first, "path", None)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("name") or (item.get("image") or {}).get("path")
        else:
            path = getattr(item, "path", None) or getattr(item, "name", None)
        if path and Path(path).is_file():
            out.append(str(Path(path).resolve()))
    return out


def save_pair(
    ctx: AppContext,
    question: str,
    a_session: Dict[str, Any],
    q_session: Dict[str, Any],
    chosen_idx: Optional[int],
    rejected_idx: Optional[int],
    question_source: str,
    annotator: str,
    notes: str,
) -> Tuple[str, str]:
    """chosen/rejected 를 확정해 페어를 저장한다."""
    cands = (a_session or {}).get("candidates", [])
    if not cands:
        return "⚠️ 먼저 후보 답변을 생성하세요.", ctx.progress_markdown()
    if chosen_idx is None or rejected_idx is None:
        return "⚠️ chosen 과 rejected 를 모두 선택하세요.", ctx.progress_markdown()
    if int(chosen_idx) == int(rejected_idx):
        return "⚠️ chosen 과 rejected 는 서로 달라야 합니다.", ctx.progress_markdown()

    # 모드 B 로 만든 질문이면 KG 출처를 함께 저장한다
    meta: Dict[str, Any] = {
        "question_type": "VQA",
        "difficulty": 1,
        "kg_provenance": {},
        "gold_answer": "",
        "paper_id": "",
    }
    picked = _matching_question_meta(q_session, question)
    if picked:
        meta.update({
            "question_type": picked.get("question_type", "VQA"),
            "difficulty": int(picked.get("difficulty", 1)),
            "kg_provenance": picked.get("kg_provenance", {}),
            "gold_answer": picked.get("gold_answer", ""),
            "paper_id": picked.get("paper_id", ""),
        })

    try:
        pair = ctx.store.add(
            question=question,
            candidates=cands,
            chosen_idx=int(chosen_idx),
            rejected_idx=int(rejected_idx),
            image_paths=a_session.get("images", []),
            question_source=question_source,
            annotator=(annotator or "user").strip() or "user",
            model_version=a_session.get("model_version", ctx.state.model_version),
            notes=notes or "",
            **meta,
        )
        ctx.state.bump_pair_count()
        msg = (
            f"✅ 저장 완료 — `{pair.pair_id}` "
            f"({pair.question_type}, 난이도 {pair.difficulty}, 이미지 {len(pair.image_paths)}장)"
        )
        return msg, ctx.progress_markdown()

    except PairValidationError as e:
        return f"❌ 저장 거부: {e}", ctx.progress_markdown()
    except Exception as e:
        logger.exception("[UI] 페어 저장 실패")
        return f"❌ 저장 실패: {e}", ctx.progress_markdown()


def _matching_question_meta(
    q_session: Dict[str, Any], question: str
) -> Optional[Dict[str, Any]]:
    """
    질문 입력창의 텍스트가 자동 생성 후보 중 하나와 일치하면 그 메타를 돌려준다.
    (사용자가 생성된 질문을 손본 경우에는 KG 출처를 붙이지 않는다 — 근거가 달라졌을 수 있음)
    """
    q = " ".join((question or "").split())
    for c in (q_session or {}).get("candidates", []):
        if " ".join(c.get("text", "").split()) == q:
            return c
    return None


def skip_and_next(ctx: AppContext) -> str:
    """현재 후보를 저장하지 않고 넘어갈 때의 안내."""
    return "↩️ 저장하지 않고 다음 항목으로 넘어갑니다."


# ─── 다음 항목으로 이동 ────────────────────────────────────────────────────
#
# 수집은 "질문 생성 → 답변 생성 → 선택 → 저장"을 수백 번 반복하는 작업이라,
# 한 건을 끝낸 뒤 손이 멈추지 않는 것이 중요하다. 저장 직후 다음 근거를 바로
# 띄우고, 이전 후보/선택은 확실히 비워 **직전 답변을 실수로 다시 저장하는 것**을 막는다.

def clear_collection_state() -> Tuple[Any, ...]:
    """
    수집 화면의 후보/선택/메모를 비운다.

    Returns: (질문 라디오, 질문 입력, chosen, rejected, 답변 상세, 메모, 답변 세션)
    """
    return (
        _visible_choices(0, "후보 질문"),
        gr.update(value=""),
        _visible_choices(0, "후보"),
        _visible_choices(0, "후보"),
        gr.update(value=""),
        gr.update(value=""),
        {"candidates": [], "images": []},
    )


def advance_paper(
    ctx: AppContext, current: Optional[str], step: int = 1
) -> Tuple[Any, str]:
    """
    논문 드롭다운을 다음(또는 이전) 논문으로 옮긴다.

    수집 가능한 논문(평가 전용 제외) 안에서만 순환하며, 끝에 닿으면 처음으로 돌아간다.
    """
    if not ctx.bridge or not ctx.bridge.is_ready:
        return gr.update(), "⚠️ 먼저 **KG 구축** 탭에서 KG를 준비하세요."

    held = ctx.bridge.excluded_papers
    papers = [p for p in ctx.bridge.papers_in_graph() if p not in held]
    if not papers:
        return gr.update(), "⚠️ 수집 가능한 논문이 없습니다."

    if current in papers:
        idx = (papers.index(current) + step) % len(papers)
    else:
        idx = 0
    nxt = papers[idx]
    return gr.update(value=nxt), f"📄 논문 이동 → `{nxt}`  ({idx + 1}/{len(papers)})"


def save_result_ok(message: str) -> bool:
    """`save_pair` 가 돌려준 메시지가 성공인지 (자동 진행 여부 판단)."""
    return message.strip().startswith("✅")


def maybe_advance(
    ctx: AppContext,
    save_message: str,
    auto_next: bool,
    paper_id: Optional[str],
    n_candidates: int,
    max_hops: int,
    require_image: bool,
    q_session: Dict[str, Any],
    progress=gr.Progress(),
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    저장 결과에 따라 다음 질문을 띄운다.
    반환 형태는 `generate_questions` 와 같다 (질문 라디오, 근거, 갤러리, 질문 세션).

    **저장에 실패했으면 넘어가지 않는다** — 중복이나 검증 실패로 저장되지 않았는데
    화면만 다음으로 넘어가면 수집자가 저장된 줄 알고 지나치게 된다.
    """
    if not auto_next or not save_result_ok(save_message):
        return gr.update(), gr.update(), gr.update(), (q_session or {"candidates": []})
    return generate_questions(
        ctx, paper_id, n_candidates, max_hops, require_image, progress
    )


# ─── 🎯 학습 탭 ────────────────────────────────────────────────────────────
#
# 차트 색상은 dataviz 기준 팔레트의 categorical slot 1~3 을 고정 순서로 쓴다.
# 이 세 슬롯은 all-pairs CVD/명도 게이트를 통과한 조합이다. 다만 aqua(slot 3)는
# 밝은 배경에서 대비 3:1 미만이라 **표(table view)를 항상 함께 제공**한다(relief rule).
_SERIES_COLORS = {
    "anls": "#2a78d6",      # slot 1 blue
    "accuracy": "#eb6834",  # slot 2 orange
    "f1": "#1baf7a",        # slot 3 aqua
}
_LOSS_COLOR = "#2a78d6"


def _history_frames(ctx: AppContext):
    """
    학습 이력 → (지표 long-form DF, loss DF, 표시용 표 DF).

    지표와 loss 는 **서로 다른 차트**로 나눈다. 축이 하나여야 한다는 원칙에 따라
    스케일이 다른 두 측정을 한 그림에 겹치지 않는다.
    """
    import pandas as pd

    records = ctx.state.history()
    if not records:
        empty = pd.DataFrame({"checkpoint": [], "value": [], "metric": []})
        return empty, pd.DataFrame({"checkpoint": [], "loss": []}), pd.DataFrame()

    metric_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    for r in records:
        loss_rows.append({"checkpoint": r.checkpoint, "loss": r.loss})
        row: Dict[str, Any] = {
            "체크포인트": r.checkpoint,
            "step": r.step,
            "loss": r.loss,
            "학습 페어": r.n_train_pairs,
            "시각": r.timestamp,
        }
        for key in ("anls", "accuracy", "f1"):
            val = r.eval.get(key)
            row[key.upper()] = "—" if val is None else round(float(val), 4)
            if val is not None:
                metric_rows.append({
                    "checkpoint": r.checkpoint,
                    "metric": key,
                    "value": float(val),
                })
        table_rows.append(row)

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(loss_rows),
        pd.DataFrame(table_rows),
    )


def refresh_training_view(ctx: AppContext):
    """학습 탭 전체 갱신 — 트리거 상태 / 차트 2개 / 이력표 / 어댑터 목록."""
    import pandas as pd

    try:
        from .trigger import TrainingTrigger

        decision = TrainingTrigger(
            ctx.store, ctx.state, ctx.cfg.get("train", {})
        ).evaluate()
        status = decision.markdown()
    except Exception as e:
        status = f"트리거 상태 계산 실패: {e}"

    metrics_df, loss_df, table_df = _history_frames(ctx)
    adapters = ctx.state.list_adapters()
    adapter_rows = pd.DataFrame([
        {
            "어댑터": a["name"],
            "활성": "✅" if a["is_active"] else "",
            "생성": a["created_at"],
        }
        for a in adapters
    ])
    choices = [a["name"] for a in adapters]
    active = next((a["name"] for a in adapters if a["is_active"]), None)

    return (
        status,
        gr.update(value=metrics_df),
        gr.update(value=loss_df),
        gr.update(value=table_df),
        gr.update(value=adapter_rows),
        gr.update(choices=choices, value=active),
    )


def run_training(
    ctx: AppContext,
    force: bool,
    limit: Optional[float],
    progress=gr.Progress(),
) -> str:
    """
    "지금 학습" — DPO 학습을 실행한다.

    학습은 수 분~수십 분이 걸릴 수 있고 그동안 GPU 를 독점한다.
    (추론용 공유 백엔드는 dpo_train 이 시작 전에 언로드한다)
    """
    from .dpo_train import make_eval_fn, record_training, run_dpo_training
    from .trigger import TrainingTrigger

    try:
        trigger = TrainingTrigger(ctx.store, ctx.state, ctx.cfg.get("train", {}))
        decision = trigger.evaluate()
        if not decision.should_train and not force:
            return (
                f"⏸️ 학습 조건 미충족 — {decision.reason}\n\n"
                "임계치와 무관하게 지금 돌리려면 **강제 실행**을 켜세요."
            )

        n_limit = int(limit) if limit and int(limit) > 0 else None
        pairs = trigger.training_pairs(limit=n_limit)
        if not pairs:
            return "❌ 학습할 유효 페어가 없습니다."

        progress(0.01, desc=f"학습 준비 중… (페어 {len(pairs)}건)")

        def _cb(frac: float, msg: str) -> None:
            progress(min(0.98, frac), desc=msg)

        result = run_dpo_training(
            pairs=pairs,
            model_cfg=ctx.cfg.get("model", {}),
            train_cfg=ctx.cfg.get("train", {}),
            state=ctx.state,
            eval_fn=make_eval_fn(ctx.config_path),
            progress_cb=_cb,
            resume_from_adapter=ctx.state.active_adapter,
        )
        if not result.ok:
            return f"❌ 학습 실패: {result.error}"

        record_training(ctx.state, result)
        # 학습 후 활성 어댑터가 바뀌었으므로 추론용 샘플러도 새 어댑터를 쓰게 한다
        ctx.sampler.set_adapter(ctx.state.active_adapter)

        eval_txt = (
            " · ".join(f"{k.upper()} {v:.4f}" for k, v in result.eval.items())
            if result.eval else "_(고정 평가셋이 없어 평가를 건너뜀)_"
        )
        return (
            f"✅ **{result.checkpoint}** 학습 완료 ({result.seconds}s)\n\n"
            f"- 페어 {result.n_pairs}건 / step {result.steps} "
            f"(스킵 {result.skipped})\n"
            f"- DPO loss **{result.loss}** · reward accuracy **{result.reward_accuracy}**\n"
            f"- 평가: {eval_txt}\n"
            f"- 어댑터가 활성화되어 이후 수집·추론에 적용됩니다."
        )

    except Exception as e:
        logger.exception("[UI] 학습 실패")
        return f"❌ 학습 중 오류: {e}"


def activate_adapter(ctx: AppContext, name: Optional[str]) -> str:
    """선택한 어댑터를 활성화한다 (base 로 되돌리기 포함)."""
    try:
        if not name or name == _BASE_CHOICE:
            ctx.state.set_active_adapter(None)
            ctx.sampler.set_adapter(None)
            return "✅ base 모델로 되돌렸습니다."
        target = next(
            (a for a in ctx.state.list_adapters() if a["name"] == name), None
        )
        if target is None:
            return f"❌ 어댑터를 찾을 수 없습니다: {name}"
        ctx.state.set_active_adapter(target["path"])
        ctx.sampler.set_adapter(target["path"])
        return f"✅ `{name}` 활성화 — 이후 수집·추론에 적용됩니다."
    except Exception as e:
        logger.exception("[UI] 어댑터 활성화 실패")
        return f"❌ 활성화 실패: {e}"


#: 어댑터 드롭다운에서 base 를 고르기 위한 값
_BASE_CHOICE = "base (어댑터 없음)"


# ─── 💬 추론 (플레이그라운드) 탭 ───────────────────────────────────────────

def playground_infer(
    ctx: AppContext,
    question: str,
    images: Optional[List[Any]],
    max_new_tokens: float,
    temperature: float,
    compare_base: bool,
    progress=gr.Progress(),
) -> str:
    """
    활성 모델로 답변을 생성한다.

    `compare_base` 를 켜면 같은 입력에 대해 base 모델 답변도 함께 보여준다 —
    학습이 실제로 무엇을 바꿨는지 눈으로 확인하는 용도.
    """
    question = (question or "").strip()
    if not question:
        return "⚠️ 질문을 입력하세요."

    image_paths = _normalize_images(images)
    try:
        from .backends import get_shared_backend

        adapter = ctx.state.active_adapter
        progress(0.2, desc=f"활성 모델({ctx.state.model_version}) 추론 중…")
        active_backend = get_shared_backend(
            ctx.cfg.get("model", {}), adapter_path=adapter, autoload=True
        )
        answer = active_backend.infer(
            image_paths, question,
            max_new_tokens=int(max_new_tokens), temperature=float(temperature),
        )
        out = f"### 🟢 활성 모델 (`{ctx.state.model_version}`)\n\n{answer or '_(빈 응답)_'}"

        if compare_base and adapter:
            progress(0.6, desc="base 모델 추론 중…")
            base_backend = get_shared_backend(
                ctx.cfg.get("model", {}), adapter_path=None, autoload=True
            )
            base_answer = base_backend.infer(
                image_paths, question,
                max_new_tokens=int(max_new_tokens), temperature=float(temperature),
            )
            out += (
                f"\n\n---\n\n### ⚪ base 모델 (학습 전)\n\n"
                f"{base_answer or '_(빈 응답)_'}"
            )
        elif compare_base:
            out += "\n\n---\n\n_활성 어댑터가 없어 비교할 대상이 없습니다 (지금이 base)._"
        return out

    except Exception as e:
        logger.exception("[UI] 플레이그라운드 추론 실패")
        return f"❌ 추론 실패: {e}"


# ─── ⚖️ 모델 비교 탭 ───────────────────────────────────────────────────────
#
# 지표(ANLS 등)는 정답이 도표 캡션이라 표현이 조금만 달라도 점수가 깎여 절대값을 믿기 어렵다.
# 같은 질문에 학습 전/후 답변을 **나란히 놓고 사람이 고르는** 것이 학습 효과를 보는 가장
# 직접적인 방법이고, 그 승패를 쌓으면 정성 판단을 정량화할 수 있다.
#
# 두 모델을 각각 로드하지 않는다. 어댑터를 얹은 모델 하나에서 LoRA 를 켜고 끄면
# base(학습 전)와 학습 후를 모두 얻을 수 있다 — 메모리가 절반이고, 같은 가중치에서
# 어댑터만 차이나므로 비교도 더 공정하다.

_BEST_CHOICE = "🏆 최고 성능 (자동 선택)"


def comparison_targets(ctx: AppContext, metric: str = "mean") -> Tuple[Any, str]:
    """비교에 쓸 어댑터 드롭다운 선택지 + 최고 성능 안내 문구."""
    adapters = ctx.state.list_adapters()
    choices = [_BEST_CHOICE] + [a["name"] for a in adapters]
    best = ctx.state.best_checkpoint(metric)

    if not adapters:
        note = (
            "⚠️ 학습된 어댑터가 없습니다. 🎯 학습 탭에서 먼저 학습하세요 "
            "(base 끼리 비교하면 결과가 같습니다)."
        )
    elif best is None:
        note = (
            f"어댑터 {len(adapters)}개가 있지만 **평가 지표가 기록된 체크포인트가 없어** "
            "최고 성능을 고를 수 없습니다. 고정 평가셋을 만든 뒤 학습하면 자동 선택됩니다. "
            "지금은 아래에서 직접 고르세요."
        )
    else:
        ev = " · ".join(f"{k.upper()} {v:.4f}" for k, v in best["eval"].items())
        warn = "" if best["exists"] else "  ⚠️ 파일이 없습니다"
        note = (
            f"🏆 최고 성능: **{best['checkpoint']}** "
            f"({metric} {best['score']:.4f}) — {ev}{warn}"
        )
    return gr.update(choices=choices, value=_BEST_CHOICE), note


def _resolve_compare_adapter(
    ctx: AppContext, selection: Optional[str], metric: str
) -> Tuple[Optional[str], str]:
    """드롭다운 선택 → (어댑터 경로, 표시 이름). 없으면 (None, 'base')."""
    if selection and selection != _BEST_CHOICE:
        for a in ctx.state.list_adapters():
            if a["name"] == selection:
                return a["path"], a["name"]
        return None, "base"
    best = ctx.state.best_checkpoint(metric)
    if best and best["exists"]:
        return best["path"], best["checkpoint"]
    # 최고 성능을 못 고르면 활성 어댑터라도 쓴다
    active = ctx.state.active_adapter
    return (active, Path(active).name) if active else (None, "base")


def load_eval_question(ctx: AppContext, index: float) -> Tuple[str, Any, str]:
    """고정 평가셋에서 문항을 불러온다 (정답이 있어 비교 판단이 쉬워진다)."""
    try:
        items = ctx.evalset.load(verify_frozen=False) if ctx.evalset else []
    except Exception as e:
        return "", gr.update(), f"❌ 평가셋을 읽지 못했습니다: {e}"
    if not items:
        return "", gr.update(), (
            "⚠️ 확정된 평가셋이 없습니다. "
            "`python -m dpo_collector.eval_ko draft` → 검토 → `confirm` 을 먼저 하세요."
        )
    i = int(index) % len(items)
    it = items[i]
    gold = " / ".join(it.ground_truths[:3])
    return (
        it.question,
        gr.update(value=list(it.image_paths)),
        f"📋 평가셋 {i + 1}/{len(items)} · `{it.paper_id[:40]}`\n\n"
        f"**참고 정답**: {gold}",
    )


def run_comparison(
    ctx: AppContext,
    question: str,
    images: Optional[List[Any]],
    adapter_choice: Optional[str],
    metric: str,
    max_new_tokens: float,
    temperature: float,
    progress=gr.Progress(),
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    같은 입력으로 base(학습 전)와 학습 모델의 답변을 각각 생성한다.

    Returns: (base 답변, 학습 모델 답변, 헤더 안내, 비교 세션)
    """
    import time

    empty: Dict[str, Any] = {}
    question = (question or "").strip()
    if not question:
        return "", "", "⚠️ 질문을 입력하거나 평가셋에서 불러오세요.", empty

    image_paths = _normalize_images(images)
    adapter_path, label = _resolve_compare_adapter(ctx, adapter_choice, metric)
    if adapter_path is None:
        return "", "", (
            "⚠️ 비교할 학습 모델이 없습니다. 먼저 🎯 학습 탭에서 학습하세요."
        ), empty

    try:
        from .backends import get_shared_backend

        progress(0.1, desc=f"모델 로드 중… ({label})")
        backend = get_shared_backend(
            ctx.cfg.get("model", {}), adapter_path=adapter_path, autoload=True
        )

        def _infer(adapter_on: bool) -> Tuple[str, float]:
            backend.set_adapter_enabled(adapter_on)
            t0 = time.time()
            out = backend.infer(
                image_paths, question,
                max_new_tokens=int(max_new_tokens), temperature=float(temperature),
            )
            return (out or "").strip(), round(time.time() - t0, 1)

        progress(0.35, desc="학습 전(base) 추론 중…")
        base_ans, base_sec = _infer(False)
        progress(0.7, desc=f"학습 후({label}) 추론 중…")
        tuned_ans, tuned_sec = _infer(True)
        backend.set_adapter_enabled(True)   # 다른 탭에 영향 없도록 원복

        same = " ".join(base_ans.split()) == " ".join(tuned_ans.split())
        header = (
            f"**질문**: {question}\n\n"
            f"학습 전 `base` ({base_sec}s)  ·  학습 후 `{label}` ({tuned_sec}s)"
            + ("\n\n> ℹ️ 두 답변이 완전히 같습니다 — 학습량이 적거나 이 질문에서는 "
               "차이가 나지 않는 경우입니다. temperature 를 올리거나 다른 질문을 시도해 보세요."
               if same else "")
        )
        session = {
            "question": question,
            "images": image_paths,
            "checkpoint": label,
            "base_answer": base_ans,
            "tuned_answer": tuned_ans,
            "temperature": float(temperature),
            "identical": same,
        }
        return (
            base_ans or "_(빈 응답)_",
            tuned_ans or "_(빈 응답)_",
            header,
            session,
        )

    except Exception as e:
        logger.exception("[UI] 모델 비교 실패")
        return "", "", f"❌ 비교 실패: {e}", empty


def save_comparison(
    ctx: AppContext,
    session: Dict[str, Any],
    verdict: Optional[str],
    note: str,
) -> Tuple[str, str]:
    """정성 판정을 기록하고 누적 승률을 돌려준다."""
    if not session or not session.get("question"):
        return "⚠️ 먼저 비교를 실행하세요.", comparison_summary(ctx)
    if not verdict:
        return "⚠️ 어느 쪽이 나은지 선택하세요.", comparison_summary(ctx)
    try:
        rec = ctx.state.add_comparison({
            **session,
            "verdict": verdict,
            "note": note or "",
            "annotator": "user",
        })
        return (
            f"✅ 기록됨 — `{rec['checkpoint']}` / 판정 **{verdict}**",
            comparison_summary(ctx),
        )
    except Exception as e:
        logger.exception("[UI] 비교 기록 실패")
        return f"❌ 기록 실패: {e}", comparison_summary(ctx)


def comparison_summary(ctx: AppContext) -> str:
    """누적 정성 비교 승률 (체크포인트별)."""
    try:
        rows = ctx.state.comparisons()
    except Exception as e:
        return f"_집계 실패: {e}_"
    if not rows:
        return "_아직 기록된 비교가 없습니다._"

    ckpts: List[str] = []
    for r in rows:
        c = r.get("checkpoint", "?")
        if c not in ckpts:
            ckpts.append(c)

    lines = [
        "| 체크포인트 | 비교 | 학습 승 | base 승 | 비김 | 승률(무승부 제외) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in ckpts:
        s = ctx.state.comparison_stats(c)
        wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "—"
        lines.append(
            f"| `{c}` | {s['n']} | {s['trained']} | {s['base']} | {s['tie']} | **{wr}** |"
        )
    total = ctx.state.comparison_stats()
    twr = f"{total['win_rate']:.0%}" if total["win_rate"] is not None else "—"
    lines.append(f"\n전체 {total['n']}건 · 학습 모델 승률 **{twr}**")
    return "\n".join(lines)


# ─── 📦 내보내기 탭 ────────────────────────────────────────────────────────

def run_export(
    ctx: AppContext,
    fmt: str,
    field_map_json: str,
    copy_images: bool,
    progress=gr.Progress(),
) -> Tuple[str, Any]:
    """공유 포맷으로 export 하고 다운로드 링크를 돌려준다."""
    import json

    try:
        fmap: Optional[Dict[str, str]] = None
        text = (field_map_json or "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("객체(JSON dict) 형태여야 합니다.")
                fmap = {str(k): str(v) for k, v in parsed.items()}
            except Exception as e:
                return f"❌ 필드 매핑 JSON 오류: {e}", gr.update(value=None)

        progress(0.3, desc="export 생성 중…")
        info = ctx.store.export(
            fmt=fmt, field_map=fmap, copy_images=bool(copy_images)
        )
        if info["n_pairs"] == 0:
            return (
                "⚠️ 내보낼 페어가 없습니다. 먼저 수집 탭에서 페어를 모으세요.",
                gr.update(value=None),
            )

        msg = (
            f"✅ **{info['n_pairs']}건** 내보냄 (포맷 `{info['format']}`)\n\n"
            f"- 파일: `{info['path']}`\n"
            f"- 이미지: {info['n_images']}장"
            + (f" → `{info['image_dir']}`" if info["image_dir"] else " (경로 그대로 기록)")
            + "\n\n"
            + (
                "> ℹ️ `rlaif_v` 는 `image` 가 단일 필드라 **이미지가 여러 장인 페어는 첫 장만** "
                "나갑니다. 전량 보존하려면 `hf_conversational` 을 쓰세요.\n"
                if info["format"] == "rlaif_v" else ""
            )
            + "> 이미지를 함께 전달하려면 `dpo_share.jsonl` 과 `images/` 폴더를 같이 압축해 주세요."
        )
        return msg, gr.update(value=info["path"])

    except Exception as e:
        logger.exception("[UI] export 실패")
        return f"❌ export 실패: {e}", gr.update(value=None)


# ─── ⚙️ 설정 탭 ────────────────────────────────────────────────────────────

def settings_snapshot(ctx: AppContext) -> Tuple[Any, ...]:
    """현재 설정을 폼 필드 값으로 펼친다."""
    c = ctx.cfg
    paths = c.get("paths", {}) or {}
    model = c.get("model", {}) or {}
    qg = c.get("question_gen", {}) or {}
    ag = c.get("answer_gen", {}) or {}
    ev = c.get("eval", {}) or {}
    tr = c.get("train", {}) or {}
    ex = c.get("export", {}) or {}
    kg = c.get("kg", {}) or {}

    import json

    return (
        c.get("language", "ko"),
        paths.get("data_ko_root", ""),
        paths.get("pairs_jsonl", ""),
        paths.get("kg_json", ""),
        paths.get("export_dir", ""),
        paths.get("adapter_dir", ""),
        model.get("name", ""),
        str(model.get("backend", "auto")),
        str(model.get("dtype", "bfloat16")),
        int(model.get("max_new_tokens", 512) or 512),
        float(model.get("temperature", 0.7) or 0.7),
        int(model.get("max_num_tiles", 12) or 12),
        qg.get("name", ""),
        str(qg.get("backend", "auto")),
        int(qg.get("n_candidates", 3) or 3),
        qg.get("prompt_template") or "",
        int(ag.get("n_candidates", 3) or 3),
        float(ag.get("temperature", 0.9) or 0.9),
        ag.get("prompt_template") or "",
        int(kg.get("max_papers", 0) or 0),
        str(kg.get("language_filter") or ""),
        list(ev.get("metrics") or ["anls", "accuracy", "f1"]),
        int(ev.get("max_new_tokens", 48) or 48),
        int(tr.get("first_train_min_pairs", 150) or 150),
        int(tr.get("retrain_every_n_pairs", 100) or 100),
        int(tr.get("min_pairs_per_type", 10) or 10),
        float(tr.get("dpo_beta", 0.1) or 0.1),
        float(tr.get("learning_rate", 5e-6) or 5e-6),
        int(tr.get("epochs", 2) or 2),
        int(tr.get("max_num_tiles", 6) or 6),
        str(ex.get("format", "rlaif_v")),
        json.dumps(ex.get("field_map") or {}, ensure_ascii=False, indent=2),
    )


def save_settings_form(ctx: AppContext, *values: Any) -> Tuple[str, str]:
    """폼 값을 config 에 반영하고 앱을 리로드한다."""
    import json

    from . import config_io

    keys: List[Tuple[str, ...]] = [
        ("language",),
        ("paths", "data_ko_root"), ("paths", "pairs_jsonl"), ("paths", "kg_json"),
        ("paths", "export_dir"), ("paths", "adapter_dir"),
        ("model", "name"), ("model", "backend"), ("model", "dtype"),
        ("model", "max_new_tokens"), ("model", "temperature"), ("model", "max_num_tiles"),
        ("question_gen", "name"), ("question_gen", "backend"),
        ("question_gen", "n_candidates"), ("question_gen", "prompt_template"),
        ("answer_gen", "n_candidates"), ("answer_gen", "temperature"),
        ("answer_gen", "prompt_template"),
        ("kg", "max_papers"), ("kg", "language_filter"),
        ("eval", "metrics"), ("eval", "max_new_tokens"),
        ("train", "first_train_min_pairs"), ("train", "retrain_every_n_pairs"),
        ("train", "min_pairs_per_type"), ("train", "dpo_beta"),
        ("train", "learning_rate"), ("train", "epochs"), ("train", "max_num_tiles"),
        ("export", "format"), ("export", "field_map"),
    ]
    if len(values) != len(keys):
        return (
            f"❌ 폼 필드 수가 맞지 않습니다 ({len(values)} vs {len(keys)}).",
            settings_yaml(ctx),
        )

    updates: Dict[Tuple[str, ...], Any] = {}
    for key, val in zip(keys, values):
        if key == ("export", "field_map"):
            text = (val or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("객체(dict) 형태여야 합니다.")
                updates[key] = {str(k): str(v) for k, v in parsed.items()}
            except Exception as e:
                return f"❌ `export.field_map` JSON 오류: {e}", settings_yaml(ctx)
            continue
        # 빈 프롬프트/언어필터는 "기본값 사용" 의미의 null 로 저장한다
        if key in (
            ("question_gen", "prompt_template"),
            ("answer_gen", "prompt_template"),
            ("kg", "language_filter"),
        ) and not str(val or "").strip():
            updates[key] = None
            continue
        updates[key] = val

    # None 은 apply_updates 에서 "건드리지 않음" 이므로, 명시적 null 은 따로 넣는다
    explicit_null = {k: v for k, v in updates.items() if v is None}
    updates = {k: v for k, v in updates.items() if v is not None}

    ok, msg = config_io.apply_updates(ctx.config_path, updates)
    if ok and explicit_null:
        config_io.apply_updates(
            ctx.config_path, {k: "" for k in explicit_null}
        )
    if not ok:
        return msg, settings_yaml(ctx)

    try:
        notes = ctx.reload()
        if notes:
            msg += f"\n\n**리로드 결과**\n{notes}"
        else:
            msg += "\n\n리로드 완료 — 변경된 설정이 즉시 적용됩니다."
    except Exception as e:
        logger.exception("[UI] 설정 리로드 실패")
        msg += f"\n\n⚠️ 저장은 됐지만 리로드에 실패했습니다: {e}"
    return msg, settings_yaml(ctx)


def settings_yaml(ctx: AppContext) -> str:
    """현재 config 파일 원문."""
    from . import config_io

    try:
        return config_io.read_text(ctx.config_path)
    except Exception as e:
        return f"# 설정 파일을 읽지 못했습니다: {e}"


def save_settings_yaml(ctx: AppContext, text: str) -> Tuple[str, Any]:
    """YAML 원문을 그대로 저장하고 리로드한다 (주석·서식 100% 보존)."""
    from . import config_io

    ok, msg = config_io.save_text(ctx.config_path, text)
    if not ok:
        return msg, gr.update()
    try:
        notes = ctx.reload()
        msg += f"\n\n**리로드 결과**\n{notes}" if notes else "\n\n리로드 완료."
    except Exception as e:
        logger.exception("[UI] 설정 리로드 실패")
        msg += f"\n\n⚠️ 저장은 됐지만 리로드에 실패했습니다: {e}"
    return msg, gr.update(choices=config_io.list_backups(ctx.config_path))


def restore_settings_backup(ctx: AppContext, name: Optional[str]) -> Tuple[str, str]:
    """백업으로 되돌린다."""
    from . import config_io

    if not name:
        return "⚠️ 되돌릴 백업을 선택하세요.", settings_yaml(ctx)
    ok, msg = config_io.restore_backup(ctx.config_path, name)
    if ok:
        try:
            ctx.reload()
            msg += "\n\n리로드 완료."
        except Exception as e:
            msg += f"\n\n⚠️ 리로드 실패: {e}"
    return msg, settings_yaml(ctx)


def export_preview(ctx: AppContext) -> str:
    """현재 저장소 상태 요약 (내보내기 전 확인용)."""
    import json

    s = ctx.store.stats()
    return (
        f"수집된 페어 **{s['total_pairs']}건** "
        f"(이미지 보유 {s['with_image']}건, 삭제 {s['deleted_pairs']}건)\n\n"
        f"- 유형별: `{json.dumps(s['by_question_type'], ensure_ascii=False)}`\n"
        f"- 생성 모델별: `{json.dumps(s['by_model_version'], ensure_ascii=False)}`\n"
        f"- 원본: `{s['path']}`"
    )


# ─── UI 구성 ───────────────────────────────────────────────────────────────

def build_ui(ctx: AppContext) -> gr.Blocks:
    kg_cfg = ctx.cfg.get("kg", {})
    sampling = kg_cfg.get("sampling", {})
    qg_cfg = ctx.cfg.get("question_gen", {})
    ans_cfg = ctx.cfg.get("answer_gen", {})

    with gr.Blocks(title="한국어 멀티모달 DPO 수집기", fill_height=True) as demo:
        gr.Markdown(
            "# 🇰🇷 한국어 멀티모달 DPO 수집기\n"
            "지식 그래프에서 질문을 만들고, 현재 모델의 후보 답변 중 더 나은 것을 골라 "
            "선호(DPO) 데이터를 모읍니다."
        )

        # 세션별 상태 — 사용자마다 독립적으로 유지된다
        q_session = gr.State({"candidates": []})
        a_session = gr.State({"candidates": []})

        with gr.Tabs():
            # ── 🕸️ KG 구축 ────────────────────────────────────────────────
            with gr.Tab("🕸️ KG 구축"):
                gr.Markdown(
                    f"`{ctx.bridge.data_root}` 의 파싱 결과로 지식 그래프를 만듭니다. "
                    "한 번 만들면 JSON으로 저장되어 다음 실행부터는 즉시 로드됩니다."
                )
                with gr.Row():
                    kg_max_papers = gr.Number(
                        value=int(kg_cfg.get("max_papers", 50)),
                        label="논문 수 (0 = 전체)", precision=0, scale=1,
                    )
                    kg_rebuild = gr.Checkbox(
                        value=False,
                        label="강제 재구축 (저장된 KG 무시)", scale=1,
                    )
                    kg_btn = gr.Button("KG 구축 / 로드", variant="primary", scale=1)
                kg_stats = gr.Markdown("아직 KG가 준비되지 않았습니다.")

            # ── ✍️ 수집 ───────────────────────────────────────────────────
            with gr.Tab("✍️ 수집"):
                with gr.Row():
                    # 좌: 질문 만들기
                    with gr.Column(scale=5):
                        gr.Markdown("### 1) 질문")
                        mode = gr.Radio(
                            choices=[
                                ("모드 B — KG 기반 자동 질문", "auto"),
                                ("모드 A — 직접 질문 입력", "user"),
                            ],
                            value="auto",
                            label="수집 모드",
                        )
                        with gr.Group(visible=True) as auto_group:
                            paper_dd = gr.Dropdown(
                                choices=[], label="논문 (비우면 전체에서 무작위)",
                                interactive=True,
                            )
                            with gr.Row():
                                n_q = gr.Slider(
                                    1, 5, value=int(qg_cfg.get("n_candidates", 3)),
                                    step=1, label="후보 질문 수",
                                )
                                max_hops = gr.Slider(
                                    1, 3, value=int(sampling.get("max_hops", 2)),
                                    step=1, label="KG 경로 홉",
                                )
                            require_img = gr.Checkbox(
                                value=bool(sampling.get("require_image", True)),
                                label="이미지가 있는 경로만 (멀티모달 수집)",
                            )
                            gen_q_btn = gr.Button("🎲 질문 생성", variant="primary")
                            # choices 는 상한까지 고정 — 이유는 MAX_QUESTION_CANDIDATES 주석 참고
                            q_radio = gr.Radio(
                                choices=_fixed_choices(MAX_QUESTION_CANDIDATES, "후보 질문"),
                                value=None,
                                label="후보 질문 — 하나를 고르면 아래 입력창에 채워집니다",
                            )

                        question_box = gr.Textbox(
                            label="질문 (직접 수정 가능)", lines=3,
                            placeholder="모드 A는 여기에 질문을 직접 입력하세요.",
                        )
                        images_gallery = gr.Gallery(
                            label="질문에 사용할 이미지", columns=4, height=200,
                            allow_preview=True,
                        )
                        upload = gr.File(
                            label="이미지 직접 추가 (모드 A)",
                            file_count="multiple", file_types=["image"],
                        )
                        evidence_md = gr.Markdown("")

                    # 우: 답변 고르기
                    with gr.Column(scale=5):
                        gr.Markdown("### 2) 후보 답변 · 선호 선택")
                        with gr.Row():
                            n_a = gr.Slider(
                                2, 6, value=int(ans_cfg.get("n_candidates", 3)),
                                step=1, label="후보 답변 수",
                            )
                            temp = gr.Slider(
                                0.1, 1.5, value=float(ans_cfg.get("temperature", 0.9)),
                                step=0.1, label="temperature (다양성)",
                            )
                        gen_a_btn = gr.Button("💬 답변 생성", variant="primary")
                        with gr.Row():
                            chosen_radio = gr.Radio(
                                choices=_fixed_choices(MAX_ANSWER_CANDIDATES, "후보"),
                                value=None, label="👍 chosen (더 나은 답변)",
                            )
                            rejected_radio = gr.Radio(
                                choices=_fixed_choices(MAX_ANSWER_CANDIDATES, "후보"),
                                value=None, label="👎 rejected (더 나쁜 답변)",
                            )
                        answers_md = gr.Markdown("")
                        with gr.Row():
                            annotator = gr.Textbox(
                                value="user", label="작성자", scale=1,
                            )
                            notes = gr.Textbox(
                                label="메모 (선택)", scale=2,
                                placeholder="예: 후보 2는 중국어가 섞임",
                            )
                        with gr.Row():
                            save_btn = gr.Button("💾 페어 저장", variant="primary", scale=2)
                            skip_btn = gr.Button("↩️ 건너뛰고 다음", scale=1)
                        with gr.Row():
                            auto_next = gr.Checkbox(
                                value=True, scale=2,
                                label="저장하면 자동으로 다음 항목 (같은 논문)",
                            )
                            next_paper_btn = gr.Button("📄 다음 논문 ▶", scale=1)
                            prev_paper_btn = gr.Button("◀ 이전 논문", scale=1)
                        save_msg = gr.Markdown("")

                gr.Markdown("---")
                progress_md = gr.Markdown(ctx.progress_markdown())
                refresh_btn = gr.Button("🔄 진행 상황 새로고침", size="sm")

            # ── 🎯 학습 ───────────────────────────────────────────────────
            with gr.Tab("🎯 학습"):
                gr.Markdown(
                    "수집된 선호 페어로 **DPO LoRA** 를 학습합니다. "
                    "policy 와 reference 는 같은 모델에서 어댑터를 켜고 끄는 방식이라 "
                    "모델을 두 번 올리지 않습니다."
                )
                train_status = gr.Markdown("_상태를 불러오는 중…_")
                with gr.Row():
                    force_train = gr.Checkbox(
                        value=False, label="강제 실행 (임계치 무시)", scale=1
                    )
                    train_limit = gr.Number(
                        value=0, precision=0, scale=1,
                        label="학습 페어 수 상한 (0 = 전체)",
                    )
                    train_btn = gr.Button("🚀 지금 학습", variant="primary", scale=1)
                    train_refresh = gr.Button("🔄 새로고침", scale=1)
                train_msg = gr.Markdown("")

                gr.Markdown("### 체크포인트별 평가 지표")
                # 지표와 loss 는 스케일이 다르므로 **축을 공유하지 않는다** — 차트를 나눈다.
                metric_plot = gr.LinePlot(
                    x="checkpoint", y="value", color="metric",
                    color_map=_SERIES_COLORS,
                    y_lim=[0, 1], height=260,
                    x_title="체크포인트", y_title="점수",
                    title="ANLS / Accuracy / F1 (고정 평가셋)",
                )
                loss_plot = gr.LinePlot(
                    x="checkpoint", y="loss",
                    color_map={"loss": _LOSS_COLOR},
                    height=220,
                    x_title="체크포인트", y_title="DPO loss",
                    title="DPO loss (낮을수록 좋음, 시작값 ≈ 0.693 = log 2)",
                )
                # 색상만으로 값을 읽지 않아도 되도록 표를 항상 함께 둔다
                gr.Markdown("#### 학습 이력 (표)")
                history_table = gr.Dataframe(
                    interactive=False, wrap=True, label=None,
                )

                gr.Markdown("### 어댑터")
                adapter_table = gr.Dataframe(interactive=False, wrap=True, label=None)
                with gr.Row():
                    adapter_dd = gr.Dropdown(
                        choices=[], label="활성화할 어댑터", scale=3,
                        allow_custom_value=False,
                    )
                    to_base_btn = gr.Button("base 로 되돌리기", scale=1)
                    activate_btn = gr.Button("이 어댑터 활성화", variant="primary", scale=1)
                adapter_msg = gr.Markdown("")

            # ── 💬 추론 (플레이그라운드) ──────────────────────────────────
            with gr.Tab("💬 추론"):
                gr.Markdown(
                    "현재 **활성 모델**(base 또는 선택한 어댑터)로 자유롭게 질문해 봅니다."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pg_images = gr.Gallery(
                            label="이미지 (선택)", columns=3, height=200,
                        )
                        pg_upload = gr.File(
                            label="이미지 추가", file_count="multiple",
                            file_types=["image"],
                        )
                        pg_question = gr.Textbox(
                            label="질문", lines=3,
                            placeholder="예: 이 그림이 무엇을 보여주는지 설명해 주세요.",
                        )
                        with gr.Row():
                            pg_max_tokens = gr.Slider(
                                32, 1024,
                                value=int(ctx.cfg.get("model", {}).get("max_new_tokens", 512)),
                                step=32, label="max_new_tokens",
                            )
                            pg_temp = gr.Slider(
                                0.0, 1.5,
                                value=float(ctx.cfg.get("model", {}).get("temperature", 0.7)),
                                step=0.1, label="temperature (0 = greedy)",
                            )
                        pg_compare = gr.Checkbox(
                            value=False,
                            label="base 모델과 비교 (학습 효과 확인)",
                        )
                        pg_btn = gr.Button("💬 답변 생성", variant="primary")
                    with gr.Column(scale=1):
                        pg_output = gr.Markdown("")

            # ── ⚖️ 모델 비교 ──────────────────────────────────────────────
            with gr.Tab("⚖️ 모델 비교"):
                gr.Markdown(
                    "같은 질문에 **학습 전(base)** 과 **학습 후** 답변을 나란히 놓고 비교합니다.\n"
                    "지표만으로는 드러나지 않는 문체·정확도·언어 품질 차이를 눈으로 확인하고, "
                    "판정을 쌓으면 승률로 정량화됩니다."
                )
                _cmp_dd, _cmp_note = comparison_targets(ctx)
                with gr.Row():
                    cmp_adapter = gr.Dropdown(
                        choices=_cmp_dd.get("choices") or [_BEST_CHOICE],
                        value=_BEST_CHOICE, scale=3,
                        label="비교할 학습 모델",
                    )
                    cmp_metric = gr.Dropdown(
                        choices=[
                            ("세 지표 평균", "mean"), ("ANLS", "anls"),
                            ("Accuracy", "accuracy"), ("F1", "f1"),
                        ],
                        value="mean", scale=2,
                        label="최고 성능 판단 기준",
                    )
                    cmp_refresh = gr.Button("🔄 목록 새로고침", scale=1)
                cmp_best_note = gr.Markdown(_cmp_note)

                with gr.Row():
                    with gr.Column(scale=3):
                        cmp_question = gr.Textbox(
                            label="질문", lines=3,
                            placeholder="예: 이 그림이 무엇을 보여주는지 설명해 주세요.",
                        )
                        cmp_images = gr.Gallery(
                            label="이미지", columns=4, height=170, allow_preview=True,
                        )
                        cmp_upload = gr.File(
                            label="이미지 추가", file_count="multiple",
                            file_types=["image"],
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("**평가셋에서 문항 불러오기** — 정답이 있어 판단이 쉽습니다")
                        with gr.Row():
                            cmp_eval_idx = gr.Number(
                                value=0, precision=0, label="문항 번호", scale=1,
                            )
                            cmp_load_eval = gr.Button("📋 불러오기", scale=1)
                        cmp_eval_note = gr.Markdown("")
                        with gr.Row():
                            cmp_tokens = gr.Slider(
                                32, 1024, value=256, step=32, label="max_new_tokens",
                            )
                            cmp_temp = gr.Slider(
                                0.0, 1.5, value=0.0, step=0.1,
                                label="temperature (0 = greedy, 비교엔 0 권장)",
                            )
                        cmp_run = gr.Button("⚖️ 두 모델로 답변 생성", variant="primary")

                cmp_header = gr.Markdown("")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ⚪ 학습 전 (base)")
                        cmp_base_out = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("### 🟢 학습 후")
                        cmp_tuned_out = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("#### 어느 쪽이 더 나은가요?")
                with gr.Row():
                    cmp_verdict = gr.Radio(
                        choices=[
                            ("🟢 학습 후가 낫다", "trained"),
                            ("⚪ 학습 전이 낫다", "base"),
                            ("= 비슷하다", "tie"),
                        ],
                        value=None, label=None, scale=3,
                    )
                    cmp_note_box = gr.Textbox(
                        label="메모 (선택)", scale=3,
                        placeholder="예: 학습 후가 중국어 혼입이 줄었음",
                    )
                    cmp_save = gr.Button("💾 판정 기록", variant="primary", scale=1)
                cmp_save_msg = gr.Markdown("")
                gr.Markdown("#### 누적 결과")
                cmp_stats = gr.Markdown(comparison_summary(ctx))

            # ── 📦 내보내기 ───────────────────────────────────────────────
            with gr.Tab("📦 내보내기"):
                gr.Markdown(
                    "수집한 페어를 **공유용 표준 포맷**으로 내보냅니다. "
                    "삭제된 페어는 자동으로 제외됩니다."
                )
                export_summary = gr.Markdown(export_preview(ctx))
                with gr.Row():
                    export_fmt = gr.Radio(
                        choices=[
                            ("RLAIF-V (image/question/chosen/rejected)", "rlaif_v"),
                            ("HF conversational (다중 이미지 보존)", "hf_conversational"),
                        ],
                        value=str(
                            ctx.cfg.get("export", {}).get("format", "rlaif_v")
                        ),
                        label="포맷",
                    )
                    export_copy = gr.Checkbox(
                        value=bool(ctx.cfg.get("export", {}).get("copy_images", True)),
                        label="이미지를 export 폴더로 복사 (상대경로로 기록)",
                    )
                export_map = gr.Textbox(
                    label="필드 이름 매핑 (JSON, 비우면 기본 스키마)",
                    lines=3,
                    placeholder='{"image": "img", "question": "prompt", '
                                '"chosen": "preferred", "rejected": "dispreferred"}',
                )
                with gr.Row():
                    export_btn = gr.Button("📦 내보내기", variant="primary", scale=2)
                    export_refresh = gr.Button("🔄 현황 새로고침", scale=1)
                export_msg = gr.Markdown("")
                export_file = gr.File(label="다운로드", interactive=False)

            # ── ⚙️ 설정 ───────────────────────────────────────────────────
            with gr.Tab("⚙️ 설정"):
                gr.Markdown(
                    f"`{ctx.config_path}` 를 편집합니다. 저장하면 **백업을 남기고** "
                    "곧바로 리로드되어 다른 탭에 적용됩니다."
                )
                _s = settings_snapshot(ctx)
                with gr.Tabs():
                    # ── 폼 편집 ──
                    with gr.Tab("폼"):
                        with gr.Accordion("언어 · 경로", open=True):
                            set_language = gr.Textbox(value=_s[0], label="language (수집 언어)")
                            set_data_root = gr.Textbox(value=_s[1], label="paths.data_ko_root")
                            set_pairs = gr.Textbox(value=_s[2], label="paths.pairs_jsonl")
                            set_kgjson = gr.Textbox(value=_s[3], label="paths.kg_json")
                            set_exportdir = gr.Textbox(value=_s[4], label="paths.export_dir")
                            set_adapterdir = gr.Textbox(value=_s[5], label="paths.adapter_dir")

                        with gr.Accordion("답변 생성 모델 (Solver = 학습 대상)", open=True):
                            set_model_name = gr.Textbox(
                                value=_s[6], label="model.name (HF 모델명 자유 입력)",
                            )
                            with gr.Row():
                                set_model_backend = gr.Dropdown(
                                    choices=list(config_io.BACKEND_CHOICES),
                                    value=_s[7], label="model.backend",
                                )
                                set_model_dtype = gr.Dropdown(
                                    choices=list(config_io.DTYPE_CHOICES),
                                    value=_s[8], label="model.dtype",
                                )
                            with gr.Row():
                                set_model_tokens = gr.Number(
                                    value=_s[9], precision=0, label="max_new_tokens",
                                )
                                set_model_temp = gr.Number(value=_s[10], label="temperature")
                                set_model_tiles = gr.Number(
                                    value=_s[11], precision=0, label="max_num_tiles (추론)",
                                )
                            gr.Markdown(
                                "_InternVL 커스텀 계열(`InternVL2_5-*`)은 bfloat16 고정입니다 — "
                                "4bit/fp16 은 이미지 이해가 깨집니다._"
                            )

                        with gr.Accordion("질문 생성 모델 (Challenger) · 프롬프트", open=False):
                            set_qg_name = gr.Textbox(value=_s[12], label="question_gen.name")
                            with gr.Row():
                                set_qg_backend = gr.Dropdown(
                                    choices=list(config_io.BACKEND_CHOICES),
                                    value=_s[13], label="question_gen.backend",
                                )
                                set_qg_n = gr.Number(
                                    value=_s[14], precision=0, label="후보 질문 수",
                                )
                            set_qg_prompt = gr.Textbox(
                                value=_s[15], lines=6,
                                label="question_gen.prompt_template "
                                      "(비우면 엣지 타입별 한국어 기본 템플릿)",
                                placeholder="{language} {src_type} {dst_type} {src_content} "
                                            "{dst_content} {edge_type} {image_hint} …",
                            )
                            with gr.Row():
                                set_ag_n = gr.Number(
                                    value=_s[16], precision=0, label="후보 답변 수",
                                )
                                set_ag_temp = gr.Number(
                                    value=_s[17], label="answer_gen.temperature",
                                )
                            set_ag_prompt = gr.Textbox(
                                value=_s[18], lines=4,
                                label="answer_gen.prompt_template (비우면 기본)",
                                placeholder="{language} {question}",
                            )

                        with gr.Accordion("KG · 평가 지표", open=False):
                            with gr.Row():
                                set_kg_max = gr.Number(
                                    value=_s[19], precision=0,
                                    label="kg.max_papers (0 = 전체)",
                                )
                                set_kg_lang = gr.Textbox(
                                    value=_s[20],
                                    label="kg.language_filter (비우면 필터 없음)",
                                )
                            set_metrics = gr.CheckboxGroup(
                                choices=list(config_io.METRIC_CHOICES),
                                value=_s[21], label="eval.metrics",
                            )
                            set_eval_tokens = gr.Number(
                                value=_s[22], precision=0, label="eval.max_new_tokens",
                            )

                        with gr.Accordion("학습 트리거 · DPO 하이퍼파라미터", open=False):
                            with gr.Row():
                                set_first = gr.Number(
                                    value=_s[23], precision=0, label="첫 학습 임계치",
                                )
                                set_retrain = gr.Number(
                                    value=_s[24], precision=0, label="재학습 주기",
                                )
                                set_per_type = gr.Number(
                                    value=_s[25], precision=0, label="유형별 최소 개수",
                                )
                            with gr.Row():
                                set_beta = gr.Number(value=_s[26], label="dpo_beta")
                                set_lr = gr.Number(value=_s[27], label="learning_rate")
                                set_epochs = gr.Number(
                                    value=_s[28], precision=0, label="epochs",
                                )
                                set_train_tiles = gr.Number(
                                    value=_s[29], precision=0, label="학습 타일 수",
                                )

                        with gr.Accordion("내보내기", open=False):
                            set_export_fmt = gr.Dropdown(
                                choices=list(config_io.EXPORT_FORMAT_CHOICES),
                                value=_s[30], label="export.format",
                            )
                            set_field_map = gr.Textbox(
                                value=_s[31], lines=6,
                                label="export.field_map (JSON, 타 기관 스키마 대응)",
                            )

                        with gr.Row():
                            settings_save_btn = gr.Button(
                                "💾 저장 & 리로드", variant="primary", scale=2,
                            )
                            settings_reset_btn = gr.Button("↩️ 파일에서 다시 읽기", scale=1)
                        settings_msg = gr.Markdown("")

                    # ── YAML 직접 편집 ──
                    with gr.Tab("YAML 직접 편집"):
                        gr.Markdown(
                            "폼에 없는 항목까지 모두 편집할 수 있습니다. "
                            "주석과 서식이 그대로 저장됩니다."
                        )
                        yaml_box = gr.Code(
                            value=settings_yaml(ctx), language="yaml", lines=28,
                            label=None,
                        )
                        with gr.Row():
                            yaml_save_btn = gr.Button(
                                "💾 저장 & 리로드", variant="primary", scale=2,
                            )
                            yaml_reload_btn = gr.Button("↩️ 파일에서 다시 읽기", scale=1)
                        yaml_msg = gr.Markdown("")

                    # ── 백업 ──
                    with gr.Tab("백업"):
                        gr.Markdown(
                            "저장할 때마다 타임스탬프 백업이 남습니다. 되돌리면 "
                            "되돌리기 직전 상태도 함께 백업됩니다."
                        )
                        backup_dd = gr.Dropdown(
                            choices=config_io.list_backups(ctx.config_path),
                            label="백업 파일 (최신순)",
                        )
                        with gr.Row():
                            backup_refresh_btn = gr.Button("🔄 목록 새로고침", scale=1)
                            backup_restore_btn = gr.Button(
                                "⏪ 이 백업으로 되돌리기", variant="primary", scale=2,
                            )
                        backup_msg = gr.Markdown("")

        # ── 이벤트 배선 ───────────────────────────────────────────────────

        kg_btn.click(
            lambda mp, rb, progress=gr.Progress(): kg_build_or_load(ctx, mp, rb, progress),
            inputs=[kg_max_papers, kg_rebuild],
            outputs=[kg_stats, paper_dd],
        )

        # 모드 전환 — 모드 A 는 자동 질문 관련 위젯을 숨긴다
        mode.change(
            lambda m: gr.update(visible=(m == "auto")),
            inputs=[mode], outputs=[auto_group],
        )

        paper_dd.change(
            lambda pid: load_paper_images(ctx, pid),
            inputs=[paper_dd], outputs=[images_gallery],
        )

        gen_q_btn.click(
            lambda pid, n, hops, ri, progress=gr.Progress():
                generate_questions(ctx, pid, n, hops, ri, progress),
            inputs=[paper_dd, n_q, max_hops, require_img],
            outputs=[q_radio, evidence_md, images_gallery, q_session],
        )

        q_radio.change(
            pick_question, inputs=[q_session, q_radio], outputs=[question_box],
        )

        # 업로드한 파일을 갤러리에 합친다 (모드 A)
        upload.change(
            lambda files, cur: gr.update(
                value=_normalize_images(cur) + _normalize_images(files)
            ),
            inputs=[upload, images_gallery], outputs=[images_gallery],
        )

        gen_a_btn.click(
            lambda q, imgs, qs, n, t, progress=gr.Progress():
                generate_answers(ctx, q, imgs, qs, n, t, progress),
            inputs=[question_box, images_gallery, q_session, n_a, temp],
            outputs=[chosen_radio, answers_md, a_session],
        ).then(
            # chosen/rejected 라디오는 같은 후보 목록을 공유한다
            lambda s: _visible_choices(len(s.get("candidates", [])), "후보"),
            inputs=[a_session], outputs=[rejected_radio],
        )

        #: 다음 항목으로 넘어갈 때 비울 컴포넌트 (이전 선택이 남아 오저장되는 것 방지)
        _clear_targets = [
            q_radio, question_box, chosen_radio, rejected_radio,
            answers_md, notes, a_session,
        ]
        _advance_inputs = [paper_dd, n_q, max_hops, require_img]
        _advance_outputs = [q_radio, evidence_md, images_gallery, q_session]

        save_btn.click(
            lambda q, a_s, q_s, ci, ri, m, an, nt:
                save_pair(ctx, q, a_s, q_s, ci, ri, m, an, nt),
            inputs=[question_box, a_session, q_session,
                    chosen_radio, rejected_radio, mode, annotator, notes],
            outputs=[save_msg, progress_md],
        ).then(
            # 저장이 성공했고 자동 진행이 켜져 있을 때만 다음 근거를 띄운다
            lambda msg, auto, pid, n, hops, ri, qs, progress=gr.Progress():
                maybe_advance(ctx, msg, auto, pid, n, hops, ri, qs, progress),
            inputs=[save_msg, auto_next, *_advance_inputs, q_session],
            outputs=_advance_outputs,
        ).then(
            # 새 질문이 떴으면 이전 후보/선택을 비운다 (질문 라디오는 위에서 갱신됨)
            lambda msg, auto: (
                clear_collection_state()[1:] if (auto and save_result_ok(msg))
                else tuple(gr.update() for _ in range(6))
            ),
            inputs=[save_msg, auto_next],
            outputs=_clear_targets[1:],
        )

        skip_btn.click(
            lambda: skip_and_next(ctx), outputs=[save_msg],
        ).then(
            lambda pid, n, hops, ri, progress=gr.Progress():
                generate_questions(ctx, pid, n, hops, ri, progress),
            inputs=_advance_inputs, outputs=_advance_outputs,
        ).then(
            lambda: clear_collection_state()[1:], outputs=_clear_targets[1:],
        )

        # 논문 이동 → 이동한 논문에서 바로 새 질문을 띄운다
        for _btn, _step in ((next_paper_btn, 1), (prev_paper_btn, -1)):
            _btn.click(
                (lambda s: lambda cur: advance_paper(ctx, cur, s))(_step),
                inputs=[paper_dd], outputs=[paper_dd, save_msg],
            ).then(
                lambda pid, n, hops, ri, progress=gr.Progress():
                    generate_questions(ctx, pid, n, hops, ri, progress),
                inputs=_advance_inputs, outputs=_advance_outputs,
            ).then(
                lambda: clear_collection_state()[1:], outputs=_clear_targets[1:],
            )
        refresh_btn.click(lambda: ctx.progress_markdown(), outputs=[progress_md])

        # ── 학습 탭 ───────────────────────────────────────────────────────
        _training_outputs = [
            train_status, metric_plot, loss_plot,
            history_table, adapter_table, adapter_dd,
        ]
        train_refresh.click(
            lambda: refresh_training_view(ctx), outputs=_training_outputs
        )
        train_btn.click(
            lambda f, l, progress=gr.Progress(): run_training(ctx, f, l, progress),
            inputs=[force_train, train_limit], outputs=[train_msg],
        ).then(
            lambda: refresh_training_view(ctx), outputs=_training_outputs
        ).then(
            # 학습으로 카운터가 리셋되고 모델이 바뀌므로 수집 탭 진행바도 갱신한다
            lambda: ctx.progress_markdown(), outputs=[progress_md]
        )
        activate_btn.click(
            lambda name: activate_adapter(ctx, name),
            inputs=[adapter_dd], outputs=[adapter_msg],
        ).then(lambda: refresh_training_view(ctx), outputs=_training_outputs)
        to_base_btn.click(
            lambda: activate_adapter(ctx, None), outputs=[adapter_msg]
        ).then(lambda: refresh_training_view(ctx), outputs=_training_outputs)

        # ── 추론 탭 ───────────────────────────────────────────────────────
        pg_upload.change(
            lambda files, cur: gr.update(
                value=_normalize_images(cur) + _normalize_images(files)
            ),
            inputs=[pg_upload, pg_images], outputs=[pg_images],
        )
        pg_btn.click(
            lambda q, imgs, mt, t, cmp_, progress=gr.Progress():
                playground_infer(ctx, q, imgs, mt, t, cmp_, progress),
            inputs=[pg_question, pg_images, pg_max_tokens, pg_temp, pg_compare],
            outputs=[pg_output],
        )

        # ── 모델 비교 탭 ──────────────────────────────────────────────────
        cmp_session = gr.State({})

        cmp_refresh.click(
            lambda m: comparison_targets(ctx, m),
            inputs=[cmp_metric], outputs=[cmp_adapter, cmp_best_note],
        )
        cmp_metric.change(
            lambda m: comparison_targets(ctx, m),
            inputs=[cmp_metric], outputs=[cmp_adapter, cmp_best_note],
        )
        cmp_upload.change(
            lambda files, cur: gr.update(
                value=_normalize_images(cur) + _normalize_images(files)
            ),
            inputs=[cmp_upload, cmp_images], outputs=[cmp_images],
        )
        cmp_load_eval.click(
            lambda i: load_eval_question(ctx, i),
            inputs=[cmp_eval_idx], outputs=[cmp_question, cmp_images, cmp_eval_note],
        )
        cmp_run.click(
            lambda q, imgs, ad, m, mt, t, progress=gr.Progress():
                run_comparison(ctx, q, imgs, ad, m, mt, t, progress),
            inputs=[cmp_question, cmp_images, cmp_adapter, cmp_metric,
                    cmp_tokens, cmp_temp],
            outputs=[cmp_base_out, cmp_tuned_out, cmp_header, cmp_session],
        ).then(
            # 새 비교가 나왔으니 이전 판정을 비운다 (직전 판정이 남아 오기록되는 것 방지)
            lambda: (gr.update(value=None), gr.update(value=""), gr.update(value="")),
            outputs=[cmp_verdict, cmp_note_box, cmp_save_msg],
        )
        cmp_save.click(
            lambda s, v, n: save_comparison(ctx, s, v, n),
            inputs=[cmp_session, cmp_verdict, cmp_note_box],
            outputs=[cmp_save_msg, cmp_stats],
        )

        # ── 내보내기 탭 ───────────────────────────────────────────────────
        export_btn.click(
            lambda f, m, c, progress=gr.Progress(): run_export(ctx, f, m, c, progress),
            inputs=[export_fmt, export_map, export_copy],
            outputs=[export_msg, export_file],
        ).then(lambda: export_preview(ctx), outputs=[export_summary])
        export_refresh.click(lambda: export_preview(ctx), outputs=[export_summary])

        # ── 설정 탭 ───────────────────────────────────────────────────────
        _settings_fields = [
            set_language, set_data_root, set_pairs, set_kgjson,
            set_exportdir, set_adapterdir,
            set_model_name, set_model_backend, set_model_dtype,
            set_model_tokens, set_model_temp, set_model_tiles,
            set_qg_name, set_qg_backend, set_qg_n, set_qg_prompt,
            set_ag_n, set_ag_temp, set_ag_prompt,
            set_kg_max, set_kg_lang,
            set_metrics, set_eval_tokens,
            set_first, set_retrain, set_per_type,
            set_beta, set_lr, set_epochs, set_train_tiles,
            set_export_fmt, set_field_map,
        ]
        settings_save_btn.click(
            lambda *vals: save_settings_form(ctx, *vals),
            inputs=_settings_fields, outputs=[settings_msg, yaml_box],
        ).then(
            lambda: gr.update(choices=config_io.list_backups(ctx.config_path)),
            outputs=[backup_dd],
        ).then(
            # 설정이 바뀌면 다른 탭 표시도 갱신한다
            lambda: ctx.progress_markdown(), outputs=[progress_md],
        )
        settings_reset_btn.click(
            lambda: settings_snapshot(ctx), outputs=_settings_fields,
        )

        yaml_save_btn.click(
            lambda text: save_settings_yaml(ctx, text),
            inputs=[yaml_box], outputs=[yaml_msg, backup_dd],
        ).then(
            lambda: settings_snapshot(ctx), outputs=_settings_fields,
        ).then(
            lambda: ctx.progress_markdown(), outputs=[progress_md],
        )
        yaml_reload_btn.click(lambda: settings_yaml(ctx), outputs=[yaml_box])

        backup_refresh_btn.click(
            lambda: gr.update(choices=config_io.list_backups(ctx.config_path)),
            outputs=[backup_dd],
        )
        backup_restore_btn.click(
            lambda name: restore_settings_backup(ctx, name),
            inputs=[backup_dd], outputs=[backup_msg, yaml_box],
        ).then(
            lambda: settings_snapshot(ctx), outputs=_settings_fields,
        )

        # 시작 시 저장된 KG 가 있으면 자동 로드해 논문 목록을 채운다
        demo.load(
            lambda: _autoload_kg(ctx),
            outputs=[kg_stats, paper_dd],
        ).then(
            lambda: refresh_training_view(ctx), outputs=_training_outputs
        )

    return demo


def _autoload_kg(ctx: AppContext) -> Tuple[str, Any]:
    """저장된 KG JSON 이 있으면 조용히 로드한다 (없으면 안내만)."""
    try:
        if ctx.bridge.load() is not None:
            n_excluded = ctx.apply_eval_exclusion()
            s = ctx.bridge.stats()
            held = ctx.bridge.excluded_papers
            selectable = [p for p in ctx.bridge.papers_in_graph() if p not in held]
            return (
                f"### ✅ 저장된 KG 로드됨 — 논문 {s['total_papers']:,}편, "
                f"노드 {s['total_nodes']:,} / 엣지 {s['total_edges']:,}\n\n"
                + (
                    f"🔒 평가 전용 {n_excluded}편 제외 → 수집 가능 {len(selectable):,}편\n\n"
                    if n_excluded else ""
                )
                + "다시 만들려면 '강제 재구축'을 켜고 버튼을 누르세요.",
                gr.update(
                    choices=selectable, value=selectable[0] if selectable else None
                ),
            )
    except Exception as e:
        logger.warning(f"[UI] KG 자동 로드 실패: {e}")
    return (
        "저장된 KG가 없습니다. 위에서 **KG 구축 / 로드** 를 눌러 시작하세요.",
        gr.update(choices=[]),
    )


# ─── 엔트리포인트 ──────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="한국어 멀티모달 DPO 수집 UI")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Gradio 공개 링크 생성")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    ctx = AppContext.load(args.config)
    demo = build_ui(ctx)
    # 이미지 갤러리가 data_ko 아래 파일을 읽어야 하므로 허용 경로에 추가한다
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(Path(ctx.bridge.data_root).resolve())],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
