"""
dpo_collector/kg_bridge.py
---------------------------
data_ko(한국어 과학기술 문서) → Knowledge Graph 구축/로드 + 질문용 경로 샘플링.

코어 `knowledge_graph/*` 를 수정하지 않고 import 로만 재사용한다:
  - `KGPipeline`        : 구조/참조/의미 3-stage KG 구축
  - `NetworkXGraphStore`: 그래프 저장/조회/직렬화
  - `NodeType/EdgeType` : 노드·엣지 타입
  - `self_play.question_generator.EDGE_TO_DIFFICULTY / EDGE_TO_QTYPE` : 엣지→난이도/유형 매핑

한국어 대응 (코어가 영어 전제로 작성되어 있어 이 모듈에서 보강):
  1. **참조 엣지 보강** — 코어 `reference_graph.py` 의 정규식은 "Figure 3"/"Table 2" 같은
     영문 표현만 잡는다(패턴이 하드코딩되어 config 로도 못 바꿈). data_ko 표본 60편 기준
     한국어 참조("그림 1", "표 2", "식 (3)")가 전체 참조의 약 46% 를 차지하므로,
     KG 구축 후 이 모듈에서 한국어 REFERENCES 엣지를 추가한다.
  2. **의미/교차문서 단계 기본 비활성화** — 코어가 쓰는 임베딩 모델이
     `all-MiniLM-L6-v2`(영어 전용, 하드코딩)라 한국어 유사도가 신뢰할 수 없다.
     config 로 켤 수는 있게 두되 기본값은 off.
  3. **gold answer 추출** — 코어의 단어 추출 정규식이 `[a-zA-Z]{4,}` 라 한글이 잡히지 않는다.
     여기서는 한글을 포함한 패턴을 쓴다.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knowledge_graph.graph_store import NetworkXGraphStore
from knowledge_graph.kg_pipeline import KGPipeline
from knowledge_graph.parser import PaddleJSONParser
from knowledge_graph.schema import EdgeType, KGEdge, KGNode, NodeType

logger = logging.getLogger(__name__)

# ─── 엣지 → 난이도/유형 매핑 ───────────────────────────────────────────────
# `self_play` 가 있으면 코어 매핑을 그대로 쓰고, 없으면 동일한 내장 사본을 쓴다.
# (수집기 단독 배포본에는 self_play 가 포함되지 않는다 — 오류가 아니라 정상 경로다)
try:
    from self_play.question_generator import EDGE_TO_DIFFICULTY, EDGE_TO_QTYPE
except Exception:  # pragma: no cover - self_play 미포함 환경
    logger.debug("[KGBridge] self_play 미포함 — 내장 엣지 매핑을 사용합니다.")
    EDGE_TO_DIFFICULTY = {
        "CONTAINS": 1, "HAS_CAPTION": 1, "DEFINES": 2, "QUANTIFIES": 2,
        "SUPPORTS": 2, "ILLUSTRATES": 2, "REFERENCES": 2, "DERIVES_FROM": 3,
        "COMPARES": 3, "CONTRADICTS": 3, "SAME_CONCEPT": 4, "CROSS_DOC": 4,
    }
    EDGE_TO_QTYPE = {
        "CONTAINS": "VQA", "HAS_CAPTION": "VQA", "DEFINES": "VQA",
        "QUANTIFIES": "VQA", "SUPPORTS": "Reasoning", "ILLUSTRATES": "VQA",
        "REFERENCES": "Reasoning", "DERIVES_FROM": "Reasoning",
        "COMPARES": "MCQ", "CONTRADICTS": "Reasoning",
        "SAME_CONCEPT": "InstructionFollowing", "CROSS_DOC": "InstructionFollowing",
    }


# ─── 한국어 참조 표현 정규식 ───────────────────────────────────────────────
# "그림 1", "그림1", "<그림 2>", "[그림 3]", "그림 4에서" 등을 포괄한다.
_KO_FIG_REFS = re.compile(r"(?:그\s?림|사진|도표|도해)\s*[<\[(]?\s*(\d+)")
_KO_TAB_REFS = re.compile(r"(?:표|테이블)\s*[<\[(]?\s*(\d+)")
_KO_EQ_REFS  = re.compile(r"(?:식|수식)\s*[<\[(]?\s*(\d+)")

#: 한글/영문/숫자를 포함하는 내용어 (gold answer 교집합 추출용)
_WORD_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{4,}|\d+(?:\.\d+)?")

#: 시각 근거가 있는 노드 타입 — 멀티모달 DPO 수집의 우선 대상
VISUAL_NODE_TYPES = (NodeType.FIGURE, NodeType.TABLE)

#: HTML 태그 (Table 노드 content 는 PaddleOCR 이 만든 HTML 표다)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_content(node: KGNode, max_len: int = 0) -> str:
    """
    노드 content 를 사람이 읽을 수 있는 평문으로 정리한다.

    Table 노드의 content 는 `<table border=1 style='…'><tr><td>…` 형태의 HTML 이라
    그대로 프롬프트나 gold answer 에 넣으면 태그가 내용을 압도한다. 셀 구분을 `|` 로
    바꿔 표 구조는 남기고 태그만 제거한다.
    """
    text = (node.content or "").strip()
    if not text:
        return ""
    if "<" in text and ">" in text:
        # 셀/행 경계를 구분자로 치환한 뒤 나머지 태그 제거
        text = re.sub(r"</t[dh]>\s*", " | ", text)
        text = re.sub(r"</tr>\s*", "\n", text)
        text = _HTML_TAG_RE.sub(" ", text)
        text = "\n".join(
            _WS_RE.sub(" ", line).strip(" |").strip()
            for line in text.splitlines()
        )
        text = "\n".join(l for l in text.splitlines() if l)
    else:
        text = _WS_RE.sub(" ", text).strip()

    if max_len and len(text) > max_len:
        text = text[:max_len] + "…"
    return text


# ─── 논문 언어 판정 ────────────────────────────────────────────────────────
#
# `data_ko` 에는 한국어 논문과 영어 논문이 섞여 있다(실측 949편 중 한국어 771·영어 177).
# 한국어 DPO 를 수집하면서 영어 논문을 근거로 삼으면 질문·답변 언어가 어긋나므로
# 본문의 한글 비율로 논문 언어를 판정해 걸러낸다.

_HANGUL_RE = re.compile(r"[가-힣]")
_LETTER_RE = re.compile(r"[A-Za-z가-힣]")

#: 이 비율 이상이면 한국어 문서로 본다 (영문 초록·용어가 섞여도 통과하도록 낮게 잡음)
HANGUL_RATIO_THRESHOLD = 0.15

#: 언어 판정 결과 캐시 파일명 (전체 스캔은 수백 편 기준 수십 초가 걸린다)
LANG_CACHE_NAME = "paper_languages.json"


def detect_paper_language(
    json_path: Path, sample_pages: int = 4, max_chars: int = 6000
) -> str:
    """
    `parsing_paddle.json` 앞부분 본문의 한글 비율로 언어를 판정한다.

    Returns: "ko" | "en" | "unknown"(본문이 너무 적어 판정 불가)
    """
    try:
        import json as _json

        with open(json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return "unknown"

    text = " ".join(
        e.get("text", "")
        for page in data.get("pages", [])[:sample_pages]
        for e in page.get("text_elements", [])
    )[:max_chars]

    letters = len(_LETTER_RE.findall(text))
    if letters < 50:
        return "unknown"
    return "ko" if len(_HANGUL_RE.findall(text)) / letters >= HANGUL_RATIO_THRESHOLD else "en"


def scan_paper_languages(
    data_root: str,
    cache_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """
    data_root 아래 모든 논문의 언어를 판정한다. 결과는 JSON 으로 캐시한다.

    Returns: {paper_id: "ko" | "en" | "unknown"}
    """
    import json as _json

    root = Path(data_root)
    cache = Path(cache_path) if cache_path else root.parent / LANG_CACHE_NAME

    cached: Dict[str, str] = {}
    if cache.exists() and not force:
        try:
            cached = _json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[KGBridge] 언어 캐시 파싱 실패, 다시 스캔합니다: {e}")

    result: Dict[str, str] = {}
    scanned = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        j = d / "parsing_paddle.json"
        if not j.exists():
            continue
        if d.name in cached:
            result[d.name] = cached[d.name]
            continue
        result[d.name] = detect_paper_language(j)
        scanned += 1

    if scanned or not cache.exists():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                _json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            logger.info(
                f"[KGBridge] 논문 언어 판정 캐시 저장 → {cache} "
                f"(새로 스캔 {scanned}편 / 전체 {len(result)}편)"
            )
        except Exception as e:
            logger.warning(f"[KGBridge] 언어 캐시 저장 실패: {e}")

    return result


def is_caption_node(node: KGNode) -> bool:
    """
    Figure/Table 캡션으로 생성된 TextBlock 인지 판별.
    코어 structural_graph.py 가 `metadata={"is_caption": True}` 로 표시하고
    node_id 는 `{paper_id}__cap_fig_{n}` 형태다.
    """
    if node.metadata.get("is_caption"):
        return True
    return bool(re.search(r"__cap_(?:fig|tab|eq)_\d+$", node.node_id))


# ─── KG 경로 (질문 1건의 근거) ─────────────────────────────────────────────

@dataclass
class KGPath:
    """
    KG 상의 경로 1개 = 질문 1개의 근거.
    `question_gen.py` 가 이걸 받아 한국어 질문을 만들고,
    `store.py` 가 `kg_provenance` 필드로 저장한다.
    """
    paper_id:      str
    node_ids:      List[str]
    nodes:         List[KGNode]
    edge_types:    List[str]
    difficulty:    int
    question_type: str
    gold_answer:   str = ""
    image_paths:   List[str] = field(default_factory=list)   # 절대경로
    context:       str = ""

    @property
    def hop(self) -> int:
        return len(self.edge_types)

    @property
    def has_image(self) -> bool:
        return bool(self.image_paths)

    def provenance(self) -> Dict[str, Any]:
        """`dpo_pairs.jsonl::kg_provenance` 형태로 직렬화 (CLAUDE_dpo.md §5.1)."""
        return {
            "src_node": self.node_ids[0] if self.node_ids else "",
            "edge":     self.edge_types[0] if self.edge_types else "",
            "dst_node": self.node_ids[-1] if len(self.node_ids) > 1 else "",
            "hop":      self.hop,
            "path":     self.node_ids,
            "edges":    self.edge_types,
        }

    def summary(self) -> str:
        """UI 표시용 한 줄 요약."""
        types = " → ".join(
            f"[{n.node_type.value}]" for n in self.nodes
        )
        edges = " / ".join(self.edge_types) or "single-node"
        return f"{types}  ({edges}, L{self.difficulty}, {self.question_type})"


# ─── KG 브리지 ─────────────────────────────────────────────────────────────

class KGBridge:
    """
    data_ko → KG 구축/로드 및 질문 근거 샘플링 담당.

        bridge = KGBridge(data_root="ICML_workshop/data_ko",
                          kg_json="dpo_collector/outputs/kg_ko.json")
        store  = bridge.build_or_load(paper_ids=bridge.list_papers()[:20])
        paths  = bridge.sample_paths(n=5, prefer_visual=True)
    """

    def __init__(
        self,
        data_root: str,
        kg_json: Optional[str] = None,
        kg_cfg: Optional[Dict[str, Any]] = None,
        augment_korean_refs: bool = True,
        lang_cache_path: Optional[str] = None,
    ):
        self.data_root = str(data_root)
        self.kg_json = kg_json
        self.augment_korean_refs = augment_korean_refs
        self.lang_cache_path = lang_cache_path
        self.kg_cfg = self._resolve_kg_cfg(kg_cfg)

        self._store: Optional[NetworkXGraphStore] = None
        self._parser = PaddleJSONParser(self.data_root)
        self._lang_map: Optional[Dict[str, str]] = None
        # 샘플링에서 제외할 논문 (평가 전용 held-out).
        # KG 자체는 평가 문항을 만들기 위해 held-out 도 포함해 구축하므로,
        # **수집용 샘플링 단계**에서 막아야 평가셋 누출이 구조적으로 차단된다(§1-4).
        self._excluded_papers: set = set()
        # 샘플링 시작 노드 후보 캐시.
        # 전체 코퍼스 KG 는 10만 노드 규모라 후보를 매번 새로 훑으면
        # 질문 1건마다 수십만 번 비교가 일어난다. (키: 필터 조합)
        self._candidate_cache: Dict[Tuple[Optional[str], bool, bool], List[KGNode]] = {}

    # ── 설정 ──────────────────────────────────────────────────────────────

    def _resolve_kg_cfg(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        코어 `config/config.yaml::knowledge_graph` 를 베이스로 하고,
        한국어 데모에 맞는 기본값 + 사용자 override 를 얹는다.
        코어 config 파일 자체는 수정하지 않는다.
        """
        base: Dict[str, Any] = {}
        try:
            from config.config_loader import get_config

            import copy

            base = copy.deepcopy(get_config().get("knowledge_graph", {}))
        except Exception as e:
            logger.warning(f"[KGBridge] 코어 config 로드 실패, 기본값 사용: {e}")

        base.setdefault("backend", "networkx")
        base["backend"] = "networkx"   # 데모는 Neo4j 미사용

        # 한국어 기본값 — 영어 전용 임베딩 단계는 끈다 (사유는 모듈 docstring 참고).
        # 아래 override 에서 명시적으로 켜면 그 값이 우선한다.
        for section in ("semantic", "cross_document", "vlm_enrichment"):
            base.setdefault(section, {})
            base[section]["enabled"] = False

        # 사용자 override 병합 (섹션 단위 deep merge)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        return base

    # ── 논문 목록 / 구축 / 로드 ───────────────────────────────────────────

    def list_papers(self, language: Optional[str] = None) -> List[str]:
        """
        `parsing_paddle.json` 이 있는 논문 폴더명 목록.

        Args:
            language: "ko" 등을 주면 본문 언어가 일치하는 논문만 반환한다.
                      (data_ko 에는 영어 논문이 섞여 있다 — `scan_paper_languages` 참고)
        """
        papers = self._parser.list_papers()
        if not language:
            return papers
        langs = self.paper_languages()
        filtered = [p for p in papers if langs.get(p) == language]
        logger.info(
            f"[KGBridge] 언어 필터 '{language}': {len(filtered)}/{len(papers)}편"
        )
        return filtered

    def set_excluded_papers(self, paper_ids: Optional[Sequence[str]]) -> None:
        """
        수집용 샘플링에서 제외할 논문을 지정한다 (보통 평가 전용 held-out).
        `sample_paths` / `path_from_node` 가 이 목록의 논문을 절대 반환하지 않는다.
        """
        new = set(paper_ids or ())
        if new != self._excluded_papers:
            self._excluded_papers = new
            self._invalidate_caches()
            logger.info(f"[KGBridge] 샘플링 제외 논문 {len(new)}편 설정 (평가셋 보호)")

    @property
    def excluded_papers(self) -> set:
        return set(self._excluded_papers)

    def paper_languages(self, force_rescan: bool = False) -> Dict[str, str]:
        """논문별 언어 판정 결과 (캐시 사용)."""
        if self._lang_map is None or force_rescan:
            self._lang_map = scan_paper_languages(
                self.data_root, self.lang_cache_path, force=force_rescan
            )
        return self._lang_map

    def language_stats(self) -> Dict[str, int]:
        """언어별 논문 수 (UI 표시용)."""
        counts: Dict[str, int] = {}
        for lang in self.paper_languages().values():
            counts[lang] = counts.get(lang, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def build(
        self,
        paper_ids: Optional[List[str]] = None,
        save: bool = True,
    ) -> NetworkXGraphStore:
        """KG 를 새로 구축한다. `paper_ids=None` 이면 전체."""
        logger.info(
            f"[KGBridge] KG 구축 시작 — data_root={self.data_root}, "
            f"papers={'전체' if paper_ids is None else len(paper_ids)}"
        )
        pipeline = KGPipeline(self.kg_cfg, self.data_root)
        store = pipeline.run(paper_ids=paper_ids)

        if self.augment_korean_refs:
            # 코어 ReferenceGraphBuilder 도 `reference.language: "ko"` 면 한국어 패턴을
            # 매칭한다. 이미 만들어진 엣지는 건너뛰므로 중복은 생기지 않고,
            # 코어 패턴이 놓치는 형태(`<그림 2>`, `[표 3]` 처럼 괄호로 감싼 표기)만 채운다.
            added = augment_korean_references(store)
            core_ko = str(
                self.kg_cfg.get("reference", {}).get("language", "")
            ).lower().startswith("ko")
            logger.info(
                f"[KGBridge] 한국어 참조 엣지 {added}개 추가 보강 "
                f"(코어 한국어 패턴 {'적용됨' if core_ko else '미적용'})"
            )

        self._store = store
        self._invalidate_caches()
        if save and self.kg_json:
            Path(self.kg_json).parent.mkdir(parents=True, exist_ok=True)
            store.save(self.kg_json)
        return store

    def load(self) -> Optional[NetworkXGraphStore]:
        """저장된 KG JSON 을 로드한다. 없으면 None."""
        if not self.kg_json or not Path(self.kg_json).exists():
            return None
        store = NetworkXGraphStore()
        store.load(self.kg_json)
        self._store = store
        self._invalidate_caches()
        logger.info(f"[KGBridge] KG 로드 완료 ← {self.kg_json}")
        return store

    def build_or_load(
        self,
        paper_ids: Optional[List[str]] = None,
        force_rebuild: bool = False,
    ) -> NetworkXGraphStore:
        """저장된 KG 가 있으면 로드, 없거나 `force_rebuild` 면 새로 구축."""
        if not force_rebuild:
            store = self.load()
            if store is not None:
                return store
        return self.build(paper_ids=paper_ids)

    @property
    def store(self) -> NetworkXGraphStore:
        if self._store is None:
            raise RuntimeError(
                "KG 가 아직 준비되지 않았습니다 — build() / load() / build_or_load() 를 먼저 호출하세요."
            )
        return self._store

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    # ── 통계 (UI 'KG 구축' 탭용) ──────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """노드/엣지 통계 + 논문 수 + 이미지 보유 노드 수."""
        s = dict(self.store.stats())
        papers = self.store.get_nodes_by_type(NodeType.PAPER)
        s["total_papers"] = len(papers)
        s["visual_nodes"] = sum(
            len(self.store.get_nodes_by_type(t)) for t in VISUAL_NODE_TYPES
        )
        s["korean_ref_edges"] = sum(
            1
            for _, _, d in self.store._g.edges(data=True)
            if d.get("metadata", {}).get("method") == "rule_ko"
        )
        return s

    # ── 경로 샘플링 ───────────────────────────────────────────────────────

    def sample_paths(
        self,
        n: int = 5,
        paper_id: Optional[str] = None,
        max_hops: int = 2,
        prefer_visual: bool = True,
        require_image: bool = False,
        difficulty: Optional[int] = None,
        skip_captions: bool = True,
        seed: Optional[int] = None,
    ) -> List[KGPath]:
        """
        질문 근거가 될 KG 경로를 샘플링한다.

        Args:
            n:             뽑을 경로 수
            paper_id:      특정 논문으로 제한 (None = 전체)
            max_hops:      최대 홉 수 (1~3 권장)
            prefer_visual: Figure/Table 노드를 시작점으로 우선 선택 (멀티모달 수집용)
            require_image: 이미지가 붙지 않는 경로는 버린다
            difficulty:    특정 난이도(1~4)만 반환 (None = 제한 없음)
            skip_captions: 캡션 TextBlock 을 시작점에서 제외 (자명한 경로 방지)
            seed:          재현성용 난수 시드
        """
        rng = random.Random(seed)
        starts = self._candidate_start_nodes(paper_id, prefer_visual, skip_captions)
        if not starts:
            logger.warning(f"[KGBridge] 후보 노드 없음 (paper_id={paper_id})")
            return []

        paths: List[KGPath] = []
        # 후보가 부족해 무한루프에 빠지지 않도록 시도 횟수를 제한한다
        for _ in range(n * 12):
            if len(paths) >= n:
                break
            start = rng.choice(starts)
            hops = rng.randint(1, max(1, max_hops))
            path = self._build_path(start, hops, rng)
            if path is None:
                continue
            if require_image and not path.has_image:
                continue
            if difficulty is not None and path.difficulty != difficulty:
                continue
            if any(p.node_ids == path.node_ids for p in paths):
                continue   # 동일 경로 중복 제거
            paths.append(path)

        if len(paths) < n:
            logger.info(
                f"[KGBridge] 요청 {n}개 중 {len(paths)}개 샘플링 "
                f"(조건: hops≤{max_hops}, require_image={require_image}, difficulty={difficulty})"
            )
        return paths

    def path_from_node(self, node_id: str, max_hops: int = 1) -> Optional[KGPath]:
        """특정 노드를 시작점으로 하는 경로 1개 (UI 에서 노드 직접 선택 시)."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        return self._build_path(node, max_hops, random.Random())

    def _candidate_start_nodes(
        self,
        paper_id: Optional[str],
        prefer_visual: bool,
        skip_captions: bool = True,
    ) -> List[KGNode]:
        """
        샘플링 시작 노드 후보.

        - `skip_captions`: 캡션 TextBlock 은 제외한다. 캡션은 자기 Figure 를 그대로
          가리키는 자명한 REFERENCES 경로("Fig. 1 …" → fig_1)를 만들어내는데,
          내용이 동일해 질문 근거로서 정보량이 없다.
        - `prefer_visual`: 본문 TextBlock 이 시각 노드보다 10배 가까이 많으므로
          (전체 코퍼스 실측 89,425 vs 8,982) 시각 노드를 크게 가중해야 실제로 뽑힌다.

        결과는 필터 조합별로 캐시한다 — 10만 노드 KG 에서 매 호출마다 전체를 훑으면
        질문 하나 뽑는 데 수 초가 걸린다.
        """
        cache_key = (paper_id, prefer_visual, skip_captions)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        types = [
            NodeType.FIGURE, NodeType.TABLE,
            NodeType.CONCEPT, NodeType.CLAIM, NodeType.TEXT_BLOCK,
        ]
        out: List[KGNode] = []
        for t in types:
            nodes = self.store.get_nodes_by_type(t)
            if paper_id:
                nodes = [x for x in nodes if x.paper_id == paper_id]
            if self._excluded_papers:
                # 평가 전용 논문은 수집 샘플링에서 원천 배제 (§1-4)
                nodes = [x for x in nodes if x.paper_id not in self._excluded_papers]
            # 내용이 비어 있는 노드는 질문 근거가 되지 못한다
            nodes = [
                x for x in nodes
                if (x.content or "").strip() or x.image_path
            ]
            if skip_captions and t == NodeType.TEXT_BLOCK:
                nodes = [x for x in nodes if not is_caption_node(x)]
            if prefer_visual and t in VISUAL_NODE_TYPES:
                out.extend(nodes * 10)
            else:
                out.extend(nodes)

        self._candidate_cache[cache_key] = out
        return out

    def _invalidate_caches(self) -> None:
        """KG 가 새로 구축/로드되면 파생 캐시를 버린다."""
        self._candidate_cache.clear()

    def _build_path(
        self, start: KGNode, max_hops: int, rng: random.Random
    ) -> Optional[KGPath]:
        """시작 노드에서 랜덤 워크하여 KGPath 를 구성한다."""
        g = self.store._g
        node_ids = [start.node_id]
        nodes = [start]
        edge_types: List[str] = []

        current = start.node_id
        for _ in range(max_hops):
            succ = [s for s in g.successors(current) if s not in node_ids]
            if not succ:
                break
            nxt = rng.choice(succ)
            edge_data = g.get_edge_data(current, nxt) or {}
            if not edge_data:
                break
            et = list(edge_data.values())[0].get("edge_type", "CONTAINS")
            nxt_node = self.store.get_node(nxt)
            if nxt_node is None:
                break
            node_ids.append(nxt)
            nodes.append(nxt_node)
            edge_types.append(et)
            current = nxt

        # 단일 노드 경로도 유효하다 (난이도 1: "이 그림은 무엇을 보여주는가?")
        if len(nodes) == 1 and not (start.content or "").strip() and not start.image_path:
            return None

        difficulty = (
            max(EDGE_TO_DIFFICULTY.get(et, 2) for et in edge_types)
            if edge_types else 1
        )
        primary_edge = edge_types[0] if edge_types else "CONTAINS"
        q_type = EDGE_TO_QTYPE.get(primary_edge, "VQA")

        return KGPath(
            paper_id=start.paper_id,
            node_ids=node_ids,
            nodes=nodes,
            edge_types=edge_types,
            difficulty=difficulty,
            question_type=q_type,
            gold_answer=extract_gold_answer(nodes, edge_types),
            image_paths=self.collect_images(nodes),
            context=build_context(nodes, edge_types),
        )

    # ── 이미지 경로 처리 ──────────────────────────────────────────────────

    def collect_images(self, nodes: List[KGNode], limit: int = 4) -> List[str]:
        """
        경로 상 노드들의 이미지를 절대경로로 수집한다.
        KGNode.image_path 는 data_root 기준 상대경로이므로 여기서 해석한다.
        """
        out: List[str] = []
        for n in nodes:
            if not n.image_path:
                continue
            abs_path = self.resolve_image(n.image_path)
            if abs_path and abs_path not in out:
                out.append(abs_path)
            if len(out) >= limit:
                break
        return out

    def resolve_image(self, image_path: str) -> Optional[str]:
        """상대/절대 이미지 경로 → 존재가 확인된 절대경로. 없으면 None."""
        p = Path(image_path)
        if p.is_absolute():
            return str(p) if p.is_file() else None

        # KGNode.image_path 는 보통 "<data_root 이름>/<paper>/images/..." 형태다
        root = Path(self.data_root).resolve()
        for cand in (root.parent / p, root / p, Path.cwd() / p):
            if cand.is_file():
                return str(cand.resolve())
        return None

    # ── 논문 단위 조회 (UI 문서 선택용) ───────────────────────────────────

    def paper_images(self, paper_id: str, limit: int = 50) -> List[Tuple[str, str]]:
        """논문의 Figure/Table 이미지 목록 → [(node_id, 절대경로), ...]."""
        out: List[Tuple[str, str]] = []
        for t in VISUAL_NODE_TYPES:
            for n in self.store.get_nodes_by_type(t):
                if n.paper_id != paper_id or not n.image_path:
                    continue
                abs_path = self.resolve_image(n.image_path)
                if abs_path:
                    out.append((n.node_id, abs_path))
                if len(out) >= limit:
                    return out
        return out

    def papers_in_graph(self) -> List[str]:
        """현재 KG 에 포함된 논문 ID 목록."""
        return sorted(
            {n.paper_id for n in self.store.get_nodes_by_type(NodeType.PAPER)}
        )


# ─── 한국어 참조 엣지 보강 ─────────────────────────────────────────────────

def augment_korean_references(store: NetworkXGraphStore) -> int:
    """
    본문 TextBlock 에서 한국어 참조 표현("그림 1", "표 2", "식 (3)")을 찾아
    REFERENCES 엣지를 추가한다. 코어 `reference_graph.py` 와 동일한 순번 매핑 규칙
    (`{paper_id}__fig_{n}` 등)을 따르되 한국어 정규식을 쓴다.

    Returns:
        추가된 엣지 수.
    """
    # 논문별 "참조 번호 → node_id" 맵.
    # 코어 structural_graph.py 가 `{paper_id}__fig_{n}` 형태로 등장 순 번호를 부여하고
    # reference_graph.py 도 같은 번호로 매칭하므로, node_id 의 접미 번호를 그대로 쓴다.
    fig_map: Dict[str, Dict[str, str]] = {}
    tab_map: Dict[str, Dict[str, str]] = {}
    eq_map:  Dict[str, Dict[str, str]] = {}

    for ntype, target, kind in (
        (NodeType.FIGURE,   fig_map, "fig"),
        (NodeType.TABLE,    tab_map, "tab"),
        (NodeType.EQUATION, eq_map,  "eq"),
    ):
        for node in store.get_nodes_by_type(ntype):
            m = re.search(rf"__{kind}_(\d+)$", node.node_id)
            if m:
                target.setdefault(node.paper_id, {})[m.group(1)] = node.node_id

    text_nodes = [
        n for n in store.get_nodes_by_type(NodeType.TEXT_BLOCK)
        if (n.content or "").strip()
    ]

    added = 0
    for tb in text_nodes:
        content = tb.content or ""
        for pattern, mapping, label in (
            (_KO_FIG_REFS, fig_map, "그림"),
            (_KO_TAB_REFS, tab_map, "표"),
            (_KO_EQ_REFS,  eq_map,  "식"),
        ):
            paper_map = mapping.get(tb.paper_id, {})
            if not paper_map:
                continue
            for num in set(pattern.findall(content)):
                dst_id = paper_map.get(num)
                if not dst_id or dst_id == tb.node_id:
                    continue
                # 이미 같은 REFERENCES 엣지가 있으면 건너뛴다 (영문 규칙과 중복 방지)
                existing = store._g.get_edge_data(tb.node_id, dst_id) or {}
                if EdgeType.REFERENCES.value in existing:
                    continue
                store.add_edge(KGEdge(
                    src_id=tb.node_id,
                    dst_id=dst_id,
                    edge_type=EdgeType.REFERENCES,
                    weight=0.8,
                    confidence=0.9,
                    metadata={
                        "method": "rule_ko",
                        "evidence": f"ref {label} {num}",
                    },
                ))
                added += 1
    return added


# ─── gold answer / context 추출 (한국어 대응) ──────────────────────────────

def extract_gold_answer(nodes: List[KGNode], edge_types: List[str]) -> str:
    """
    KG 경로의 마지막 노드에서 gold answer 후보를 뽑는다.
    수집 UI 에서 사람이 chosen/rejected 를 고를 때의 **참고용**이며 정답 확정치가 아니다.

    코어 `question_generator._extract_gold_answer` 와 같은 전략이되,
    단어 추출 정규식이 한글을 포함하도록 바꿨다.
    """
    if not nodes:
        return ""
    target = nodes[-1]
    content = clean_content(target)
    last_edge = edge_types[-1] if edge_types else ""

    if last_edge in ("QUANTIFIES", "MEASURES"):
        nums = re.findall(r"\d+\.?\d*", content)
        return nums[0] if nums else content[:100]

    if last_edge == "DEFINES":
        # 한국어 문장은 마침표 외에 '다.' 로도 끝난다
        parts = re.split(r"(?<=다)\.|\.", content)
        first = next((s.strip() for s in parts if s.strip()), "")
        return first or content[:100]

    if last_edge == "SAME_CONCEPT" and len(nodes) >= 2:
        words_a = set(_WORD_RE.findall(clean_content(nodes[-2])))
        words_b = set(_WORD_RE.findall(content))
        common = words_a & words_b
        return " ".join(sorted(common)[:5]) if common else content[:100]

    return content


def build_context(nodes: List[KGNode], edge_types: List[str]) -> str:
    """
    질문 생성 프롬프트에 넣을 경로 컨텍스트 문자열.
    노드 타입/페이지와 내용 일부를 사람이 읽을 수 있는 형태로 이어붙인다.
    """
    lines: List[str] = []
    for i, n in enumerate(nodes):
        content = clean_content(n, max_len=300).replace("\n", " ")
        head = f"[{n.node_type.value}] (p.{n.page})" if n.page >= 0 else f"[{n.node_type.value}]"
        img = " (이미지 있음)" if n.image_path else ""
        lines.append(f"{head}{img} {content}")
        if i < len(edge_types):
            lines.append(f"   --{edge_types[i]}-->")
    return "\n".join(lines)


# ─── 설정 파일에서 브리지 생성 ─────────────────────────────────────────────

def from_config(config_path: str = "dpo_collector/config_dpo.yaml") -> KGBridge:
    """`config_dpo.yaml` 의 `paths` / `kg` 섹션으로 KGBridge 를 만든다."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths", {})
    kg = cfg.get("kg", {})
    return KGBridge(
        data_root=paths.get("data_ko_root", "data_ko"),
        kg_json=paths.get("kg_json"),
        kg_cfg=kg.get("overrides"),
        augment_korean_refs=bool(kg.get("augment_korean_refs", True)),
        lang_cache_path=paths.get("lang_cache_json"),
    )


def papers_for_collection(
    bridge: KGBridge,
    cfg: Dict[str, Any],
    exclude_held_out: bool = True,
) -> List[str]:
    """
    KG 에 넣을 논문 목록을 설정에 따라 고른다.

    1) `kg.language_filter` 로 본문 언어를 거른다 (data_ko 에 영어 논문이 섞여 있음).
    2) `kg.max_papers` 로 개수를 제한한다 (0 = 전체).
    3) `exclude_held_out` 이면 평가 전용 논문을 제외한다 — 다만 평가셋을 만들려면
       held-out 논문도 KG 에 있어야 하므로, KG 구축 자체는 보통 전체로 하고
       **샘플링 단계**에서 수집 풀만 쓰는 편이 낫다(기본 KG 구축 경로는 False 로 호출).
    """
    kg_cfg = cfg.get("kg", {})
    lang = kg_cfg.get("language_filter") or None
    papers = bridge.list_papers(language=lang)

    if exclude_held_out:
        try:
            from .eval_ko import PaperSplit

            split = PaperSplit(
                cfg.get("paths", {}).get(
                    "splits_json", "dpo_collector/outputs/splits.json"
                )
            )
            if split.is_initialized:
                held = set(split.held_out_papers)
                papers = [p for p in papers if p not in held]
        except Exception as e:
            logger.warning(f"[KGBridge] held-out 제외 실패(무시): {e}")

    max_papers = int(kg_cfg.get("max_papers", 0) or 0)
    return papers[:max_papers] if max_papers > 0 else papers


# ─── CLI: KG 구축 / 샘플링 점검 ────────────────────────────────────────────

def _main() -> int:  # pragma: no cover - 수동 검증용
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="data_ko KG 구축 및 질문 근거 경로 샘플링 점검"
    )
    ap.add_argument("--config", default="dpo_collector/config_dpo.yaml")
    ap.add_argument("--rebuild", action="store_true", help="저장된 KG 를 무시하고 재구축")
    ap.add_argument("--max-papers", type=int, default=None,
                    help="구축할 논문 수 (미지정 시 config 의 kg.max_papers)")
    ap.add_argument("--sample", type=int, default=5, help="출력할 샘플 경로 수")
    ap.add_argument("--paper", default=None, help="특정 논문 ID 로 제한")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    kg_cfg = cfg.get("kg", {})
    sampling = kg_cfg.get("sampling", {})

    bridge = from_config(args.config)

    max_papers = args.max_papers
    if max_papers is None:
        max_papers = int(kg_cfg.get("max_papers", 0))
    papers = bridge.list_papers()
    if max_papers > 0:
        papers = papers[:max_papers]

    if args.rebuild:
        bridge.build(paper_ids=papers, save=True)
    else:
        bridge.build_or_load(paper_ids=papers)

    print("\n=== KG 통계 ===")
    print(json.dumps(bridge.stats(), ensure_ascii=False, indent=2))

    if args.sample > 0:
        print(f"\n=== 샘플 경로 {args.sample}개 ===")
        paths = bridge.sample_paths(
            n=args.sample,
            paper_id=args.paper,
            max_hops=int(sampling.get("max_hops", 2)),
            prefer_visual=bool(sampling.get("prefer_visual", True)),
            require_image=bool(sampling.get("require_image", True)),
        )
        for i, p in enumerate(paths, 1):
            print(f"\n--- {i}. {p.summary()} ---")
            print(f"  논문   : {p.paper_id}")
            print(f"  이미지 : {p.image_paths}")
            print(f"  gold   : {p.gold_answer[:120]}")
            print("  컨텍스트:")
            for line in p.context.splitlines():
                print(f"    {line[:180]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
