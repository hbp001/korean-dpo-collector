"""
models/eval_metrics.py
-----------------------
평가 메트릭 계산 모듈

지원 메트릭:
  - ANLS       (Average Normalized Levenshtein Similarity) — DocVQA 공식 메트릭
  - MCQ Accuracy — 객관식 정답률 (ScienceQA / MMMU 방식)
  - F1 (token-level) — 오픈엔드 답변
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ─── ANLS ─────────────────────────────────────────────────────────────────

def _normalized_levenshtein(s1: str, s2: str) -> float:
    """0~1 사이 정규화 편집거리. 0=동일, 1=완전히 다름."""
    s1, s2 = s1.strip().lower(), s2.strip().lower()
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[j] = min(prev[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[len2] / max(len1, len2)


def anls_score(
    prediction: str,
    ground_truths: List[str],
    threshold: float = 0.5,
) -> float:
    """
    ANLS (Average Normalized Levenshtein Similarity).
    복수 정답 중 가장 유사한 것과 비교합니다.
    """
    if not ground_truths:
        return 0.0
    scores = []
    for gt in ground_truths:
        nl = _normalized_levenshtein(prediction, gt)
        scores.append(0.0 if nl > threshold else 1.0 - nl)
    return max(scores)


def batch_anls(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """배치 ANLS 계산."""
    assert len(predictions) == len(ground_truths_list)
    scores = [
        anls_score(p, gts, threshold)
        for p, gts in zip(predictions, ground_truths_list)
    ]
    return {
        "anls":     round(sum(scores) / len(scores), 4) if scores else 0.0,
        "per_sample": scores,
    }


# ─── MCQ Accuracy ─────────────────────────────────────────────────────────

def _extract_mcq_answer(prediction: str, n_choices: int = 8) -> str:
    """
    모델 응답에서 MCQ 정답 레터를 추출합니다.

    n_choices: 해당 문항의 선택지 개수 (기본 8 = A~H 허용).
      ScienceQA에는 선택지가 5개 이상인 문항이 존재하므로 A~D 고정은 오답 처리 원인이 됨.

    다양한 응답 형식을 처리합니다:
      - "A" / "A)" / "(A)" / "The answer is A" / "정답: A" / "Answer: A. xxx"
    """
    text = prediction.strip()
    hi   = "ABCDEFGH"[: max(1, min(n_choices, 8))]
    lo   = hi.lower()
    cls  = f"[{hi}{lo}]"

    # 1순위: "Answer: A" 또는 "정답: A" 패턴
    m = re.search(rf"(?:answer|정답)\s*[:\-]?\s*\(?({cls})\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2순위: 문장 시작의 단독 레터 "A." 또는 "A)"
    m = re.match(rf"^\s*\(?({cls})\)?[.):\s]", text)
    if m:
        return m.group(1).upper()

    # 3순위: 괄호 안 레터 "(A)"
    m = re.search(rf"\(({cls})\)", text)
    if m:
        return m.group(1).upper()

    # 4순위: 응답 전체가 레터 하나
    if len(text) == 1 and text.upper() in hi:
        return text.upper()

    # 5순위: 텍스트 어디서든 단독 레터
    m = re.search(rf"\b({cls})\b", text)
    if m:
        return m.group(1).upper()

    # 6순위: 첫 글자
    if text and text[0].upper() in hi:
        return text[0].upper()

    return ""


def _extract_final_answer(prediction: str, answer_type: str = "free_form") -> str:
    """
    장문 응답에서 최종 답변을 추출합니다.
    MathVista / ChartQA 공식 평가의 answer extraction 단계에 해당합니다.

    우선순위:
      1. \\boxed{} LaTeX 표현
      2. "The answer is X", "Therefore X" 패턴
      3. "ANSWER: X" 패턴 (ChartQA 권장 형식)
      4. MCQ이면 A/B/C/D 레터 추출
      5. 마지막 줄
    """
    if not prediction:
        return ""

    text = prediction.strip()

    # 1순위: LaTeX boxed answer
    boxed = re.search(r"\\boxed\{([^\}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # 2순위: "ANSWER: X" (ChartQA 권장 형식)
    ans_tag = re.search(r"ANSWER\s*:\s*(.+)", text, re.IGNORECASE)
    if ans_tag:
        return ans_tag.group(1).strip()

    # 3순위: "The answer is X", "Therefore X" 패턴
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*([^\n\.]+)",
        r"therefore[,\s]+(?:the\s+answer\s+is\s+)?([^\n\.]+)",
        r"(?:so|thus)[,\s]+(?:the\s+answer\s+is\s+)?([^\n\.]+)",
        r"(?:결론|정답|답)\s*[:\-]\s*([^\n\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    # 4순위: MCQ이면 레터 추출
    if answer_type == "multi_choice":
        letter = _extract_mcq_answer(text)
        if letter:
            return letter

    # 5순위: 마지막 비어있지 않은 줄
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return text[:100]


def mcq_accuracy(prediction: str, correct_answer: str, n_choices: int = 8) -> float:
    """
    MCQ 단일 샘플 정답 여부.
    correct_answer: "A" ~ "H"
    """
    pred_letter    = _extract_mcq_answer(prediction, n_choices)
    correct_letter = correct_answer.strip().upper()[:1]
    return float(pred_letter == correct_letter) if pred_letter else 0.0


# ─── 자체 crossdoc 벤치마크용 (run_benchmark_eval.py와 동일 로직) ─────────

_CROSSDOC_STOP = {
    "that", "this", "with", "from", "have", "been", "they", "their", "also",
    "which", "were", "would", "could", "into", "more", "some", "when", "then",
    "than", "used", "both", "using", "between", "through", "while", "these",
    "those", "about", "after", "before",
}


def _cd_keywords(text: str) -> set:
    return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            if w not in _CROSSDOC_STOP}


def _cd_nums(text: str) -> set:
    return set(re.findall(r"\b\d+\.?\d*\b", text))


def crossdoc_hop_acc(prediction: str, ground_truths: List[str]) -> Tuple[float, str]:
    """KGVerifier.verify()와 동일한 5단계 매칭. (score, method) 반환."""
    from difflib import SequenceMatcher

    if not prediction or not ground_truths:
        return 0.0, "empty"
    best, method = 0.0, "no_match"
    pred = re.sub(r"\s+", " ", prediction.lower()).strip()
    for gt in ground_truths:
        gold = re.sub(r"\s+", " ", (gt or "").lower()).strip()
        if not gold:
            continue
        if gold in pred or pred in gold:
            return 1.0, "contain"
        gn, pn = _cd_nums(gold), _cd_nums(pred)
        if gn:
            if gn == pn:
                return 1.0, "num_exact"
            if gn & pn and best < 0.7:
                best, method = 0.7, "num_partial"
        gk, pk = _cd_keywords(gold), _cd_keywords(pred)
        if gk:
            ov = len(gk & pk) / len(gk)
            if ov >= 0.7 and best < 0.8:
                best, method = 0.8, "kw_high"
            elif ov >= 0.4 and best < 0.5:
                best, method = 0.5, "kw_mid"
        sim = SequenceMatcher(None, pred, gold).ratio()
        if sim >= 0.5 and sim > best:
            best, method = sim, "seq"
    return best, method


def batch_mcq_accuracy(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    n_choices_list: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    배치 MCQ Accuracy.
    장문 응답에서 answer extraction 후 정답 레터와 비교합니다.
    ScienceQA / MMMU 공식 평가 방식과 동일합니다.

    n_choices_list: 문항별 선택지 개수 (미지정 시 A~H 허용).
    """
    scores    = []
    extracted = []
    for i, (pred, gts) in enumerate(zip(predictions, ground_truths_list)):
        correct = gts[0] if gts else ""
        nc      = n_choices_list[i] if n_choices_list else 8
        # 장문 응답에서 최종 답 추출 (MathVista/ScienceQA 공식 방식)
        extracted_ans = _extract_final_answer(pred, answer_type="multi_choice")
        score         = mcq_accuracy(extracted_ans, correct, nc)
        scores.append(score)
        extracted.append(extracted_ans)

    n       = len(scores)
    correct = sum(scores)
    return {
        "accuracy":          round(correct / n, 4) if n else 0.0,
        "correct":           int(correct),
        "total":             n,
        "per_sample":        scores,
        "extracted_answers": extracted,
    }


# ─── 기존 Accuracy (VQA 단답형용 유지) ────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """소문자화, 구두점 제거, 공백 정규화."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    """정규화 후 정확히 일치하면 1.0. VQA 단답형용."""
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(gt) == pred_norm for gt in ground_truths))


def batch_accuracy(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    question_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    배치 Accuracy.
    MCQ → _extract_mcq_answer() 기반 정답 추출
    VQA → exact_match
    """
    scores = []
    for i, (pred, gts) in enumerate(zip(predictions, ground_truths_list)):
        qtype = question_types[i] if question_types else "VQA"
        if qtype == "MCQ" and gts:
            scores.append(mcq_accuracy(pred, gts[0]))
        else:
            scores.append(exact_match(pred, gts))
    return {
        "accuracy":   round(sum(scores) / len(scores), 4) if scores else 0.0,
        "per_sample": scores,
    }


# ─── Relaxed Accuracy (ChartQA / MathVista free_form) ────────────────────

def _try_parse_number(text: str) -> Optional[float]:
    """문자열에서 숫자 추출 시도."""
    import re
    text = text.strip().replace(",", "").replace("%", "")
    m = re.search(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def relaxed_accuracy(
    prediction: str,
    ground_truths: List[str],
    tolerance: float = 0.05,
) -> float:
    """
    Relaxed Accuracy (ChartQA / MathVista 공식 평가 방식).

    1. 장문 응답에서 최종 답 추출 (answer extraction)
    2. 수치형: 정답 ±5% 오차 허용
    3. 텍스트형: 정규화 후 완전 일치
    """
    # 장문 응답에서 최종 답 추출
    prediction = _extract_final_answer(prediction, answer_type="free_form")
    pred_norm  = _normalize_answer(prediction)

    for gt in ground_truths:
        gt_norm = _normalize_answer(gt)

        # 텍스트 완전 일치
        if pred_norm == gt_norm:
            return 1.0

        # 수치형 ±tolerance 허용
        pred_num = _try_parse_number(prediction)
        gt_num   = _try_parse_number(gt)
        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                if abs(pred_num - gt_num) <= tolerance:
                    return 1.0
            elif abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
                return 1.0

    return 0.0


def batch_relaxed_accuracy(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """배치 Relaxed Accuracy."""
    scores = [
        relaxed_accuracy(p, gts, tolerance)
        for p, gts in zip(predictions, ground_truths_list)
    ]
    n       = len(scores)
    correct = sum(scores)
    return {
        "relaxed_accuracy": round(correct / n, 4) if n else 0.0,
        "correct":          int(correct),
        "total":            n,
        "per_sample":       scores,
    }

# ─── 한국어 VQA (AI Hub 시각화 자료 질의응답) ────────────────────────────────
# 정답이 완전한 문장으로 주어진다 (예: "투자 실적은 54.1조 원입니다.").
# 공식 리더보드 지표가 공개돼 있지 않아 **아래 프로토콜을 정의해 사용**한다.
#   수치형: 정답 문장의 핵심 수치가 예측에 존재하면 정답 (콤마/단위 표기 정규화)
#   문자형: 정답의 핵심 명사구가 예측에 포함되거나(양방향), 토큰 F1 >= 0.5면 정답
# 리포트에는 이 정의를 반드시 명시할 것.

_KO_JOSA = (
    "입니다", "이에요", "예요", "습니다", "합니다", "이다", "있습니다", "있어요",
    "많습니다", "많아요", "했습니다", "됩니다", "죠", "요",
)


def _ko_normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[,\s]+", " ", t)
    t = re.sub(r"[.。!?~\"'()\[\]]", " ", t)
    return " ".join(t.split())


def _ko_numbers(text: str) -> List[str]:
    """콤마 제거 후 숫자 추출 ('1,234.5' → '1234.5')."""
    t = text.replace(",", "")
    return re.findall(r"-?\d+(?:\.\d+)?", t)


def _ko_content_tokens(text: str) -> set:
    """조사·서술어를 제거한 내용어 토큰."""
    toks = set()
    for w in _ko_normalize(text).split():
        for suf in _KO_JOSA:
            if w.endswith(suf) and len(w) > len(suf):
                w = w[: -len(suf)]
                break
        w = re.sub(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만)$", "", w)
        if len(w) >= 2:
            toks.add(w)
    return toks


def ko_vqa_accuracy(prediction: str, gold: str, a_type: str = "") -> float:
    """AI Hub 시각화 자료 질의응답 채점 (0.0 | 1.0)."""
    if not prediction or not gold:
        return 0.0
    pred_n, gold_n = _ko_normalize(prediction), _ko_normalize(gold)

    g_nums = _ko_numbers(gold)
    p_nums = _ko_numbers(prediction)

    # 수치형: 정답 수치가 예측에 있으면 정답
    if a_type == "수치형" or (g_nums and not a_type):
        if g_nums:
            if any(g in p_nums for g in g_nums):
                return 1.0
            # 소수점 표기 차이 허용 (54.1 vs 54.10)
            for g in g_nums:
                for p in p_nums:
                    try:
                        if abs(float(g) - float(p)) < 1e-6:
                            return 1.0
                    except ValueError:
                        continue
        # 수치형인데 숫자가 없으면 문자 비교로 폴백

    # 문자형: 포함 관계 (양방향) 또는 내용어 F1
    if gold_n and (gold_n in pred_n or pred_n in gold_n):
        return 1.0
    g_tok, p_tok = _ko_content_tokens(gold), _ko_content_tokens(prediction)
    if g_tok and p_tok:
        if g_tok <= p_tok:                     # 정답 내용어를 모두 포함
            return 1.0
        common = len(g_tok & p_tok)
        if common:
            prec, rec = common / len(p_tok), common / len(g_tok)
            if 2 * prec * rec / (prec + rec) >= 0.5:
                return 1.0
    return 0.0


def batch_ko_vqa_accuracy(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    a_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    scores = []
    for i, (p, gts) in enumerate(zip(predictions, ground_truths_list)):
        at = a_types[i] if a_types else ""
        scores.append(max((ko_vqa_accuracy(p, g, at) for g in gts), default=0.0))
    n = len(scores)
    return {
        "accuracy":   round(sum(scores) / n, 4) if n else 0.0,
        "correct":    int(sum(scores)),
        "total":      n,
        "per_sample": scores,
    }


def token_f1(prediction: str, ground_truths: List[str]) -> float:
    """토큰 수준 F1. 복수 정답 중 최고값 반환."""
    def _f1(pred: str, gt: str) -> float:
        pred_tokens = _normalize_answer(pred).split()
        gt_tokens   = _normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    if not ground_truths:
        return 0.0
    return max(_f1(prediction, gt) for gt in ground_truths)


def batch_f1(
    predictions: List[str],
    ground_truths_list: List[List[str]],
) -> Dict[str, float]:
    scores = [
        token_f1(p, gts)
        for p, gts in zip(predictions, ground_truths_list)
    ]
    return {
        "f1":         round(sum(scores) / len(scores), 4) if scores else 0.0,
        "per_sample": scores,
    }


# ─── 통합 메트릭 계산 ─────────────────────────────────────────────────────

def compute_all_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    question_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    ANLS, Accuracy, F1을 한 번에 계산합니다.
    """
    anls_result = batch_anls(predictions, ground_truths_list)
    acc_result  = batch_accuracy(predictions, ground_truths_list, question_types)
    f1_result   = batch_f1(predictions, ground_truths_list)

    return {
        "n_samples": len(predictions),
        "anls":      anls_result["anls"],
        "accuracy":  acc_result["accuracy"],
        "f1":        f1_result["f1"],
        "per_sample": [
            {
                "anls":     anls_result["per_sample"][i],
                "accuracy": acc_result["per_sample"][i],
                "f1":       f1_result["per_sample"][i],
            }
            for i in range(len(predictions))
        ],
    }
