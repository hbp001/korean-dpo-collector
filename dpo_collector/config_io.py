"""
dpo_collector/config_io.py
---------------------------
`config_dpo.yaml` 읽기/쓰기 + 검증 (UI 설정 탭이 쓰는 백엔드).

주석을 지키는 것이 이 모듈의 핵심이다. `config_dpo.yaml` 의 주석에는
"왜 이 값인지"(예: InternVL 4bit 금지, 학습 타일 수를 낮춘 이유, ANLS 특성)가 적혀 있어
설정을 한 번 저장했다고 날아가면 안 된다. 그래서 `ruamel.yaml` 의 round-trip 모드로
**기존 문서를 제자리 수정**하고, ruamel 이 없을 때만 pyyaml 로 폴백한다(주석 소실 경고).

저장은 항상 `.bak` 백업 → 임시 파일 → rename 순서로 원자적으로 수행한다.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 설정에 반드시 있어야 하는 키 (없으면 앱이 뜨지 않는다)
REQUIRED_KEYS = (
    ("paths", "data_ko_root"),
    ("paths", "pairs_jsonl"),
    ("model", "name"),
)

#: UI 에서 고를 수 있는 백엔드
BACKEND_CHOICES = ("auto", "hf", "internvl")
#: 지원 dtype (InternVL 은 4bit 금지 — §1-5)
DTYPE_CHOICES = ("bfloat16", "float16", "float32", "auto")
#: export 포맷
EXPORT_FORMAT_CHOICES = ("rlaif_v", "hf_conversational")
#: 평가 지표 (models.eval_metrics.compute_all_metrics 와 정합)
METRIC_CHOICES = ("anls", "accuracy", "f1")


def _has_ruamel() -> bool:
    try:
        import ruamel.yaml  # noqa: F401

        return True
    except Exception:
        return False


# ─── 읽기 ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """평범한 dict 로 읽는다 (읽기 전용 용도)."""
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_roundtrip(path: str):
    """
    주석·순서를 보존하는 형태로 읽는다.
    ruamel 이 없으면 `(dict, False)` 를 돌려준다.
    """
    if not _has_ruamel():
        return load_config(path), False
    from ruamel.yaml import YAML

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(path, "r", encoding="utf-8") as f:
        return yaml_rt.load(f), True


def read_text(path: str) -> str:
    """YAML 원문 (직접 편집 모드용)."""
    return Path(path).read_text(encoding="utf-8")


# ─── 검증 ──────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def message(self) -> str:
        parts: List[str] = []
        if self.errors:
            parts.append("**오류**\n" + "\n".join(f"- ❌ {e}" for e in self.errors))
        if self.warnings:
            parts.append("**경고**\n" + "\n".join(f"- ⚠️ {w}" for w in self.warnings))
        return "\n\n".join(parts) if parts else "✅ 검증 통과"


def validate_config(cfg: Dict[str, Any]) -> ValidationResult:
    """
    저장 전에 확인한다.

    - 오류: 저장을 막는다 (필수 키 누락, 값 범위 위반, 규칙 위반)
    - 경고: 저장은 되지만 알려준다 (경로 없음, 성능/품질 영향)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(cfg, dict):
        return ValidationResult(False, ["설정이 매핑(dict) 형태가 아닙니다."])

    for section, key in REQUIRED_KEYS:
        if not (cfg.get(section) or {}).get(key):
            errors.append(f"`{section}.{key}` 는 반드시 있어야 합니다.")

    paths = cfg.get("paths", {}) or {}
    data_root = paths.get("data_ko_root")
    if data_root and not Path(data_root).is_dir():
        errors.append(f"`paths.data_ko_root` 경로가 없습니다: {data_root}")

    # 출력 경로들은 아직 없어도 되지만, 상위 디렉토리는 만들 수 있어야 한다
    for key in ("pairs_jsonl", "state_json", "kg_json", "eval_jsonl"):
        p = paths.get(key)
        if not p:
            continue
        parent = Path(p).parent
        if parent and not parent.exists():
            warnings.append(f"`paths.{key}` 상위 폴더가 없어 저장 시 새로 만들어집니다: {parent}")

    model = cfg.get("model", {}) or {}
    backend = str(model.get("backend", "auto")).lower()
    if backend not in BACKEND_CHOICES:
        errors.append(f"`model.backend` 는 {list(BACKEND_CHOICES)} 중 하나여야 합니다: {backend}")

    dtype = str(model.get("dtype", "bfloat16")).lower()
    name = str(model.get("name", ""))
    is_internvl_custom = "internvl2" in name.lower() or backend == "internvl"
    if is_internvl_custom and dtype not in ("bfloat16", "bf16"):
        errors.append(
            "InternVL 커스텀 계열은 bfloat16 이어야 합니다 — "
            "4bit/float16 은 InternViT 출력이 깨집니다."
        )
    if model.get("load_in_4bit") or model.get("load_in_8bit"):
        errors.append("양자화(load_in_4bit/8bit)는 지원하지 않습니다 (이미지 이해 손상).")

    lang = str(cfg.get("language", "ko"))
    if not lang:
        errors.append("`language` 가 비어 있습니다.")

    train = cfg.get("train", {}) or {}
    for key, lo in (
        ("first_train_min_pairs", 1), ("retrain_every_n_pairs", 1),
        ("epochs", 1), ("gradient_accumulation_steps", 1),
    ):
        val = train.get(key)
        if val is not None and int(val) < lo:
            errors.append(f"`train.{key}` 는 {lo} 이상이어야 합니다: {val}")
    beta = train.get("dpo_beta")
    if beta is not None and not (0 < float(beta) <= 1):
        errors.append(f"`train.dpo_beta` 는 0 초과 1 이하가 적절합니다: {beta}")
    lr = train.get("learning_rate")
    if lr is not None and not (0 < float(lr) < 1e-2):
        warnings.append(f"`train.learning_rate` 가 비정상적으로 큽니다: {lr}")

    ev = cfg.get("eval", {}) or {}
    metrics = ev.get("metrics") or []
    unknown = [m for m in metrics if m not in METRIC_CHOICES]
    if unknown:
        errors.append(
            f"`eval.metrics` 에 지원하지 않는 지표가 있습니다: {unknown} "
            f"(가능: {list(METRIC_CHOICES)})"
        )
    if metrics and not unknown and len(metrics) == 0:
        warnings.append("`eval.metrics` 가 비어 학습 후 지표가 기록되지 않습니다.")

    exp = cfg.get("export", {}) or {}
    fmt = str(exp.get("format", "rlaif_v"))
    if fmt not in EXPORT_FORMAT_CHOICES:
        errors.append(
            f"`export.format` 은 {list(EXPORT_FORMAT_CHOICES)} 중 하나여야 합니다: {fmt}"
        )
    fmap = exp.get("field_map")
    if fmap is not None and not isinstance(fmap, dict):
        errors.append("`export.field_map` 은 매핑(dict) 형태여야 합니다.")

    kg = cfg.get("kg", {}) or {}
    if int(kg.get("max_papers", 0) or 0) < 0:
        errors.append("`kg.max_papers` 는 0 이상이어야 합니다 (0 = 전체).")
    if (kg.get("overrides", {}) or {}).get("semantic", {}).get("enabled"):
        warnings.append(
            "semantic 단계를 켰습니다 — 전체 코퍼스(약 10만 노드) 임베딩에 "
            "상당한 시간과 메모리가 듭니다."
        )

    return ValidationResult(not errors, errors, warnings)


# ─── 쓰기 ──────────────────────────────────────────────────────────────────

def _backup(path: Path) -> Optional[Path]:
    """저장 전 백업. 되돌릴 수 있어야 안심하고 설정을 만질 수 있다."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, dest)
    return dest


def save_text(path: str, text: str) -> Tuple[bool, str]:
    """
    YAML 원문을 그대로 저장한다 (직접 편집 모드).
    사용자가 쓴 주석·서식이 100% 보존된다.
    """
    import yaml

    p = Path(path)
    try:
        parsed = yaml.safe_load(text)
    except Exception as e:
        return False, f"❌ YAML 파싱 실패: {e}"
    if not isinstance(parsed, dict):
        return False, "❌ 최상위가 매핑(dict) 형태여야 합니다."

    v = validate_config(parsed)
    if not v.ok:
        return False, f"저장하지 않았습니다.\n\n{v.message()}"

    bak = _backup(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    msg = f"✅ 저장 완료 → `{p}`"
    if bak:
        msg += f"\n\n백업: `{bak.name}`"
    if v.warnings:
        msg += f"\n\n{v.message()}"
    return True, msg


def apply_updates(path: str, updates: Dict[Tuple[str, ...], Any]) -> Tuple[bool, str]:
    """
    폼에서 온 값들을 기존 YAML 에 **제자리 반영**한다 (주석 보존).

    Args:
        updates: {("model","name"): "…", ("train","epochs"): 2, …}
                 값이 None 이면 그 키는 건드리지 않는다.
    """
    p = Path(path)
    doc, roundtrip = load_roundtrip(str(p))

    for keys, value in updates.items():
        if value is None:
            continue
        node = doc
        for k in keys[:-1]:
            if k not in node or node[k] is None:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    # 검증은 평범한 dict 로 (ruamel 객체도 매핑처럼 동작한다)
    v = validate_config(_to_plain(doc))
    if not v.ok:
        return False, f"저장하지 않았습니다.\n\n{v.message()}"

    bak = _backup(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        if roundtrip:
            from ruamel.yaml import YAML

            yaml_rt = YAML()
            yaml_rt.preserve_quotes = True
            with open(tmp, "w", encoding="utf-8") as f:
                yaml_rt.dump(doc, f)
        else:
            import yaml

            tmp.write_text(
                yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        tmp.replace(p)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f"❌ 저장 실패: {e}"

    msg = f"✅ 저장 완료 → `{p}`"
    if bak:
        msg += f"\n\n백업: `{bak.name}`"
    if not roundtrip:
        msg += "\n\n⚠️ `ruamel.yaml` 이 없어 **주석이 사라졌습니다**. 백업에서 확인하세요."
    if v.warnings:
        msg += f"\n\n{v.message()}"
    return True, msg


def _to_plain(node: Any) -> Any:
    """ruamel 노드를 평범한 dict/list 로 변환 (검증용)."""
    if isinstance(node, dict):
        return {k: _to_plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_to_plain(v) for v in node]
    return node


def list_backups(path: str, limit: int = 10) -> List[str]:
    """최근 백업 파일명 (최신순)."""
    p = Path(path)
    files = sorted(
        p.parent.glob(p.name + ".*.bak"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return [f.name for f in files[:limit]]


def restore_backup(path: str, backup_name: str) -> Tuple[bool, str]:
    """백업으로 되돌린다 (되돌리기 전 현재 상태도 백업)."""
    p = Path(path)
    src = p.parent / backup_name
    if not src.is_file():
        return False, f"❌ 백업 파일이 없습니다: {backup_name}"
    try:
        text = src.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"❌ 백업을 읽지 못했습니다: {e}"
    return save_text(str(p), text)
