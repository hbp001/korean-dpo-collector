#!/usr/bin/env python
"""
scripts/run_dpo_collector.py
-----------------------------
한국어 멀티모달 DPO 수집 UI 실행 엔트리포인트.

    python scripts/run_dpo_collector.py
    python scripts/run_dpo_collector.py --port 7861 --config dpo_collector/config_dpo.yaml

프로젝트 루트를 sys.path 에 넣어 `dpo_collector` / `knowledge_graph` 등을
어디서 실행하든 import 할 수 있게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dpo_collector.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
