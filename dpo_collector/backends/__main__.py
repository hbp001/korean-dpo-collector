"""
dpo_collector/backends/__main__.py
-----------------------------------
스모크 테스트 진입점 — `python -m dpo_collector.backends ...` 로 실행.
실제 구현은 `__init__.py::_main()`.
"""
from . import _main

raise SystemExit(_main())
