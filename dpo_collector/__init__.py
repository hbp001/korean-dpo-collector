"""
dpo_collector
-------------
한국어 과학기술 문서 기반 멀티모달 DPO(선호) 데이터 수집 UI + 파이프라인.

이 패키지는 코어(`knowledge_graph/`, `models/`, `self_play/`, `config/config.yaml`)를
**수정하지 않고 import로만 재사용**합니다. 신규 코드는 전부 이 패키지 안에 격리됩니다.

빌드 순서(CLAUDE_dpo.md §10) 중 현재 구현된 범위:
  1. backends/    — 모델 무관 VLM 백엔드 추상화 (로드 / 추론)
  2. kg_bridge.py — data_ko KG 구축·로드 + 질문 근거 경로 샘플링
  3. store.py     — DPO 페어 저장(append-only) + 공유 포맷 export
     state.py     — 활성 어댑터 / 페어 카운터 / 학습 이력
  4. question_gen.py   — KG 근거 한국어 후보 질문 (Challenger)
     answer_sampler.py — 활성 모델의 한국어 후보 답변 (Solver)
  5. app.py            — Gradio 수집 UI (KG 구축 탭 + 수집 탭)
  6. eval_ko.py        — 고정 한국어 평가셋(논문 split 동결) + ANLS/Accuracy/F1
  7. trigger.py        — 자동 학습 트리거 판정 (수량 + 유형 편향 가드레일)
     dpo_train.py      — 수동 DPO LoRA 루프 (adapter 토글로 policy/reference) + 평가 로깅
  8. app.py            — 학습 / 추론 / 내보내기 탭 추가
  9. config_io.py      — config_dpo.yaml 편집·검증·백업 (주석 보존) + 설정 탭

→ CLAUDE_dpo.md §10 빌드 순서 1~9 완료.
"""
__all__ = [
    "backends", "kg_bridge", "store", "state",
    "question_gen", "answer_sampler", "app", "eval_ko",
    "trigger", "dpo_train", "config_io",
]
