# 한국어 멀티모달 DPO 수집기

한국어 과학기술 문서(논문 PDF 파싱 결과)에서 **지식 그래프를 만들고, 그 근거로 질문을 생성하고,
VLM 후보 답변 중 더 나은 것을 골라 DPO 선호 데이터를 모으는** 웹 UI입니다.
페어가 쌓이면 **DPO LoRA 학습 → 평가 → 어댑터 교체**까지 같은 화면에서 이어집니다.

```
문서 파싱 결과 → 지식 그래프 → 질문 생성 → 후보 답변 N개
                                              ↓
                              사람이 chosen / rejected 선택
                                              ↓
                        DPO 페어 저장 → 임계치 도달 → LoRA 학습
                                              ↓
                          고정 평가셋 평가 → 어댑터 교체 → 반복
```

**7개 탭** 🕸️ KG 구축 · ✍️ 수집 · 🎯 학습 · 💬 추론 · ⚖️ 모델 비교 · 📦 내보내기 · ⚙️ 설정

---

## 요구 사항

| 항목 | 최소 | 비고 |
|---|---|---|
| Python | 3.10+ | 3.11에서 검증 |
| GPU | VRAM 24GB+ | 4B 모델 추론 기준. **학습은 40GB+ 권장** |
| 디스크 | 20GB+ | 모델 가중치 + KG JSON |
| OS | Linux | 파일 락에 `fcntl` 사용 (Windows는 락 없이 동작) |

---

## 빠른 시작

### 1. 설치

```bash
git clone <이 저장소 URL>
cd korean-dpo-collector

python -m venv .venv && source .venv/bin/activate

# PyTorch 를 먼저 (CUDA 버전에 맞게 — https://pytorch.org 참고)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### 2. 데이터 배치

`data_ko/` 아래에 **논문 폴더별로** 파싱 결과를 둡니다.

```
data_ko/
├── 논문_제목_또는_ID_1/
│   ├── parsing_paddle.json
│   └── images/
│       ├── img_in_image_box_138_766_338_1017.jpg
│       └── img_in_table_box_110_989_525_1301.jpg
└── 논문_제목_또는_ID_2/
    └── ...
```

`parsing_paddle.json` 형식 (PaddleOCR 레이아웃 파싱 결과):

```json
{
  "document_path": "...",
  "total_pages": 8,
  "pages": [
    {
      "page_number": 1,
      "title_elements":  [{"text": "제목", "confidence": 0.98}],
      "author_elements": [],
      "text_elements":   [{"text": "본문 …", "confidence": 0.87}],
      "visual_elements": [{"image_path": "images/img_in_image_box_...jpg",
                           "caption": "그림 1. 시스템 구조"}],
      "table_elements":  [{"image_path": "images/img_in_table_box_...jpg",
                           "caption": "표 1. 실험 결과",
                           "html": "<table>…</table>"}]
    }
  ]
}
```

> 이미지 파일명의 `box_{x1}_{y1}_{x2}_{y2}` 좌표는 선택입니다.
> 데이터가 없으면 `data_ko/` 폴더만 만들어 두고 UI에서 안내에 따라 진행하세요.

### 3. 설정

`dpo_collector/config_dpo.yaml` 에서 최소 세 가지만 확인하면 됩니다.

```yaml
paths:
  data_ko_root: "data_ko"                 # 2번에서 배치한 경로

model:
  name: "OpenGVLab/InternVL3_5-4B-HF"     # HF 모델명 — 자유 교체
  device_map: "auto"                      # ⚠️ 단일 GPU 환경이면 "cuda:0" 으로 (아래 참고)
```

모델은 **HuggingFace 이름만 바꾸면** 됩니다 (`Qwen/Qwen2.5-VL-7B-Instruct` 등).
계열은 `config.json` 으로 자동 판별하며, 판별이 애매하면 `model.backend` 를 `hf` 또는
`internvl` 로 지정하라는 오류가 납니다. 나머지 항목은 UI ⚙️ 설정 탭에서 편집할 수 있습니다.

> #### ⚠️ 단일 GPU 환경이면 `device_map` 을 반드시 바꾸세요
>
> 기본값 `"auto"` 는 **모델을 여러 GPU에 쪼개 얹는** 설정입니다(accelerate).
> GPU가 하나뿐이거나 멀티 GPU를 쓸 수 없는 환경에서 그대로 두면 레이어가
> 엉뚱한 디바이스에 배치되어 아래 같은 오류가 납니다.
>
> ```
> CUDA error: device-side assert triggered
> Expected all tensors to be on the same device
> ```
>
> `config_dpo.yaml` 의 **두 곳**을 모두 바꾸세요 — 답변 생성 모델과 질문 생성 모델입니다.
>
> ```yaml
> model:
>   device_map: "cuda:0"        # ← "auto" 에서 변경
>
> question_gen:
>   device_map: "cuda:0"        # ← 여기도 함께
> ```
>
> 특정 GPU를 쓰려면 `"cuda:1"` 처럼 번호를 지정하거나, 실행할 때
> `CUDA_VISIBLE_DEVICES=1 python scripts/run_dpo_collector.py` 로 넘겨도 됩니다.
> `null` 로 두면 코드가 알아서 단일 디바이스에 올립니다(`cuda` 없으면 CPU).
>
> 멀티 GPU 서버라면 `"auto"` 가 맞습니다 — 4B 모델도 한 장에 다 안 들어갈 때 자동 분산됩니다.

### 4. 실행

```bash
python scripts/run_dpo_collector.py --port 7860
```

브라우저에서 `http://<서버주소>:7860` 접속. 원격 서버라면 SSH 터널을 쓰세요.

```bash
ssh -L 7860:localhost:7860 <user>@<server>
```

### 5. 수집 시작

1. **🕸️ KG 구축** 탭 → `KG 구축 / 로드` 클릭 (첫 실행은 논문 언어 판정 때문에 조금 걸립니다)
2. **✍️ 수집** 탭 → `🎲 질문 생성` → 후보 질문 선택 → `💬 답변 생성` → 👍/👎 선택 → `💾 페어 저장`
3. 페어가 임계치(기본 150건)에 도달하면 **🎯 학습** 탭에서 학습이 가능해집니다

---

## 평가셋 (권장)

학습 후 성능 변화를 보려면 **고정 평가셋**이 필요합니다. 없으면 학습은 되지만 지표가 기록되지 않습니다.

```bash
# 1) 논문을 평가 전용 / 수집용으로 분할하고 동결
python -m dpo_collector.eval_ko split

# 2) 평가 문항 후보 생성 (평가 전용 논문에서만)
python -m dpo_collector.eval_ko draft --n 120

# 3) outputs/eval_ko_draft.jsonl 을 열어 사람이 검토
#    → 확정
python -m dpo_collector.eval_ko confirm
```

**검토할 때 가장 중요한 것**: `ground_truths` 에 **허용 표현을 여러 개 추가**하세요.

```jsonc
// 이렇게 하면 의미가 맞아도 표현이 다르다는 이유로 0점이 됩니다
"ground_truths": ["캐시 사이즈 변화에 따른 히트율"]

// 이렇게 하세요
"ground_truths": ["캐시 사이즈 변화에 따른 히트율",
                  "캐시 크기에 따른 히트율",
                  "캐시 크기별 히트율 변화"]
```

분할 결과는 `outputs/splits.json` 에 **동결**되며, 평가 전용 논문은 수집 샘플링에서
구조적으로 제외됩니다. 논문을 나중에 추가해도 기존 배정은 바뀌지 않습니다.

---

## 학습

임계치에 도달하면 🎯 학습 탭의 `🚀 지금 학습` 으로 실행합니다. CLI 도 있습니다.

```bash
python -m dpo_collector.dpo_train --status    # 트리거 상태만 확인
python -m dpo_collector.dpo_train --force     # 임계치 무시하고 지금 학습
```

- policy 와 reference를 **같은 모델에서 LoRA 를 켜고 끄는 방식**으로 계산합니다 (모델 2개를 올리지 않음)
- 학습 직후 고정 평가셋으로 평가해 `outputs/training_history.json` 에 기록합니다
- 학습된 어댑터가 자동 활성화되어 이후 수집·추론에 적용됩니다

### GPU 메모리가 부족하면

DPO 는 페어당 forward 4회를 돌리고 policy 그래프 2개를 동시에 들고 있어야 합니다.
`config_dpo.yaml` 의 아래 값을 낮추세요.

```yaml
train:
  max_num_tiles: 6         # → 4 또는 1 (이미지 타일 수)
  max_answer_tokens: 256   # → 128
  gradient_checkpointing: true   # 켜 두세요
```

---

## 모델 비교 (정성 평가)

⚖️ 탭에서 **같은 질문에 학습 전(base)과 학습 후 답변을 나란히** 놓고 비교합니다.

지표만으로는 판단이 어렵습니다 — 정답이 도표 캡션이라 의미가 맞아도 표현이 다르면
점수가 깎이기 때문입니다. 사람이 직접 고른 승패를 쌓으면 "실제로 나아졌는가"를
훨씬 직접적으로 볼 수 있습니다.

- **🏆 최고 성능 자동 선택** — `training_history.json` 의 평가 지표에서 가장 좋은
  체크포인트를 고릅니다. 기준은 세 지표 평균 / ANLS / Accuracy / F1 중에서 선택하며,
  기준에 따라 다른 체크포인트가 뽑힐 수 있습니다. 특정 체크포인트를 직접 지정해도 됩니다.
- **📋 평가셋에서 문항 불러오기** — 참고 정답이 함께 표시돼 판단이 쉬워집니다.
- **판정 기록** — 어느 쪽이 나은지 고르면 `outputs/comparisons.jsonl` 에 쌓이고,
  체크포인트별 **승률**이 집계됩니다.

> 모델을 두 번 로드하지 않습니다. 어댑터를 얹은 모델 하나에서 LoRA 를 켜고 끄면
> 학습 전/후를 모두 얻을 수 있어 **메모리가 절반**이고, 같은 가중치에서 어댑터만
> 차이나므로 비교도 더 공정합니다.

비교에는 `temperature: 0`(greedy)을 권합니다 — 샘플링 노이즈가 섞이면 두 모델의
차이인지 운인지 구분할 수 없습니다.

---

## 내보내기

📦 탭에서 공유용 포맷으로 내보냅니다.

| 포맷 | 스키마 | 비고 |
|---|---|---|
| `rlaif_v` | `{image, question, chosen, rejected}` | `image` 가 단일 필드라 **다중 이미지는 첫 장만** |
| `hf_conversational` | `{images, prompt, chosen, rejected}` | 다중 이미지 보존 |

필드 이름은 JSON 매핑으로 바꿀 수 있어 타 기관 스키마에 맞출 수 있습니다.
결과는 `datasets.load_dataset("json", data_files=...)` 로 바로 읽힙니다.

---

## 저장 형식

```
outputs/
├── kg_ko.json              # 지식 그래프
├── dpo_pairs.jsonl         # 학습용 페어 (전체 메타, append-only)
├── splits.json             # 논문 분할 (동결)
├── eval_ko.jsonl           # 확정 평가셋
├── state.json              # 활성 어댑터 / 카운터
├── training_history.json   # 체크포인트별 loss + 지표
├── comparisons.jsonl       # base vs 학습 모델 정성 비교 판정 (append-only)
├── adapters/               # 학습된 LoRA
└── export/                 # 공유용 내보내기
```

`dpo_pairs.jsonl` 은 **append-only** 입니다. 삭제는 파일을 다시 쓰지 않고 tombstone
레코드를 덧붙이며, 읽을 때 `pair_id` 별 마지막 레코드가 유효 상태가 됩니다.

---

## 알아두면 좋은 점

**모델 교체** `model.name` 만 바꾸면 됩니다. 검증된 조합:
`OpenGVLab/InternVL3_5-4B-HF`(HF 네이티브), `OpenGVLab/InternVL2_5-4B`(커스텀 코드),
`Qwen/Qwen2.5-VL-*`, `Qwen/Qwen3-VL-*`.

**InternVL 은 bfloat16 고정** 4bit/fp16 으로 올리면 InternViT 출력이 깨져 이미지 이해가
무너집니다. 설정 검증이 이를 막습니다.

**한국어 참조 보강** 본문의 "그림 1", "표 2", "식 (3)" 을 인식해 `REFERENCES` 엣지를
추가합니다. 실측에서 한국어 참조가 전체 참조의 약 40%를 차지했습니다.

**언어 필터** `kg.language_filter: "ko"` 면 본문 한글 비율로 논문 언어를 판정해
한국어 문서만 KG 에 넣습니다. 판정 결과는 캐시됩니다.

**설정 저장 시 주석 보존** `ruamel.yaml` 이 설치되어 있으면 설정 탭에서 저장해도
`config_dpo.yaml` 의 주석이 그대로 남습니다. 저장할 때마다 타임스탬프 백업이 생깁니다.

**평가 지표는 상대 비교용** 정답이 도표 캡션이라 표현이 조금만 달라도 점수가 깎입니다.
절대값보다 **체크포인트 간 변화**를 보세요. `ground_truths` 를 늘릴수록 정확해집니다.

---

## 문제 해결

| 증상 | 해결 |
|---|---|
| `CUDA error: device-side assert triggered` | 단일 GPU 환경인데 `device_map: "auto"` 인 경우가 대부분입니다. `model`·`question_gen` 의 `device_map` 을 `"cuda:0"` 으로 바꾸세요 ([위 안내](#3-설정)) |
| `Expected all tensors to be on the same device` | 위와 같은 원인입니다 |
| `백엔드 계열을 판별할 수 없습니다` | `config_dpo.yaml` 의 `model.backend` 를 `hf` 또는 `internvl` 로 지정 |
| KG 구축 후 논문이 0편 | `kg.language_filter` 를 `null` 로 두거나 데이터가 한국어인지 확인 |
| 질문 생성이 템플릿 문장으로만 나옴 | 모델 로드 실패 시 폴백입니다. 로그에서 로드 오류를 확인하세요 |
| 학습에서 페어가 계속 스킵됨 | GPU 메모리 부족입니다. 위 "GPU 메모리가 부족하면" 참고 |
| 후보 답변이 전부 동일 | temperature 를 올리세요. 2개 미만이면 페어를 만들 수 없습니다 |
| 학습 후 지표가 안 남음 | 고정 평가셋이 없습니다. 위 "평가셋" 절차를 먼저 수행하세요 |

---

## 구성

```
dpo_collector/          수집기 본체
├── backends/           모델 무관 VLM 백엔드 (hf / internvl)
├── kg_bridge.py        KG 구축·로드 + 질문 근거 경로 샘플링
├── question_gen.py     KG 근거 한국어 질문 생성 (Challenger)
├── answer_sampler.py   후보 답변 생성 (Solver)
├── store.py            페어 저장 + 공유 포맷 내보내기
├── state.py            어댑터 / 카운터 / 학습 이력
├── eval_ko.py          고정 평가셋 (분할 동결) + ANLS/Accuracy/F1
├── trigger.py          자동 학습 트리거 판정
├── dpo_train.py        DPO LoRA 학습 루프
├── config_io.py        설정 편집·검증·백업
└── app.py              Gradio UI

knowledge_graph/        KG 구축 (파서 · 3-stage 빌더 · 그래프 스토어)
models/                 평가 지표 / 평가 데이터셋
config/                 설정 로더
```

`knowledge_graph`, `models`, `config` 는 원본 연구 저장소에서 **이 UI 가 실제로 쓰는 부분만**
추려온 사본입니다. 학습 파이프라인(self-play 등)은 포함되어 있지 않습니다.

---

