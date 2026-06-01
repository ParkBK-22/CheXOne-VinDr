# VinDr-CXR Single-bbox VQA Pipeline (Original vs Crop margins)

## 무엇인가
**모델 하나를 정하면** VinDr-CXR test 셋에서 단일 bbox 이미지 272장을 자동 추출하고, 그 이미지들의 관련 질문(qa_set1)을 자동 필터한 뒤, **원본 + bbox 중심 crop (margin 1.0 / 1.5 / 2.0)** 4 조건에서 모델 답변을 받아 정확도를 비교하는 end-to-end 파이프라인.

목적: lesion 픽셀에 얼마나 의존하는지, 주변 context가 얼마나 필요한지를 정량 비교.

## 폴더 구조
```
vindr_bbox_crop/
├── README.md                            ← 이 파일
├── code/
│   ├── 01_prepare_dataset.py            단일 bbox 추출 + 질문 필터 + 모든 PNG 생성
│   ├── 02_run_inference.py              모델 1개로 4 조건 추론 (chexagent/medgemma 지원)
│   ├── 03_compare.py                    qtype × condition 정확도 표 + summary CSV
│   ├── crop_bboxes.py                   crop 단독 생성 (구버전, 01에 통합됨)
│   └── viz_crop_samples.py              10 샘플 시각 검증
├── output/                              파이프라인 산출물 (생성)
│   ├── single_bbox_info.csv             272 단일 bbox 정보
│   ├── single_bbox_questions.jsonl      qa_set1 중 해당 이미지 질문만
│   ├── png_original/{image_id}.png      장변 1024 (kit spec)
│   ├── png_crop/
│   │   ├── margin_1.0/{image_id}.png    bbox 정확히
│   │   ├── margin_1.5/{image_id}.png    bbox 1.5배
│   │   └── margin_2.0/{image_id}.png    bbox 2.0배
│   ├── results_{model}.jsonl            추론 raw 결과
│   ├── results_{model}.csv              CSV 형식
│   └── results_{model}_summary.csv      정확도 요약
└── samples_viz/
    └── crop_samples_viz.png             10 샘플 시각 검증
```

## 파이프라인 (모델만 정하면 됨)

### 사전 준비 (1회성)
- VinDr-CXR test DICOMs 다운로드 (S3 access point 권장, README_download 참고)
- `qa_set1_options.jsonl` 준비 (`vindr_vqa_kit/build_vqa.py` 로 생성)
- 모델 가중치 (예: CheXagent-8b, MedGemma-1.5-4b-it)

### Step 1 — 데이터 준비 (단일 bbox 추출 + crop 생성)

```bash
python code/01_prepare_dataset.py \
    --vindr_root /workspace/physionet.org/files/vindr-cxr/1.0.0 \
    --qa_jsonl   /workspace/qa_set1_options.jsonl \
    --out_dir    /workspace/pipeline_out \
    --margins    1.0 1.5 2.0 \
    --qtypes     Q1_abnormality Q2_finding Q5_disease
```

**기본 qtype = Q1, Q2, Q5만** (Q3 GT=1 trivial, Q4 crop에서 frame 손실 → 무의미).
모두 받으려면 `--qtypes Q1_abnormality Q2_finding Q3_count Q4_location Q5_disease`.

생성물 (기본 qtype 기준):
- `single_bbox_info.csv` (272 rows)
- `single_bbox_questions.jsonl` (Q1 272 + Q2 272 + Q5 68 = **612 질문**)
- `png_original/` 272 PNG (장변 1024)
- `png_crop/margin_{1.0,1.5,2.0}/` 각 272 PNG

### Step 2 — 모델 추론 (모델별로 1회 실행)

**CheXagent-8b**:
```bash
python code/02_run_inference.py \
    --questions  /workspace/pipeline_out/single_bbox_questions.jsonl \
    --png_orig   /workspace/pipeline_out/png_original \
    --crop_root  /workspace/pipeline_out/png_crop \
    --margins    1.0 1.5 2.0 \
    --model_type chexagent \
    --model_dir  /workspace/chexagent8b \
    --out_jsonl  /workspace/pipeline_out/results_chexagent.jsonl
```

**MedGemma-1.5-4b-it** (배치 64 권장):
```bash
python code/02_run_inference.py \
    --questions  /workspace/pipeline_out/single_bbox_questions.jsonl \
    --png_orig   /workspace/pipeline_out/png_original \
    --crop_root  /workspace/pipeline_out/png_crop \
    --margins    1.0 1.5 2.0 \
    --model_type medgemma \
    --model_dir  /workspace/medgemma-1.5-4b-it \
    --batch      64 \
    --out_jsonl  /workspace/pipeline_out/results_medgemma.jsonl
```

다른 모델 추가 시: `02_run_inference.py` 안에 `run_<model>(model_dir, ...)` 함수 추가 + `main()` 에 `--model_type` 옵션 분기 추가.

### Step 3 — 정확도 분석

```bash
python code/03_compare.py \
    --results /workspace/pipeline_out/results_chexagent.jsonl
```

출력: qtype × condition 정확도 격자 표 (행=qtype, 열=조건):
```
qtype              N      pred_original    pred_margin_1.0  pred_margin_1.5  pred_margin_2.0
Q1_abnormality    272      230/272(84%)     245/272(90%)     242/272(89%)     235/272(86%)
Q2_finding        272       ##/##(##%)      ##/##(##%)       ##/##(##%)       ##/##(##%)
Q5_disease         68       ##/68(##%)      ##/68(##%)       ##/68(##%)       ##/68(##%)
```
→ `results_<model>_summary.csv` 도 저장.

## Crop 규칙 (정확)

각 single-bbox 이미지의 `(x_min, y_min, x_max, y_max)`:

1. bbox 중심: `(cx, cy) = ((x_min+x_max)/2, (y_min+y_max)/2)`
2. bbox 크기: `(w, h) = (x_max - x_min, y_max - y_min)`
3. 새 crop 영역: 중심 유지, 크기 `(w*margin, h*margin)`
   - **margin=1.0**: bbox 정확히 그대로 (lesion만, context 없음)
   - **margin=1.5**: 각 변마다 25%씩 패딩 (bbox + 약간 context)
   - **margin=2.0**: 각 변마다 50%씩 패딩 (bbox + 풍성한 context)
4. 이미지 경계로 clip — bbox가 가장자리면 패딩 잘림
5. **원본 픽셀 해상도 유지** (resize 없음)

PNG 변환은 kit spec과 동일: pydicom 로드 → MONOCHROME1 시 색 반전 → **percentile 1–99 windowing** → uint8 → RGB.

## Single-bbox 272장 finding 분포

| Finding | N | 카테고리 |
|---|---|---|
| Cardiomegaly | 81 | shape |
| Aortic enlargement | 50 | shape |
| Nodule/Mass | 27 | focal |
| Pulmonary fibrosis | 24 | diffuse |
| Calcification | 20 | focal |
| Lung Opacity | 13 | focal |
| ILD | 12 | diffuse |
| Other lesion | 9 | other |
| Consolidation | 9 | focal |
| Atelectasis | 8 | focal |
| Infiltration | 6 | focal |
| Pleural effusion | 5 | focal |
| Pleural thickening | 3 | border |
| Rib fracture | 2 | other |
| Pneumothorax | 2 | border |
| Lung cavity | 1 | focal |

→ focal 89, shape 131, diffuse 36, 기타 16.

## 시각 검증

```bash
python code/viz_crop_samples.py \
    --vindr_dir /workspace/physionet.org/files/vindr-cxr/1.0.0/test \
    --bbox_csv  /workspace/pipeline_out/single_bbox_info.csv \
    --crop_dir  /workspace/pipeline_out/png_crop \
    --out_png   /workspace/pipeline_out/crop_samples_viz.png \
    --n 10
```

→ 10 row × 4 col (원본+bbox / margin 1.0 / 1.5 / 2.0).

## 실험 해석 가이드

각 condition별 정확도 변화 패턴:

| 패턴 | 의미 |
|---|---|
| margin 1.0 ≪ margin 2.0 ≪ original | 모델이 context 의존 강함 (lesion만으로 부족) |
| margin 1.0 ≈ margin 2.0 ≈ original | 모델이 lesion-grounded (crop만으로도 충분) |
| margin 1.0 > original | 모델이 distractor 무시 어려움 (작은 lesion에 집중 못함) |
| Q3/Q4 만 떨어짐 | 글로벌 정보 (count, position) 손실 — 예상되는 자연스러운 결과 |

## 출처
- VinDr-CXR v1.0.0 (PhysioNet credentialed)
- `vindr_vqa_kit/build_vqa.py` 로 생성한 `qa_set1_options.jsonl`

## crop 비교에서 qtype별 의미

| qtype | 의미 | 단일 bbox subset 수 |
|---|---|---|
| **Q2 finding** | ✅ **가장 의미 있음** — 16개 finding 중 분류, lesion만 보고 정확히 식별하는지 | 272 |
| Q1 abnormality | ⚠️ sanity check — GT 모두 Yes, 모델이 명백한 lesion 놓치면 실패 신호 | 272 (모두 Yes) |
| Q5 disease | ⚠️ secondary — 글로벌 정보 손실 정도 측정 (단 N 작음) | 68 |
| Q3 count | ❌ trivial — GT 모두 1 | 272 |
| Q4 location | ❌ frame 깨짐 — crop 중심이 항상 lesion | 272 |

**기본 파이프라인은 Q1/Q2/Q5만 (612 질문). 더 보고 싶으면 `--qtypes` 옵션으로 추가.**

## 한계 / 주의
- Q3/Q4 는 의미 적어 기본 제외. 필요 시 `--qtypes Q3_count Q4_location` 추가 가능.
- Crop은 원본 해상도 유지하므로 PNG 파일 크기 다양 (10K~수백K 픽셀). 모델 image processor가 자기 해상도로 resize.
- 단일 bbox = 1 lesion 이미지만 사용. 다중 lesion은 본 파이프라인 범위 밖.
- Q5 N=68로 작음 — accuracy 절대값보다 trend (original→crop 변화) 위주로 해석 권장.
