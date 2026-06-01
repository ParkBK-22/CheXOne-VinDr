# VinDr-CXR Visual-Faithfulness Perturbation Pipeline — 조건 A / B (사용설명서)

병변을 정상 조직으로 치환한 **반사실(counterfactual) 이미지**를 만들어, 모델이 실제로
이미지를 보고 답하는지(**visual grounding**) 아니면 텍스트 prior 로 찍는지를, **답 변화율
(flip rate)** 로 검증하는 파이프라인입니다.

> ⚠️ **데이터 취급**: VinDr-CXR 은 PhysioNet credentialed(DUA). 생성물도 병변/GT 정보를
> 담으므로 **자격 보유 팀원 내부 공유만**. GitHub/HF 등 공개 업로드 금지.

> 📦 전체 실험은 세 개 set(A·B·crop)이며, **이 패키지는 A·B 담당분**입니다.
> **crop margin** set 은 효근의 별도 패키지(여기 미포함). 셋 다 같은 272 단일병변 기반이고,
> **추론은 각 팀원이 본인 모델로** 수행합니다.

---

## 1. 두 조건 (이 패키지)

| 조건 | 대상 | 조작 | grounded 모델 기대 | 의미 |
|---|---|---|---|---|
| **A** lesion-removal | 단일 병변(bbox 1개) **272** | 병변 bbox → **다른 정상 이미지(cross-donor)** 의 동일영역 패치 | Q1 `Yes→No`, Q3 `1→0`, Q5 `→No specific disease` | 병변 지우면 답 바뀜 = **진짜 보고 있음** |
| **B** normal-manip (★음성대조) | 정상(No finding) **272** | A 와 같은 크기/위치 분포 영역 → 다른 정상 donor 패치 | Q1 `No` **유지** | flip(No→Yes) = **조작 흔적 자체가 abnormal 로 오인되는 비율** = A 해석의 baseline |

핵심: **A 의 Q1 flip(Yes→No)** 이 grounding 신호, **B 의 Q1 flip(No→Yes)** 이 그 신호를
오염시키는 artifact 의 양. `Δ = A flip − B flip` 이 grounding 의 순효과.

---

## 2. 질문은 Q1 / Q3 / Q5 만

병변을 **제거한 뒤에도 '정답 선지'가 존재**하는 유형만 flip 지표로 씁니다.

| Q | 제거 후 정답 | 사용 |
|---|---|---|
| **Q1** abnormality | **No** (선지 O) | ✅ 핵심 |
| **Q3** count | **0** (선지 O) | ✅ |
| **Q5** disease | **No specific disease** (선지 O) | ✅ (A 만; 정상엔 Q5 없음) |
| ~~Q2 finding~~ / ~~Q4 location~~ | "no abnormality" 선지가 **없음** | ✖ 제거 후 ill-posed → 제외 |

> Set1 의 Q2/Q4 는 *병변이 존재한다*를 전제로 만들어져 "정상" 선지가 없습니다. 병변을
> 지운 뒤엔 정답이 옵션에 없어 flip-to-correct 측정이 불가 → 이번 실험에서 제외.
> (필요하면 선지 추가본으로 따로 재추론 가능하나, Q2+선지추가 ≈ Q1 과 중복이라 보류.)
> 조건 B 는 정상 이미지라 애초에 Q1(+Q3=0) 만 존재.

---

## 3. 요구사항 & 입력

```bash
pip install numpy pydicom pillow         # build_perturbations.py
pip install matplotlib                    # plot_sanity.py (검수용, 선택)
```
- **DICOM 디렉토리**: `{image_id}.dicom` (VinDr test 3000장)
- **annotations CSV**: `annotations_test.csv`
- **qa_set1_options.jsonl**: 질문 필터용 (kit `build_vqa.py` 산출)

---

## 4. 실행

### 4-1. 전체 생성 (A 272 + B 272 + 질문필터)
```bash
python build_perturbations.py \
    --dicom-dir /본인경로/vindr-cxr/1.0.0/test \
    --ann-csv   /본인경로/vindr-cxr/1.0.0/annotations/annotations_test.csv \
    --qa-jsonl  /본인경로/qa_set1_options.jsonl \
    --out-dir   ./perturb_set \
    --save-original
```
> ★ **친구별로 `--dicom-dir`/`--ann-csv`/`--qa-jsonl` 만 본인 경로로** 바꾸면 동일 결과.
> 모든 random 은 (seed, image_id) 결정론 → **모두 동일한 perturbed PNG 재현**.

### 4-2. sanity 검수 (각 10장)
```bash
python build_perturbations.py ... --out-dir ./perturb_sanity --n-a 10 --n-b 10 --save-original
python plot_sanity.py --out-dir ./perturb_sanity --qa <qa_set1_options.jsonl>
# → perturb_sanity/sanity_A.png, sanity_B.png 로 눈 검수
```

---

## 5. 산출물

```
<out-dir>/
  A/<image_id>.png                  조건 A perturbed (1024 RGB, 모델 입력용)
  B/<image_id>.png                  조건 B perturbed
  original/<image_id>.png           windowed 원본 (A·B 대상 모두; --save-original)
  questions_A.jsonl                 A 대상 Q1/Q3/Q5 (qa_set1 레코드 그대로 필터)
  questions_B.jsonl                 B 대상 Q1/Q3
  perturbation_manifest.jsonl       {image_id,condition,donor_id,bbox/region,used_box,orig_hw,seed}
```
> `questions_*.jsonl` 은 **qa_set1 스키마 그대로**(qid/image_id/qtype/question/answer/options) →
> crop 파이프라인의 `single_bbox_questions.jsonl` 과 동일 형식이라 **같은 추론 스크립트로 처리 가능**.

---

## 6. 추론 & flip 집계 (모델별 각자)

생성 PNG 는 **이미 windowing+1024+RGB 끝난 모델 입력용**. DICOM 다시 windowing 하지 말 것.

1. **추론 (3-way)**: `questions_A.jsonl` 의 각 질문을
   ① `original/<image_id>.png` ② `A/<image_id>.png` 에 추론. (B 는 `questions_B` × original/B)
   - 옵션 셔플은 각자 모델 Set1 러너와 **동일 규칙** 유지.
   - 예측을 아래 **표준 스키마**로 저장(한 파일에 original/A/B 모두):
     ```json
     {"qid":123,"image_id":"...","qtype":"Q1_abnormality","condition":"original|A|B","gt":"Yes","pred":"Yes"}
     ```
     `pred` 는 파싱된 **옵션 텍스트**(예: "No", "0", "No specific disease").
2. **집계**:
   ```bash
   python compute_flip.py --preds preds_<model>.jsonl --out-csv flip_<model>.csv
   ```
   → (조건 × qtype) `orig_acc / flip_rate / to_expected` 표 + 핵심 요약
   (A Yes→No flip, B No→Yes flip, Δ).

---

## 7. windowing 주의 (cross-set 일관성)

- 본 A/B 파이프라인 windowing = **percentile 1–99 → 8bit → 1024 RGB** (`run_qwen`/`run_internvl`
  의 `dicom_to_pil` 과 동일, **apply_voi_lut 미적용** → 우리 Set1 baseline 과 일치).
- 효근 **crop 파이프라인**(`crop_margin/`)은 `apply_voi_lut` 를 추가 적용. 따라서 crop set 의
  `original` 과 본 A/B set 의 `original` 픽셀이 미세하게 다를 수 있음(같은 이미지여도). flip /
  accuracy 는 각 set 내부에서 original↔perturbed 로 비교하므로 **set 내부 분석엔 문제 없음**.
  cross-set 절대값 비교 시 이 차이를 감안할 것.

---

## 8. 파일

| 파일 | 역할 |
|---|---|
| `build_perturbations.py` | A/B perturbed PNG + 질문필터 + manifest 생성 |
| `compute_flip.py` | 표준 pred jsonl → flip 표/CSV |
| `plot_sanity.py` | 원본\|조작 + 질문/GT/기대답 시각 검수 |
| `README.md` | 본 문서 |
