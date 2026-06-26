# benchmark

`control-server-detection` FastAPI 서비스의 송장 OCR+LLM 목적지 추출이 실제 이미지에서
잘 도는지 확인하는 벤치 도구다.

- `waybill`: `POST /v1/waybill/scan` OCR + Qwen/LLM 판단

벤치는 내부 함수를 직접 부르지 않고, 이미 실행 중인 서비스에 HTTP 요청을 보낸다.
그래서 운영 때 쓰는 provider, PaddleOCR, Qwen 모델, GPU/CPU 설정을 그대로 잰다.

## 실행

터미널 1:

```bash
cd ~/control-server-detection
./run.sh
```

터미널 2:

```bash
cd ~/control-server-detection
python3 benchmark/run.py
```

입력을 생략하면 기본으로 현재 `resized_images` 스냅샷에서 만든 `current`
preset을 쓴다. 먼저 삭제된 파일과 non-waybill 제외를 반영한 manifest를 만든다.

```bash
python3 benchmark/gt.py build
python3 benchmark/run.py --dataset current
```

이미지 파일, 디렉터리, glob을 직접 줄 수도 있다.

```bash
python3 benchmark/run.py benchmark/images
python3 benchmark/run.py "benchmark/images/*.jpg"
python3 benchmark/run.py /data/waybill --recursive --limit 20
```

결과는 기본적으로 `benchmark/runs/<timestamp>/`에 생긴다.

- `manifest.json`: 서비스 health, 입력 이미지, 옵션
- `results.jsonl`: 이미지 x 모드별 전체 응답
- `failures.jsonl`: HTTP 실패 또는 정답 검증 실패만 모은 파일. 각 row의
  `failure_analysis`에 `low_confidence`, `ocr_miss_expected_room`,
  `destination_mismatch` 같은 원인이 들어간다.
- `summary.json`: 집계용 JSON
- `summary.md`: 사람이 빠르게 보는 요약

## 기존 waybill_ocr 데이터셋

사용 가능한 preset:

```bash
python3 benchmark/run.py --list-datasets
```

기본 preset은 `current`다. 현재 로컬 파일 기준으로:

```bash
python3 benchmark/gt.py build
python3 benchmark/run.py --dataset current
```

`current`는 현재 남아 있는 개별 `resized_images`만 사용하고, 수동 annotation에서
`exclude`로 표시한 non-waybill 이미지는 manifest에서 뺀다. 라벨이 확실한 샘플만
pass/fail 평가되고, `no_destination_detail`, `low_confidence`, `unreadable`은
서비스 응답만 기록된다.

검증 라벨이 있는 샘플만 strict하게 확인하려면:

```bash
python3 benchmark/run.py --dataset current_verified
```

예전 라벨 eval 82장을 태우려면:

```bash
python3 benchmark/run.py --dataset labeled82
```

QC 제외 1장까지 포함한 ground_truth 83장을 태우려면:

```bash
python3 benchmark/run.py --dataset groundtruth83
```

전체 235장 640x480 이미지셋을 태우되 정답 있는 row만 검증하려면:

```bash
python3 benchmark/run.py --dataset full235
```

box-scene 198장 이미지셋을 태우되 라벨 있는 row만 검증하려면:

```bash
python3 benchmark/run.py --dataset full198
```

옛 smoke manifest 5장을 쓰려면:

```bash
python3 benchmark/run.py --dataset smoke5
```

정답 파일과 이미지가 실제로 맞물리는지만 빠르게 확인하려면:

```bash
python3 benchmark/gt.py build
python3 benchmark/check_dataset.py --dataset current_verified --strict
```

GT를 한 장씩 직접 확인하려면 정적 리뷰어를 만든다.

```bash
python3 benchmark/review.py
```

기본 생성 위치:

```text
~/data/benchmarks/waybill_ocr/run_full_640x480/review/index.html
```

브라우저에서 열면 `current_manifest.json` 기준으로 현재 남아 있는 이미지를 순서대로
볼 수 있다. 수정한 라벨은 브라우저에서 `review_corrections.jsonl`로 export된다.

원격 브라우저에서 바로 GT 파일까지 수정하려면 editable 리뷰 서버를 쓴다.

```bash
python3 benchmark/review_server.py --host 0.0.0.0 --port 8790
```

접속 경로:

```text
http://<tailscale-ip>:8790/review/index.html
```

이 모드에서 `저장`을 누르면 `benchmark/ground_truth/current_annotations.jsonl`이
갱신되고, `current_manifest.json`과 리뷰 HTML도 다시 생성된다.

서비스까지 포함한 smoke test는:

```bash
benchmark/test.sh
```

기본값은 `current_verified` 중 1장만 `waybill`로 태우고, HTTP 실패나 정답
mismatch가 있으면 non-zero exit code로 끝난다. 오래 걸려도 현재 검증 라벨 전체를
다 보고 싶으면:

```bash
CONTROL_SERVER_DETECTION_BENCH_LIMIT=0 benchmark/test.sh
```

작은 송장 글자를 더 적극적으로 읽는 OCR 변형까지 포함하려면:

```bash
CONTROL_SERVER_DETECTION_BENCH_LIMIT=0 \
CONTROL_SERVER_DETECTION_BENCH_FULL_IMAGE_VARIANTS=1 \
CONTROL_SERVER_DETECTION_BENCH_CROP_VARIANTS=1 \
benchmark/test.sh
```

실패 원인을 더 정확히 보려면 OCR debug까지 같이 저장한다.

```bash
python3 benchmark/run.py --dataset current_verified \
  --ocr-full-image-variants \
  --ocr-crop-variants \
  --include-debug
```

`--include-debug`가 켜져 있으면 실패 row에서 GT 방번호가 OCR text 안에 있었는지도
`failure_analysis.details.ocr_contains_expected_room`으로 표시된다. 없으면 실제 OCR이
방번호를 못 읽은 케이스라 `ocr_miss_expected_room`으로 분류되고, 있으면 후보 선택이나
confidence 문제를 따로 볼 수 있다. 기본 low-confidence 기준은 `0.75`이며 필요하면
`--low-confidence-threshold` 또는
`CONTROL_SERVER_DETECTION_BENCH_LOW_CONFIDENCE_THRESHOLD`로 조절한다.

이미 켜진 서비스를 쓰고 싶으면 `./run.sh`를 먼저 띄운 뒤 그대로 실행하면 된다.
서비스가 꺼져 있으면 `benchmark/test.sh`가 기본으로 `./run.sh`를 background에서
올리고 끝날 때 내린다.

다른 manifest나 이미지 디렉터리도 직접 지정할 수 있다.

```bash
python3 benchmark/run.py \
  --dataset ~/data/benchmarks/waybill_ocr/run_box_only_640x480/resized_images \
  --modes waybill
```

## Hugging Face에서 내려받아 바로 eval

가장 간단한 방식은 `benchmark/run.py`에 `hf:<repo_id>`를 직접 주는 것이다.

```bash
python3 benchmark/run.py \
  --dataset hf:Fnhid/indory-waybill-ocr-640x480 \
  --ocr-full-image-variants \
  --ocr-crop-variants \
  --include-debug
```

이 명령은 내부적으로 `huggingface_hub.snapshot_download()`를 호출해서
`benchmark/datasets/hf/` 아래에 dataset snapshot을 받고, 그 안의 `manifest.json`을
평가한다. 특정 revision을 고정하려면 `@revision`을 붙인다.

```bash
python3 benchmark/run.py \
  --dataset hf:Fnhid/indory-waybill-ocr-640x480@main
```

## 정답 라벨 없이 보기

이미지만 있으면 송장 목적지, `auto_accept`, `needs_manual_review`, latency를 기록한다.

```bash
python3 benchmark/run.py /path/to/images \
  --include-debug \
  --out benchmark/runs/manual_check
```

`--include-debug`를 켜면 waybill 응답에 OCR boxes, 후보, raw LLM 응답, prompt가
포함된다. 결과 파일이 커질 수 있다.

## 정답 라벨 붙이기

현재 로컬 송장 GT annotation은 여기에 둔다.

```text
benchmark/ground_truth/current_annotations.jsonl
```

이 파일을 수정한 뒤 manifest와 리뷰 목록을 다시 만들려면:

```bash
python3 benchmark/gt.py build
```

생성물:

| 파일 | 설명 |
|---|---|
| `~/data/benchmarks/waybill_ocr/run_full_640x480/current_manifest.json` | 현재 benchmark manifest |
| `~/data/benchmarks/waybill_ocr/run_full_640x480/current_gt_summary.json` | GT/제외/리뷰 count |
| `~/data/benchmarks/waybill_ocr/run_full_640x480/review_low_confidence.jsonl` | 사람이 다시 볼 low-confidence 샘플 |
| `~/data/benchmarks/waybill_ocr/run_full_640x480/review_unreadable.jsonl` | 사람이 다시 볼 unreadable 샘플 |
| `~/data/benchmarks/waybill_ocr/run_full_640x480/review_excluded_non_waybill.jsonl` | benchmark에서 제외한 non-waybill 이미지 |
| `~/data/benchmarks/waybill_ocr/run_full_640x480/review_unscored.jsonl` | 송장은 맞지만 채점용 destination detail이 없는 샘플 |

라벨은 JSONL 한 줄에 이미지 하나다. `image`는 전체 경로, 파일명, stem 중 하나로
매칭된다. 송장 벤치마크 정답은 `destination_room`, `destination_floor`,
`destination_dong`이다.

기존 benchmark manifest나 CSV도 직접 받을 수 있다.

- `selected_manifest.json`
- `benchmark_manifest.json`
- `evaluation_rows.csv`
- `eval_records.csv`

```jsonl
{"image":"sample_001.jpg","destination_floor":"5F","destination_room":"528-1호","auto_accept":true,"needs_manual_review":false}
{"image":"sample_002.jpg","destination_contains":"601호","destination_floor":"6F","destination_room":"601호"}
```

실행:

```bash
python3 benchmark/run.py /path/to/images \
  --expected-jsonl benchmark/expected.jsonl \
  --fail-on-error
```

라벨 필드:

| 필드 | 모드 | 설명 |
|---|---|---|
| `destination_floor` | `waybill` | 예: `5F`, `5`, `F5` |
| `destination_room` | `waybill` | 예: `528-1호`, `528-1` |
| `destination_dong` | `waybill` | 동 정보가 필요한 경우 |
| `destination` | `waybill` | 최종 destination 문자열에 포함되어야 하는 값 |
| `destination_contains` | `waybill` | `destination` 별칭 |
| `auto_accept` | `waybill` | 자동 승인 여부 |
| `needs_manual_review` | `waybill` | 수동 확인 필요 여부 |

## 옵션

서비스 URL:

```bash
python3 benchmark/run.py /path/to/images \
  --url http://127.0.0.1:8767
```

Qwen GGUF 옵션:

```bash
python3 benchmark/run.py /path/to/images \
  --judge-mode llama_cpp \
  --model-path ~/waybill_ocr_llm/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --llm-ctx 4096 \
  --max-new-tokens 128 \
  --llm-gpu-layers -1
```

Ollama/OpenAI-compatible judge:

```bash
python3 benchmark/run.py /path/to/images \
  --judge-mode ollama \
  --model qwen2.5

python3 benchmark/run.py /path/to/images \
  --judge-mode openai \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct
```

PaddleOCR/provider 옵션은 `--option KEY=VALUE`로 그대로 넘길 수 있다.

```bash
python3 benchmark/run.py /path/to/images \
  --option ocr_scales=1.0,2.0 \
  --option ocr_max_side=1280
```

송장 benchmark의 작은 글자 OCR 변형은 전용 플래그로도 켤 수 있다.

```bash
python3 benchmark/run.py --dataset current_verified \
  --ocr-full-image-variants \
  --ocr-crop-variants
```

## 판정 기준

송장 벤치마크는 OCR 결과를 송장 목적지 판단 prompt로 넘기는 전체 경로를 검증한다.
여기서 목적지 방/층/동이 기대값과 맞는지, 필요하면 `auto_accept`,
`needs_manual_review`까지 따로 본다.

## HF 데이터셋으로 옮길 때

나중에 Hugging Face에 올릴 때도 runner 쪽 contract는 단순하다.

- 이미지 파일들
- `selected_manifest.json` 또는 `benchmark_manifest.json`
- 각 sample의 `ground_truth.destination_room`, `destination_floor`, `destination_dong`

HF에서 받은 디렉터리에 manifest가 있으면 그대로 지정하면 된다.

```bash
python3 benchmark/run.py --dataset /path/to/hf_snapshot/selected_manifest.json
```
