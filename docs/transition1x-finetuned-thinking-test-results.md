# Transition1x Fine-Tuned Thinking Test Results

## Evaluation scope

- Dataset split: `test`
- Evaluated reactions: **996**
- Model: fine-tuned `Qwen/Qwen3.6-27B`
- Inference: thinking enabled, one sampled candidate per reaction (`@1`)
- Generation: temperature 0.6, top-p 0.95, top-k 20, maximum 8,192 new tokens
- Runtime: vLLM 0.25.1, BF16, pipeline parallelism 2, request concurrency 8
- Selected checkpoint: `final`, epoch 5, global step 2490
- Checkpoint fingerprint:
  `73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3`

This test run is diagnostic and does not affect checkpoint selection. Evaluation
loss was not recomputed on the held-out test split.

## Aggregate metrics

Lower edit bond is better; all other metrics are higher-is-better.

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | **28.1124% (280/996)** | **17.9719% (179/996)** | **43.7329 (6/137)** | **0.4962** | **0.5085** |

## 95% confidence intervals

Connectivity and Exact use Wilson intervals. Mean auxiliary metrics use
reaction-level BCa bootstrap intervals with 10,000 resamples and seed 42.

| k | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---:|---:|---:|---:|---:|---:|
| @1 | 25.4088–30.9843% | 15.7119–20.4780% | 40.7063–46.9036 | 0.4660–0.5259 | 0.4780–0.5386 |

## Output validity

- Parse-valid candidates: **521/996 (52.3092%)**
- Parse-invalid candidates: **475/996 (47.6908%)**
- Candidates specifically missing a closing `</think>` marker: **460/996 (46.1847%)**

Invalid candidates receive no special exclusion: they remain in the 996-reaction
denominator and are penalized by the reported metrics.

## Metric definitions

- **Connectivity:** 1 only when the complete predicted edge set is correct;
  bond order is ignored.
- **Exact:** 1 only when every predicted edge and bond order is correct.
- **edit bond:** number of bond edits including bond-order changes; a wrong
  bond order on an existing edge counts once.
- **Edge IoU / F1:** compare edge sets without considering bond order.

## Provenance

- Run fingerprint:
  `b45dc9e31aa21a4e715de30ac458988ebd83c492d4d5c5cde1cb13ef22a70f64`
- Data fingerprint:
  `cbc77bf825f7580ac584089273db5c322d7fa58bc9b14e697553023537d61e7f`
- Metrics source:
  `artifacts/runs/transition1x-b45dc9e31aa21a4e/evaluations/thinking-evaluations/fine-tuned/fe94597b2c4a64af1eac831f495e42365b532ea3bb983718fcf765fb5b4329d4/metrics.json`
- Metrics SHA256:
  `36e8e5e267bc070dc500d0e6fd9911eef17c940f83da484a4f5483d5125fe052`
- Scored predictions SHA256:
  `23b8ab4fbd15a1a2e0e899648f64dc228ca7ebe79e68f16b89ae3bf07cd61571`
- Raw predictions SHA256:
  `217e67ef0e6420b94afc2f6a808ded79785aba27a900a8c537cda8f6213b0cd6`
