# Transition1x Four-Mode Test Results

## Run status and protocol

- Workflow status: **complete**
- Test split: **996 reactions**
- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Representation: `MoleCode-TS/v1`
- Selected checkpoint: `final`, epoch 5, global step 2490
- Selected checkpoint train loss: `0.0016134435310959816`
- Validation loss: `0.020286142190791397`
- Checkpoint fingerprint:
  `73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3`

The train-loss minimum and final checkpoint were identical. Checkpoint
selection therefore evaluated one deduplicated candidate using validation
`@10` metrics in priority order: Connectivity, Exact, edit bond, Edge IoU,
Edge F1, then evaluation loss. Test loss was not recomputed.

## Four-mode comparison at @1

Binary metrics show `successes / 996`. Lower edit bond is better; all other
metrics are higher-is-better.

| Model | Inference | Connectivity | Exact | edit bond ↓ | Edge IoU ↑ | Edge F1 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Fine-tuned | non-thinking | **60.9438% (607/996)** | **47.3896% (472/996)** | **1.3353** | **0.9574** | **0.9769** |
| Fine-tuned | thinking | 28.1124% (280/996) | 17.9719% (179/996) | 43.7329 | 0.4962 | 0.5085 |
| Zero-shot | non-thinking | 23.8956% (238/996) | 9.1365% (91/996) | 17.8233 | 0.7292 | 0.7569 |
| Zero-shot | thinking | 15.0602% (150/996) | 6.3253% (63/996) | 42.5422 | 0.4837 | 0.5035 |

Non-thinking `@1` is the first result from a beam-10 search. Thinking `@1`
is one stochastic sample, so cross-mode rows do not represent equal inference
cost. Fine-tuned versus zero-shot comparisons within a mode are the cleaner
measure of training effect.

## Non-thinking results

Each metric independently uses its best candidate among the first *k* ordered
beams. Edit bond includes reaction-level P50/P95.

### Fine-tuned

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | 60.9438% (607/996) | 47.3896% (472/996) | 1.3353 (1/5) | 0.9574 | 0.9769 |
| @2 | 70.7831% (705/996) | 59.7390% (595/996) | 0.9428 (0/4) | 0.9699 | 0.9838 |
| @3 | 75.5020% (752/996) | 63.6546% (634/996) | 0.8133 (0/4) | 0.9748 | 0.9864 |
| @4 | 77.9116% (776/996) | 67.4699% (672/996) | 0.7219 (0/4) | 0.9774 | 0.9878 |
| @5 | 79.8193% (795/996) | 68.9759% (687/996) | 0.6667 (0/3) | 0.9799 | 0.9892 |
| @10 | **84.3373% (840/996)** | **74.1968% (739/996)** | **0.5141 (0/3)** | **0.9851** | **0.9920** |

### Zero-shot

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | 23.8956% (238/996) | 9.1365% (91/996) | 17.8233 (3/92) | 0.7292 | 0.7569 |
| @2 | 26.3052% (262/996) | 11.2450% (112/996) | 9.5452 (3/56) | 0.8223 | 0.8542 |
| @3 | 27.0080% (269/996) | 11.8474% (118/996) | 6.1988 (3/46) | 0.8615 | 0.8952 |
| @4 | 27.1084% (270/996) | 12.3494% (123/996) | 4.9227 (2/8) | 0.8827 | 0.9182 |
| @5 | 27.3092% (272/996) | 12.8514% (128/996) | 3.8715 (2/7) | 0.8973 | 0.9338 |
| @10 | **27.7108% (276/996)** | **13.7550% (137/996)** | **2.5371 (2/6)** | **0.9156** | **0.9529** |

## Thinking results and output validity

Thinking used one sampled candidate per reaction with temperature 0.6,
top-p 0.95, top-k 20, and at most 8,192 new tokens.

| Model | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ | Parse valid |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned | 28.1124% (280/996) | 17.9719% (179/996) | 43.7329 (6/137) | 0.4962 | 0.5085 | 52.3092% (521/996) |
| Zero-shot | 15.0602% (150/996) | 6.3253% (63/996) | 42.5422 (6/137) | 0.4837 | 0.5035 | 52.7108% (525/996) |

### Thinking 95% confidence intervals

Connectivity and Exact use Wilson intervals. Auxiliary metrics use
reaction-level BCa bootstrap intervals with 10,000 resamples and seed 42.

| Model | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---|---:|---:|---:|---:|---:|
| Fine-tuned | 25.4088–30.9843% | 15.7119–20.4780% | 40.7063–46.9036 | 0.4660–0.5259 | 0.4780–0.5386 |
| Zero-shot | 12.9735–17.4155% | 4.9750–8.0112% | 39.6212–45.6561 | 0.4554–0.5127 | 0.4741–0.5334 |

Invalid thinking outputs remain in the 996-reaction denominator. Fine-tuned
thinking had 475 parse-invalid outputs; zero-shot thinking had 471. In both
runs, 460 outputs specifically lacked a closing `</think>` marker.

## Fine-tuning effect

Values are fine-tuned minus zero-shot under the same inference mode. Negative
edit-bond delta is an improvement.

| Protocol | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---|---:|---:|---:|---:|---:|
| non-thinking @1 | +37.0482 pp | +38.2530 pp | -16.4880 | +0.2282 | +0.2201 |
| non-thinking @10 | +56.6265 pp | +60.4418 pp | -2.0231 | +0.0695 | +0.0391 |
| thinking @1 | +13.0522 pp | +11.6466 pp | +1.1908 | +0.0125 | +0.0049 |

Fine-tuning strongly improves the two primary metrics in both modes. The best
overall result is fine-tuned non-thinking at `@10`. Under the tested thinking
protocol, fine-tuning improves Connectivity and Exact, but the high invalid
output rate limits the auxiliary metrics and edit bond is slightly worse.

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
- Four-mode comparison:
  `artifacts/runs/transition1x-b45dc9e31aa21a4e/evaluations/inference-mode-comparison.json`
- Comparison SHA256:
  `2f73534e3c71b17dc3747521b66b60d9a665fb66e95092fb41fc4bc6ceb7f371`

| Result | Metrics SHA256 | Predictions SHA256 |
|---|---|---|
| Fine-tuned non-thinking | `bd309dedf9001272063764cd8e60f9e8e11792e228d0493f282c3303a8c7a533` | `85b7408e848e596c6f76e3052be3ff565873f74a14ff6eb5d66630c6669df4c0` |
| Fine-tuned thinking | `36e8e5e267bc070dc500d0e6fd9911eef17c940f83da484a4f5483d5125fe052` | `23b8ab4fbd15a1a2e0e899648f64dc228ca7ebe79e68f16b89ae3bf07cd61571` |
| Zero-shot non-thinking | `ee627278405832dc0174d1a777752ea46a56c57f5e7713a5fc7ffe3ac6627baf` | `2031d67f6643681c12e29deaee0d7b5b7c65a143ce8efbc3bc16b3ff15942dde` |
| Zero-shot thinking | `fcbc41f5714120bdf463f044ad161dedbf74f3dd41e2e8bb3019b31cfe111abd` | `c933e22d166f1fef48de769095634a7fbd993e4b0bb2a6fb19d0fa20ef8ee586` |

Detailed confidence intervals for the other modes are available in the
mode-specific reports in `docs/`.
