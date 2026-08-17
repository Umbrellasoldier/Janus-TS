# Transition1x Fine-Tuned Non-Thinking Test Results

## Evaluation scope

- Dataset split: `test`
- Evaluated reactions: **996**
- Model: fine-tuned `Qwen/Qwen3.6-27B`
- Inference: deterministic non-thinking, beam size 10, 10 returned candidates
- Maximum generated tokens: 512
- Selected checkpoint: `final`, epoch 5, global step 2490
- Checkpoint train loss: `0.0016134435310959816`
- Checkpoint fingerprint:
  `73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3`

The test artifact records `eval_loss: null`. Evaluation loss was computed on
the validation split for checkpoint selection and was not recomputed on the
held-out test split.

## Aggregate metrics

Each metric uses its own best candidate among the first *k* ordered beams.
Connectivity and Exact show `successes / 996`. Lower edit bond is better;
all other metrics are higher-is-better.

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | 60.9438% (607/996) | 47.3896% (472/996) | 1.3353 (1/5) | 0.9574 | 0.9769 |
| @2 | 70.7831% (705/996) | 59.7390% (595/996) | 0.9428 (0/4) | 0.9699 | 0.9838 |
| @3 | 75.5020% (752/996) | 63.6546% (634/996) | 0.8133 (0/4) | 0.9748 | 0.9864 |
| @4 | 77.9116% (776/996) | 67.4699% (672/996) | 0.7219 (0/4) | 0.9774 | 0.9878 |
| @5 | 79.8193% (795/996) | 68.9759% (687/996) | 0.6667 (0/3) | 0.9799 | 0.9892 |
| @10 | **84.3373% (840/996)** | **74.1968% (739/996)** | **0.5141 (0/3)** | **0.9851** | **0.9920** |

## 95% confidence intervals

Connectivity and Exact use Wilson intervals. Mean auxiliary metrics use
reaction-level BCa bootstrap intervals with 10,000 resamples and seed 42.

| k | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---:|---:|---:|---:|---:|---:|
| @1 | 57.8774–63.9261% | 44.3046–50.4946% | 1.2289–1.4538 | 0.9530–0.9615 | 0.9744–0.9792 |
| @2 | 67.8834–73.5232% | 56.6614–62.7416% | 0.8544–1.0442 | 0.9661–0.9733 | 0.9816–0.9857 |
| @3 | 72.7364–78.0716% | 60.6203–66.5840% | 0.7299–0.9083 | 0.9712–0.9780 | 0.9843–0.9882 |
| @4 | 75.2308–80.3780% | 64.4981–70.3074% | 0.6426–0.8112 | 0.9740–0.9804 | 0.9859–0.9895 |
| @5 | 77.2143–82.1951% | 66.0347–71.7713% | 0.5914–0.7520 | 0.9766–0.9827 | 0.9874–0.9907 |
| @10 | 81.9488–86.4621% | 71.3901–76.8176% | 0.4478–0.5884 | 0.9823–0.9875 | 0.9904–0.9933 |

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
  `artifacts/runs/transition1x-b45dc9e31aa21a4e/evaluations/test.73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3.metrics.json`
- Metrics SHA256:
  `bd309dedf9001272063764cd8e60f9e8e11792e228d0493f282c3303a8c7a533`
- Predictions SHA256:
  `85b7408e848e596c6f76e3052be3ff565873f74a14ff6eb5d66630c6669df4c0`
