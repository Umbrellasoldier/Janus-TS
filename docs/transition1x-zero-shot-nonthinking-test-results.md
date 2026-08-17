# Transition1x Zero-Shot Non-Thinking Test Results

## Evaluation scope

- Dataset split: `test`
- Evaluated reactions: **996**
- Model: unmodified `Qwen/Qwen3.6-27B` (zero training updates, no adapter)
- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Representation: `MoleCode-TS/v1`
- Inference: deterministic non-thinking, beam size 10, 10 returned candidates
- Maximum generated tokens: 512
- Model dtype: BF16
- Model fingerprint:
  `3ec0e9824f3d99068944ca1e440df97259ffed3ff48a1de0163d5e0427f0e47e`

The test artifact records `eval_loss: null`; no evaluation loss is defined for
this zero-shot baseline.

## Aggregate metrics

Each metric uses its own best candidate among the first *k* ordered beams.
Connectivity and Exact show `successes / 996`. Lower edit bond is better;
all other metrics are higher-is-better.

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | 23.8956% (238/996) | 9.1365% (91/996) | 17.8233 (3/92) | 0.7292 | 0.7569 |
| @2 | 26.3052% (262/996) | 11.2450% (112/996) | 9.5452 (3/56) | 0.8223 | 0.8542 |
| @3 | 27.0080% (269/996) | 11.8474% (118/996) | 6.1988 (3/46) | 0.8615 | 0.8952 |
| @4 | 27.1084% (270/996) | 12.3494% (123/996) | 4.9227 (2/8) | 0.8827 | 0.9182 |
| @5 | 27.3092% (272/996) | 12.8514% (128/996) | 3.8715 (2/7) | 0.8973 | 0.9338 |
| @10 | **27.7108% (276/996)** | **13.7550% (137/996)** | **2.5371 (2/6)** | **0.9156** | **0.9529** |

## 95% confidence intervals

Connectivity and Exact use Wilson intervals. Mean auxiliary metrics use
reaction-level BCa bootstrap intervals with 10,000 resamples and seed 42.

| k | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---:|---:|---:|---:|---:|---:|
| @1 | 21.3507–26.6411% | 7.5007–11.0864% | 15.8594–19.9952 | 0.7045–0.7522 | 0.7313–0.7806 |
| @2 | 23.6656–29.1269% | 9.4300–13.3577% | 8.2679–10.9979 | 0.8035–0.8394 | 0.8349–0.8714 |
| @3 | 24.3428–29.8499% | 9.9855–14.0025% | 5.3573–7.2018 | 0.8458–0.8753 | 0.8792–0.9089 |
| @4 | 24.4397–29.9531% | 10.4496–14.5385% | 4.2189–5.8032 | 0.8698–0.8938 | 0.9052–0.9290 |
| @5 | 24.6333–30.1595% | 10.9149–15.0734% | 3.3428–4.5843 | 0.8865–0.9061 | 0.9232–0.9421 |
| @10 | 25.0209–30.5720% | 11.7548–16.0337% | 2.3745–2.8183 | 0.9094–0.9206 | 0.9476–0.9564 |

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
  `artifacts/runs/transition1x-b45dc9e31aa21a4e/evaluations/test.3ec0e9824f3d99068944ca1e440df97259ffed3ff48a1de0163d5e0427f0e47e.metrics.json`
- Metrics SHA256:
  `ee627278405832dc0174d1a777752ea46a56c57f5e7713a5fc7ffe3ac6627baf`
- Predictions SHA256:
  `2031d67f6643681c12e29deaee0d7b5b7c65a143ce8efbc3bc16b3ff15942dde`
