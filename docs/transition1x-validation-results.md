# Transition1x Validation Results

## Run identity

- Split: `val`
- Reactions: 994
- Model: fine-tuned `Qwen/Qwen3.6-27B`, non-thinking, beam 10
- Selected checkpoint: `final`, epoch 5, global step 2490
- Checkpoint train loss: `0.0016134435310959816`
- Checkpoint fingerprint:
  `73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3`
- Run fingerprint:
  `b45dc9e31aa21a4e715de30ac458988ebd83c492d4d5c5cde1cb13ef22a70f64`
- Data fingerprint:
  `cbc77bf825f7580ac584089273db5c322d7fa58bc9b14e697553023537d61e7f`
- Evaluation loss: **0.020286142190791397**

The train-loss minimum and final checkpoint were identical, so validation
contained one deduplicated checkpoint candidate.

## Aggregate metrics

Each metric uses its independent oracle over the first *k* raw beams. Binary
metrics show `successes / 994`. `edit bond` is lower-is-better and includes
its reaction-level P50/P95; all other metrics are higher-is-better.

| k | Connectivity | Exact | edit bond ↓ (P50/P95) | Edge IoU ↑ | Edge F1 ↑ |
|---:|---:|---:|---:|---:|---:|
| @1 | 63.1791% (628/994) | 46.7807% (465/994) | 1.4044 (1/5) | 0.9608 | 0.9780 |
| @2 | 71.7304% (713/994) | 56.8410% (565/994) | 1.0342 (0/4) | 0.9715 | 0.9843 |
| @3 | 75.5533% (751/994) | 61.8712% (615/994) | 0.7767 (0/4) | 0.9776 | 0.9881 |
| @4 | 77.2636% (768/994) | 64.7887% (644/994) | 0.6952 (0/3) | 0.9796 | 0.9892 |
| @5 | 78.4708% (780/994) | 66.3984% (660/994) | 0.6449 (0/3) | 0.9813 | 0.9901 |
| @10 | **83.5010% (830/994)** | **72.9376% (725/994)** | **0.4859 (0/3)** | **0.9865** | **0.9929** |

## 95% confidence intervals

Connectivity and Exact use Wilson intervals. The mean auxiliary metrics use
reaction-level BCa bootstrap intervals with 10,000 resamples and seed 42.

| k | Connectivity | Exact | edit bond | Edge IoU | Edge F1 |
|---:|---:|---:|---:|---:|---:|
| @1 | 60.1353–66.1214% | 43.6972–49.8890% | 1.2485–1.7850 | 0.9555–0.9650 | 0.9732–0.9807 |
| @2 | 68.8515–74.4420% | 53.7415–59.8880% | 0.9095–1.4165 | 0.9671–0.9749 | 0.9804–0.9863 |
| @3 | 72.7866–78.1233% | 58.8116–64.8395% | 0.6982–0.8612 | 0.9745–0.9803 | 0.9864–0.9896 |
| @4 | 74.5560–79.7613% | 61.7677–67.6959% | 0.6197–0.7746 | 0.9768–0.9822 | 0.9876–0.9906 |
| @5 | 75.8086–80.9138% | 63.4038–69.2667% | 0.5757–0.7213 | 0.9786–0.9838 | 0.9886–0.9914 |
| @10 | 81.0654–85.6786% | 70.0913–75.6073% | 0.4286–0.5503 | 0.9843–0.9885 | 0.9917–0.9940 |

## Interpretation and provenance

Checkpoint selection uses the @10 metrics lexicographically in this order:
Connectivity, Exact, edit bond, Edge IoU, Edge F1, then evaluation loss.
Connectivity ignores bond order; Exact requires both edge pairs and bond
orders to match. A wrong bond order on an existing edge counts as one edit.

Source metrics:
`artifacts/runs/transition1x-b45dc9e31aa21a4e/evaluations/val.73e680ca5691cd311e7ecb097456f80356c3050d2548c7e641e761d4b6256bc3.metrics.json`

Source SHA256:
`4569a7d172c2321e4e1ee480d85c99b68b207f51883d0f7a08cee4cc20f9246b`
