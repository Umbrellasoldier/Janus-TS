# Janus-TS

`Janus-TS` fine-tunes a pinned Qwen text model to predict a discretized
transition-state molecular graph from reactant and product molecular graphs.

The first locked experiment is Transition1x with the `MoleCode-TS/v1` text
representation. Raw chemistry datasets and the historical `chemformer_bo`
checkout are read-only inputs; generated data and run artifacts live below
`artifacts/` and are intentionally excluded from Git.

## Commands

```bash
uv sync --frozen
uv run janus-ts --help
uv run janus-ts data preprocess --config configs/transition1x.yaml
uv run janus-ts data audit --config configs/transition1x.yaml
uv run janus-ts train smoke --config configs/transition1x.yaml
uv run janus-ts train run --config configs/transition1x.yaml
uv run janus-ts evaluate run --config configs/transition1x.yaml
uv run janus-ts infer thinking --help
```

`train smoke` performs the CPU/data/snapshot/native audits, the two-rank CUDA
kernel and BF16 dtype gates, and immutable PiSSA preparation. Each GPU phase
takes the dedicated gu30 lock so two Janus-TS launchers cannot compete with
each other. Other users' GPU jobs may coexist and are never signalled or
killed. Training geometry is fixed at microbatch 1/GPU and accumulation 8;
there is no candidate search or duplicate 27B pretraining run.

`train run` repeats or verifies those lightweight content-addressed gates,
resumes the greatest valid local/durable manifest step, and starts the formal
five-epoch training directly.
It retains the exact-step train-loss minimum checkpoint and compares only that
checkpoint and the final checkpoint on validation (one run if they coincide).
Each candidate's rank-64 portable adapter is checked against its rank-32
resume form before formal validation. After the @10 lexicographic winner is
locked, only that fine-tuned checkpoint is used for the complete
996-reaction test split. The four modes run in this order: raw-Qwen non-thinking, raw-Qwen
thinking, selected-checkpoint non-thinking, selected-checkpoint thinking.
Non-thinking uses deterministic beam 10 and reports @1/@2/@3/@4/@5/@10.
Each complete-test thinking call returns ten independently sampled candidates
in one batch, retains every raw reasoning trace, and scores the answers after
`</think>` at @1/@2/@3/@4/@5/@10. Thinking remains ineligible for checkpoint
selection. The final artifact compares all four modes.
Checkpoint and stage ordering use fingerprints, explicit global steps, and
completion markers only—never filesystem modification time. Trainer events
are appended and fsynced to the run's `logs/train.jsonl` every configured ten
steps in addition to TensorBoard output.

## Optional subset thinking exploration

In addition to the automatic complete-test evaluations, the `infer thinking`
command is an explicitly non-formal subset inspection tool. Supply
the processed dataset and the selected durable evaluation checkpoint directly; the
selection proof contains content fingerprints but does not contain enough path
information to resolve either directory safely.

```bash
uv run janus-ts infer thinking \
  --config configs/transition1x.yaml \
  --processed-path /absolute/path/to/processed/transition1x/FINGERPRINT \
  --checkpoint-dir /absolute/path/to/selected/evaluation-checkpoint \
  --output-dir artifacts/exploration \
  --split val \
  --reaction-id rxn0001 \
  --reaction-id rxn0002
```

The launcher uses the project's absolute `torchrun`, takes the same exclusive
gu30 GPU lock, and repeats the host-readiness checks before and after the
two-rank job. Without `--reaction-id` it inspects only the first
two records by default; `--limit` changes that finite selection. An odd number
of selected records adds one deterministic dummy `generate` call so both
ZeRO-3 ranks execute the same number of collectives, but the dummy is never
written as a prediction.

The frozen exploratory profile enables thinking and sampling with beam 1,
temperature 0.6, top-p 0.95, top-k 20, min-p 0, repetition penalty 1, and up
to 8192 new tokens. The subset CLI retains its configured one-sample default;
the automatic complete-test runner records and applies a ten-sample evaluation
override without changing the training run identity. `presence_penalty=0` is
recorded in provenance only and is not passed to Transformers. Results are content-addressed below
`OUTPUT_DIR/thinking-exploratory/` and are marked `formal_eligible=false`.
They do not compute formal metrics, alter checkpoint selection or run state,
or acquire the one-time formal test lease.
