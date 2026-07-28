# Transition1x experiment contract

This document records the decisions frozen during the `grill with docs`
review.  A change to a frozen field creates a new experiment fingerprint; it
must never silently alter an in-progress run.

## Scope

- Train two independent models eventually: Transition1x first, RGD1 second.
- This repository currently implements the Transition1x experiment only.
- Base model: `Qwen/Qwen3.6-27B` at commit
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, text decoder only.
- Formal training runs on both RTX 4090 GPUs on `gu30` with seed 42.
- The only historical comparison is the exact Chemformer/ReactionT5v2 v24
  checkpoint 413820; no heuristic or additional model baseline is run.

## Data

Raw split directory:

```text
/mnt/sto3/shared_files/lgh to cxy/GeoDiff/training_data/reaction_data/
train_addprop_wbo_raw_energy_kcalmol_split_8_1_1_rxn9233_test
```

The raw split is authoritative for membership and TS Wiberg bond orders.
Expected raw counts are train/val/test = 7,968/996/996.  TS WBO is discretized
exactly as `GeoDiff/utils/datasets.py:604`, after dropping values below 0.1:

```text
w < 0.10 -> absent
w < 0.75 -> 0.5
w < 1.25 -> 1.0
w < 1.75 -> 1.5
w < 2.25 -> 2.0
w < 2.75 -> 2.5
otherwise -> 3.0
```

Endpoint recovery uses the immutable hybrid mapped-SMILES file copied from
cu07 into the Janus-TS cache.  It is authoritative only for reactant/product
mapped SMILES; it does not define splits, energies, coordinates, TS labels, or
any other field.

```text
final hybrid JSONL       5843fc724f4a0dff9b624617c383ed81532191e1de8bfaf21570e26ac9cf54ad
raw topology B           bae8812dfde0b85f3e3eb24f465c37355aad2c9eeb3b5aff0d5bf39c497f46ab
source A                 102c5bfe236763196ff11975f9591a52e4fefb5bb0578105fa32a2fcaad85cb9
base recovery script     cbf3bf1d52cbd848c29f239242a656b7165f612a9f6269015055bc798fb43345
pre-retry JSONL          61d31c1a5073bdd5ac1f8394b8052b1866d596311130f397679aaa601a69320d
four-reaction retry      0e3d4e3553244ae178137186ccee43bddfac53528ef3882ad2e6859ed973bc93
retry output             109ee38dee6fe9c993bd1d59884f23128bdc81fb4c7a921b107dc61ed54d8d92
retry merge script       0a3cb3199bf6b85aab193eabbfcc08d5bb6866402298fe138301a10668c2dfcf
```

Every endpoint is reparsed with strict RDKit sanitization and explicit H
retention. Atom maps must be exactly 1..N and atomic number at map `i` must
match `atom_types[i-1]`. R/P atoms, attributes, components, bonds, E/Z and R/S
are regenerated from this validated molecule. In particular, the malformed
pickle endpoint edges must not be retained for the 476 recovered endpoints.

Sixteen unresolved reactions are quarantined as whole examples:

```text
rxn0951 rxn1323 rxn1324 rxn1434 rxn1889 rxn3034 rxn3760 rxn4187
rxn4998 rxn5062 rxn5063 rxn5065 rxn5570 rxn7147 rxn7475 rxn9958
```

Expected retained counts are 7,954/994/996.  There must be no reaction-ID or
full directional, mapless canonical `R>>P` signature overlap across splits.
Reusable individual endpoints are allowed.  The processed Arrow dataset is
disk-backed and content-addressed; no truncation is permitted.

## MoleCode-TS/v1

- One shared, explicit-H, zero-based atom table is used by R, P, and TS.
- R and P have independent component, sparse charge/radical/R/S atom, and
  bond-order/E/Z edge sections, followed by atom- and edge-change sections.
- The TS target is only a canonical `<TS_EDGES>` block. Any atom pair is
  permitted, so hydrogen-transfer forming/breaking bonds are representable.
- TS stereochemistry and coordinates are deliberately not predicted.
- At training time, R and P edge lines receive a deterministic stateless
  permutation keyed by `(seed, logical epoch, reaction ID, section)`. The same
  molecule is therefore perturbed again each epoch, reproducibly across
  workers, ranks, and resume. Validation/test input and every target are
  canonical.

## Optimization

- Qwen chat template, non-thinking. Mask the system message, user message, and
  empty `<think>\n\n</think>\n\n` prefix. Supervise the target through
  `<|im_end|>`.
- Maximum full sequence length 2,048, dynamic padding to a multiple of 8, and
  hard failure instead of truncation.
- BF16 frozen base and FP32 trainable adapters. This is implemented as an
  audited two-phase DeepSpeed precision contract: native BF16 is enabled only
  while ZeRO-Init streams and partitions the residual base; before Trainer
  initializes its engine, native BF16 is disabled and DeepSpeed
  `torch_autocast` is enabled with dtype BF16 for exact `torch.nn.Linear`
  modules. This prevents DeepSpeed from recasting LoRA storage to BF16. The
  persistent LoRA ZeRO shards, gradient-accumulation buffers, AdamW master
  weights, and AdamW moments remain FP32; Linear forward/backward compute and
  LoRA communication use BF16. Both the pre-engine and post-engine dtypes are
  hard gates.
- Exact LoRA coverage: 496 projections across all 64 layers. Rank 32,
  alpha 16, rsLoRA enabled, dropout 0, bias none, PiSSA `pissa_niter_16`.
  Expected trainable parameter count: 233,455,616.
- PiSSA initialization and residual reload are compared with the untouched
  checkpoint under the same explicit BF16 autocast used by training. LoRA A/B
  and final-logit BF16 dtypes, finite logits, and exact argmax equality are
  mandatory. Behavioral parity additionally requires at least 90% overlap of
  the top 32 tokens, total-variation distance at most 0.05, Jensen-Shannon
  divergence at most 0.002 nats, and centered-logit NRMSE at most 0.05. The
  maximum single-token probability change remains a recorded diagnostic and
  is mathematically bounded by the total-variation gate. Raw maximum and mean
  absolute logit errors are also diagnostics; pointwise relative error near
  zero and softmax-invariant global logit shifts are not gate criteria.
- AdamW (`adamw_torch`), learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8,
  weight decay 0, cosine schedule, warmup ratio 0.05, max gradient norm 1.
- Five epochs. Default geometry is microbatch 1/GPU x 2 GPUs x accumulation 8
  = global batch 16. Microbatch 2 / accumulation 4 is selected only if the
  worst 2,048-token smoke stays within 44 GiB per GPU with no material swap
  growth.
- ZeRO-3 without CPU/NVMe offload, reentrant activation checkpointing,
  `use_cache=false`, global supervised-token-normalized loss, and no W&B. The
  reentrant variant is required by the locked DeepSpeed 0.19.2/PyTorch 2.9.1
  combination: a real 2,048-token two-rank smoke showed that non-reentrant
  recomputation observed already-partitioned frozen weights as shape `[0]`
  after the original forward had saved their full shapes. Qwen passes its
  checkpointed hidden state positionally, input gradients are explicitly
  enabled, every decoder layer is checkpointed once, and unused-parameter
  discovery is disabled, satisfying the reentrant variant's constraints.
- Linear-attention kernels use FLA; full-attention layers use SDPA. Kernel
  lengths 63/64/65, two-rank NCCL, ZeRO-3, resume, and portable-adapter parity
  are hard launch gates.

The upstream `causal-conv1d` release wheel requires `GLIBC_2.32` and is not
loadable on gu30 (`glibc 2.28`).  The run therefore uses a locally built,
content-pinned SM89 wheel from upstream commit
`4f6ae4e26ae5fe8af9372f8d312ab25cc4595223`, with project patch SHA256
`ef259fe6a265096e02a20da5659037e9f5f4918c4eb74fb562360eac037c01f5`.
The wheel SHA256 is
`33a15b2e298de90aa095680bbaa1025dc43a1a27b07488c5e060c2cea28b4ed0`;
its installed extension SHA256 is
`f8f840d36f5b822cf8a36e69653a68f4b53b387a8dcd61294e31577a5cfa0c76`
and its maximum GLIBC requirement is 2.14.  `scripts/build_causal_conv1d.sh`
records the rebuild recipe.  Runtime also supplies the pinned Python 3.11
headers through `CPATH`, which Triton needs to compile its driver helper.

Checkpoint 0 is written after PiSSA plus optimizer/scheduler initialization
and before the first real update. During an epoch a complete resume checkpoint
is written every 50 optimizer steps and only the latest two local copies are
retained. Every epoch writes a durable full resume checkpoint and a portable
ordinary-LoRA adapter. The portable conversion has rank 64, alpha `16*sqrt(2)`
and 466,911,232 parameters. Completion markers are written atomically, and
resume selection uses manifest `global_step` plus fingerprints, never mtime.
Because the cosine/warmup scheduler is owned by Transformers rather than the
DeepSpeed engine, its state is a separately hashed checkpoint payload;
`last_epoch` and `_step_count` must agree with the persisted global step before
save and after restore. Each torchrun rank saves and replays only its current
GPU's RNG state, never the peer GPU's generator. On mid-epoch resume the same
verified RNG payload is replayed after Trainer fast-forwards the dataloader.

## Formal evaluation and selection

Generation is non-thinking and deterministic:

```text
beam=10, num_return_sequences=10, do_sample=false, max_new_tokens=512,
length_penalty=1, early_stopping=false, eos=<|im_end|>, pad=<|endoftext|>
```

Before the first formal evaluation, the longest canonical validation prompt
must pass both a real formal beam-10 decode/parse and a separate non-formal
memory stress with `min_new_tokens=max_new_tokens=512`. The latter must return
exactly ten sequences of prompt length plus 512 on both ranks; an early EOS
cannot satisfy this memory gate.

The first 1, 2, 3, 4, 5 and 10 raw beams are evaluated independently without
deduplication. Invalid syntax receives the frozen worst edit penalty. Metrics,
in priority order, are:

1. Connectivity (all predicted edge pairs exactly correct; bond order ignored)
2. Exact (all edge pairs and bond orders correct)
3. edit bond (lower is better; wrong order on an existing pair counts once)
4. edge IoU (bond order ignored)
5. edge F1 (bond order ignored)
6. evaluation loss (lower is better)

Checkpoint selection compares the @10 values lexicographically in that order.
Loss differences no larger than 1e-8 are ties, resolved in favor of the
earlier epoch. Report exact integer/rational aggregates, Wilson confidence
intervals for the two binary metrics, and reaction-level BCa intervals with
10,000 seed-42 replicates for mean auxiliary metrics; also report edit P50/P95.

All five sealed epoch checkpoints are evaluated on the 994-example validation
set after training; Trainer does not run a second implicit validation pass
inside the training epoch. This keeps each durable checkpoint at the true
post-training boundary and makes its saved RNG state the state entering the
next epoch. After locking the selected checkpoint, the 996-example test set is
evaluated once. Optional thinking inference is a separately labeled
exploratory profile and cannot affect formal selection.

The v24 comparison is computed only after Qwen selection, on the exact common
102-reaction test intersection under the new 0.1-discretized gold labels and
the same metrics. Frozen v24 prediction SHA256:
`11e4eea54a1f2d5e45af003a3607fca75fc670fdb96b35ef7fa38b10cf34bc3c`.

## Persistence and host safety

The default policy is to start GPU work only when both GPUs have no foreign
compute process.  The explicitly confirmed one-time exception for this
Transition1x workflow permits only the eight LAMMPS processes whose full
process identities were captured before the shared launch.  Any new,
restarted, PID-reused, or otherwise unbound GPU process blocks the next GPU
phase; the allowlist cannot grow, although it may shrink as an original
process exits.  The workflow never signals or kills any of these processes or
any other user's process.

Sharing does not relax the `45,056 MiB` per-GPU peak limit.  It also retains
`MemAvailable >= 16 GiB` and host swap growth `<= 256 MiB` for every compute,
memory-smoke, resume, and formal-training/evaluation phase.  The sole scoped
exception is one-time PiSSA bundle serialization and cached payload hashing:
both their starting and ending `MemAvailable` must be at least 16 GiB, while
swap growth is recorded as a diagnostic rather than a rejection criterion.
The initial 54.7 GB serialization completed with `MemAvailable` approximately
20--23 GiB and observed peak swap growth 3,081.7 MiB; the user explicitly
accepted this file-cache-induced cold-page eviction on 2026-07-28. Cached
reuse must retain non-default resource evidence and may not silently report a
missing observation as zero. The GPU peak is measured as whole-device memory
usage, including the allowed LAMMPS allocation, rather than as Janus-TS
process memory alone.  The launcher sets `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`,
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and
`DS_BUILD_OPS=0`; `CPATH` points only to the pinned build environment's Python
3.11 headers. A tagged tmux supervisor plus a preserving `@reboot` crontab
entry may retry transient failures after 1, 5, and 15 minutes. It must never
silently change configuration in response to an OOM or other permanent error.

A permanent supervisor state is never cleared by an ordinary launch or reboot.
After a diagnosed implementation fault is fixed, tested, committed, and the
tracked source tree is clean, the operator may run the explicit
`rearm-after-fix` action.  It requires the exact SHA256 values of the stopped
`state.json` and `PERMANENT_FAILURE.json`, the distinct failed and fixed 40-hex
Git revisions, a nonempty reason, and the unchanged full supervisor launch
identity.  Under the same nonblocking singleton lock it verifies that the
failure payload exactly matches the terminal state and that `HEAD` is the fixed
revision.  It then writes a content-addressed receipt below
`artifacts/logs/supervisor/recovery-history/`, writes an atomic
`ACKNOWLEDGED_FAILURE.json` pointer, and only then rearms the state as the final
atomic commit while preserving launch and transient-failure counters.  The
original `PERMANENT_FAILURE.json` bytes are never overwritten, so a failed
final state write remains terminal and the same exact hashes can be retried
idempotently.  Thus a failed first launch is retained verbatim and the next
delegated output is `command-attempt-0002.log`; hash, revision, dirty-tree,
active-lock, identity, or nonterminal-state mismatches all fail closed.  The
existing tagged `@reboot` entry remains valid because the tag, log directory,
lock, and delegated command do not change.
