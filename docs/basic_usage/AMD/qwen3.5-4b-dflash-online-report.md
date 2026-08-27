# Qwen3.5-4B DFlash Draft — Online Training & Served Acceptance Report (ROCm)

This report documents a full, reproducible run that trains a Qwen3.5-4B **DFlash**
draft with SpecForge's online disaggregated pipeline on AMD ROCm, exports it to a
servable HF draft, serves it under SGLang speculative decoding, and **measures the
real end-to-end acceptance length** on three held-out datasets.

Unlike the short functional runs in [`amd_rocm.md`](./amd_rocm.md) (which only
report training-time `acc` and an *estimated* acceptance length), this run trains
the draft to convergence and reports **measured served** numbers.

## 1. Environment

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI355X (gfx950, 309 GB HBM/GPU) |
| Container | `specforge-514` (image `lmsysorg/sglang:v0.5.14-rocm720-mi35x`) |
| Torch / ROCm | torch 2.9.1+rocm7.2.0 |
| Fused kernels | liger-kernel (RMSNorm + SwiGLU), AITER (serving + capture) |
| Target model | `Qwen/Qwen3.5-4B` — hybrid Mamba (24 linear-attention + 8 full-attention layers), vocab 248320 |
| Draft | DFlash, 5 decoder layers, hidden 2560, `block_size=16`, `mask_token_id=248070`, `target_layer_ids=[1,8,15,22,29]` |

**Liger prerequisite (torch < 2.11):** enabling `use_liger_kernel` triggers an
inductor compile crash (`NameError: 'TRITON' is not defined`) when `flex_attention`
is compiled. The run works around it with **`TORCHDYNAMO_DISABLE=1`**, which runs
flex eager/unfused while keeping liger's own Triton RMSNorm/SwiGLU kernels active
(the log shows the expected `flex_attention called without torch.compile()` notice).

## 2. Data

~10k ShareGPT prompts prepared with a 5% eval split:

```bash
python scripts/prepare_data.py --dataset sharegpt --sample-size 10000 \
  --split-eval --output-path cache/dataset10k
```

The online producer ingested **9,244 base prompts** and replayed them for
**100 prompt-epochs** (`prompt_epochs=100`), regenerating fresh target continuations
each epoch (so no empty-loss-region samples, unlike the offline truncation path).

## 3. Training configuration

Base recipe:
[`examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-amd.yaml`](../../../examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-amd.yaml)
(already `use_liger_kernel: true`), with the overrides shown below.

| Parameter | Value |
| --- | --- |
| Strategy | `dflash` |
| Deployment | online, disaggregated (Mooncake), single node, 2 GPUs (capture GPU0 + trainer GPU1) |
| Attention backend (trainer) | `flex_attention` (eager, via `TORCHDYNAMO_DISABLE=1`) |
| `use_liger_kernel` | `true` (fused RMSNorm + SwiGLU) |
| `max_length` | 2048 |
| `chat_template` | `qwen3.5` |
| `batch_size` | 2 |
| `accumulation_steps` | 4 |
| `learning_rate` | 6.0e-4 (cosine, `warmup_ratio=0.04`) |
| `max_grad_norm` | 1 |
| `num_anchors` | 512 |
| `loss_decay_gamma` | 7 |
| `num_epochs` | 100 |
| `save_interval` | 2000 steps |

Launch (trainer side, GPU1):

```bash
TORCHDYNAMO_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-amd.yaml \
  data.train_data_path=./cache/dataset10k/sharegpt_train.jsonl \
  training.num_epochs=100 training.save_interval=2000 training.log_interval=50
```

The producer/capture side runs the AITER-patched SGLang capture server with
`--attention-backend aiter --disable-radix-cache` and aux-layer-ids
`1 8 15 22 29` (must match the draft's `target_layer_ids`), fronted by a Mooncake
master. See [`amd_rocm.md`](./amd_rocm.md) Section 4 for the full disaggregated
launch.

## 4. Training process & accuracy

Top-1 draft-token accuracy (`train/acc`, the per-position acceptance rate) and
loss over the run:

| Step | `train/acc` | `train/loss` |
| --- | --- | --- |
| 50 | 0.045 | 8.34 |
| 100 | 0.043 | 7.64 |
| 24000 (exported) | ~0.91 | ~0.18 |
| 25000 (final) | 0.937 | 0.134 |

`acc` climbs from ~0.04 to a plateau around **~0.92** (final-step readings
0.91–0.94) — far above the short-run plateau of ~0.13 reported in the base
tutorial. Training was early-stopped once `acc` clearly flattened; the best
checkpoint (`step24000`) was used for export. Checkpoints were saved every 2000
steps.

**Training throughput** (single MI355X trainer, `batch_size=2`, `max_length=2048`):
~**2.5 global samples/s** and ~**1,150 optimizer steps/hour** (`optimizer_step_time`
≈ 3.0–3.3 s, data-wait ≈ 1 ms — the run is compute-bound on target regeneration,
not data-starved).

> **Caveat:** `acc ≈ 0.92` is measured on the training pool, where ~100 replays
> over 9,244 prompts allow some memorization. The honest quality signal is the
> **served** acceptance length on held-out datasets (Section 6), which is disjoint
> from the ShareGPT training data.

## 5. Export to HF draft

```bash
specforge export --to hf \
  --checkpoint outputs/qwen3.5-4b-dflash-online/qwen3.5-4b-dflash-online-step24000 \
  --draft-config configs/qwen3.5-4b-dflash.json \
  --output-dir export/qwen3.5-4b-dflash-hf
```

Produces `config.json` + `model.safetensors` (58 tensors: `fc`, `hidden_norm`,
`norm`, 5 decoder layers). A DFlash draft **owns no embedding / lm_head** — it
shares the target's at serve time (`draft_vocab_size == target vocab`), so
`--embedding-source` is a no-op here.

## 6. Serving & measured acceptance length

Served on a single MI355X GPU (target + draft co-located):

```bash
SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1 \
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path export/qwen3.5-4b-dflash-hf \
  --speculative-dflash-block-size 16 \
  --attention-backend aiter --page-size 16 --disable-radix-cache \
  --mem-fraction-static 0.85 --host 127.0.0.1 --port 30000
```

Both weights load (`type=DFlashDraftModel, mem usage=1.10 GB`), CUDA graphs
capture, `/health` returns 200, and a single greedy `/generate` returns
`meta_info.spec_accept_length = 2.21` (> 1 — draft tokens are genuinely accepted).
`--disable-radix-cache` is required for the hybrid-Mamba target on ROCm.

> **Serving gotcha (shared node):** `HIP_VISIBLE_DEVICES=N` can use a *different*
> device ordering than torch's default enumeration. Probe real free memory under
> the target env before launching:
> `for n in 0 1 2 3; do HIP_VISIBLE_DEVICES=$n python -c 'import torch;print(torch.cuda.mem_get_info(0)[0]/1e9)'; done`
> and pick an N reporting ~308 GB. Landing on a busy GPU shows up as
> `avail mem≈28 GB` → `ValueError: Loaded weights leave no GPU memory for the KV cache`.

### Measured results (`specforge benchmark`, greedy, `block_size=16`)

```bash
for ds in gsm8k humaneval mt-bench; do
  specforge benchmark --model Qwen/Qwen3.5-4B --dataset $ds \
    --base-url http://127.0.0.1:30000 --concurrency 16 \
    --max-new-tokens 256 --temperature 0 --trust-remote-code \
    --output-json bench/$ds.json
done
```

| Dataset | Acceptance length | Output throughput | Samples |
| --- | --- | --- | --- |
| **GSM8K** (math) | **2.07** | 2,124 tok/s | 1,024 |
| **HumanEval** (code) | **2.99** | 2,601 tok/s | 1,024 |
| **MT-Bench** (chat) | **1.74** | 1,677 tok/s | 1,024 |

Acceptance is highest on code (HumanEval — predictable continuations) and lowest
on open-ended chat (MT-Bench). Because the benchmark datasets are disjoint from the
ShareGPT training pool, these reflect genuine generalization, not memorization.

## 7. Summary

| Metric | Result |
| --- | --- |
| Training-time draft accuracy (`acc`) | ~0.04 → **~0.92** (final 0.937) |
| Draft loss | 8.34 → **0.134** |
| Training throughput | ~2.5 samples/s, ~1,150 steps/hr |
| Served single-request `spec_accept_length` | **2.21** |
| Acceptance length — GSM8K / HumanEval / MT-Bench | **2.07 / 2.99 / 1.74** |
| Output throughput — GSM8K / HumanEval / MT-Bench | 2,124 / 2,601 / 1,677 tok/s |

This confirms DFlash speculative decoding trains **and serves** end-to-end on
ROCm/AITER against a hybrid-Mamba target, with real measured acceptance lengths of
1.7–3.0 depending on domain — a substantial improvement over the base tutorial's
estimated 1.1–1.2.
