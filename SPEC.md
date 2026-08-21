# Qwen3.5-4B Palettization — Full Project Specification

**Status**: Updated — Trainable indices + LoRA-only architecture  
**Date**: 2026-08-21  
**Target model**: `Qwen/Qwen3.5-4B` (`Qwen3_5ForConditionalGeneration`)  
**Goal**: 2-bit per-group (GS=256) scalar palettization with **trainable indices** via Gumbel-Softmax + LoRA rank-16 on all palettized Linears. No dense correction layer. Target: cos > 0.999 vs fp16 teacher.

---

## 1. Model Architecture

### 1.1 Top-level
```
Qwen3_5ForConditionalGeneration
├── model.visual           (24-block vision encoder)        ← SKIP
├── model.mtp              (1 multi-token-prediction layer)  ← SKIP
└── model.language_model   (32-layer hybrid decoder)         ← TARGET
    ├── embed_tokens       [248320, 2560]  635.7M  bf16     ← SKIP (keep fp16)
    ├── layers[0..31]      32 hybrid layers                ← PALETTIZE
    └── norm               RMSNorm [2560]                   ← KEEP fp16
    (lm_head = tied with embed_tokens)                       ← SKIP
```

### 1.2 Hybrid layer pattern (8 super-blocks)
```
Super-block 0: layers [0,1,2,3]   = L L L F   (3 GatedDeltaNet + 1 Full-attn)
Super-block 1: layers [4,5,6,7]   = L L L F
...
Super-block 7: layers [28,29,30,31] = L L L F
```

### 1.3 Per-layer inventory (palettizable Linears)

#### GatedDeltaNet layer (24 layers, 6 palettizable Linears each)
| Tensor | Shape | Params |
|---|---|---|
| linear_attn.in_proj_qkv.weight | [8192, 2560] | 21.0M |
| linear_attn.in_proj_z.weight | [4096, 2560] | 10.5M |
| linear_attn.out_proj.weight | [2560, 4096] | 10.5M |
| mlp.gate_proj.weight | [9216, 2560] | 23.6M |
| mlp.up_proj.weight | [9216, 2560] | 23.6M |
| mlp.down_proj.weight | [2560, 9216] | 23.6M |

#### Full-attn layer (8 layers, 7 palettizable Linears each)
| Tensor | Shape | Params |
|---|---|---|
| self_attn.q_proj.weight | [8192, 2560] | 21.0M |
| self_attn.k_proj.weight | [1024, 2560] | 2.6M |
| self_attn.v_proj.weight | [1024, 2560] | 2.6M |
| self_attn.o_proj.weight | [2560, 4096] | 10.5M |
| mlp.gate_proj.weight | [9216, 2560] | 23.6M |
| mlp.up_proj.weight | [9216, 2560] | 23.6M |
| mlp.down_proj.weight | [2560, 9216] | 23.6M |

### 1.4 Kept fp16 (NOT palettized)
- SSM params (A_log, dt_bias) — tiny (<1K), sensitive
- conv1d — not a Linear
- RMSNorms — tiny (<3K), sensitive
- in_proj_a, in_proj_b — too small (32 rows)
- embed_tokens / lm_head — kept fp16 (may palettize later)

---

## 2. Architecture: Trainable Indices + LoRA-only

### 2.1 Design

```
Super-block structure:

  Input: H_in (from teacher's previous super-block output)
    │
    ▼
  ┌──────────────────────────────────────────┐
  │ Layer 0: GatedDeltaNet (2-bit)          │
  │   palette (G, 4) bf16 — TRAINABLE       │
  │   index_logits (4, K, N) fp16 — TRAINABLE│
  │   LoRA rank-16 on all 6 Linears          │
  └──────────────────────────────────────────┘
    │
    ▼
  ┌──────────────────────────────────────────┐
  │ Layer 1: GatedDeltaNet (2-bit)          │  (same structure)
  └──────────────────────────────────────────┘
    │
    ▼
  ┌──────────────────────────────────────────┐
  │ Layer 2: GatedDeltaNet (2-bit)          │  (same structure)
  └──────────────────────────────────────────┘
    │
    ▼
  ┌──────────────────────────────────────────┐
  │ Layer 3: Full Attention (2-bit)         │
  │   palette (G, 4) bf16 — TRAINABLE       │
  │   index_logits (4, K, N) fp16 — TRAINABLE│
  │   LoRA rank-16 on all 7 Linears          │
  └──────────────────────────────────────────┘
    │
    ▼
  Output: H_out (matched to teacher's super-block output)
```

**No dense correction layer. No stage-2 palettization.**

### 2.2 Why trainable indices

The previous approach had frozen indices (from GPTQ + kmeans at calibration time). The model plateaued at cos~0.93 because:
1. **Assignment error** — kmeans put weights in wrong clusters; frozen indices can't fix this
2. The correction layer tried to compensate downstream but couldn't undo per-layer error

With **trainable indices** (Gumbel-Softmax relaxation):
- The model can reassign weights to better clusters during training
- Gradients flow through the softmax to index logits, enabling continuous optimization
- As temperature anneals to 0, logits become one-hot (hard indices)

### 2.3 Why LoRA on all Linears (not a correction layer)

With trainable indices eliminating assignment error, the remaining error is **representation error** (4 palette values can't perfectly represent 256 distinct weights). This residual error is small per-layer (~1%) but compounds through 4 layers (0.99^4 ≈ 0.96).

LoRA rank-16 on each palettized Linear directly compensates for this per-layer residual:
- More direct than a downstream correction layer (fixes error at source)
- Tiny: ~160K params per Linear × 25 Linears = ~8M total
- Mergeable into weights at inference (zero overhead)

### 2.4 What's trainable per super-block

| Component | Params | Trainable? | Dtype |
|---|---|---|---|
| Palettes (25 Linears × n_groups × 4) | ~2,500 | ✅ | bf16 |
| Index logits (25 Linears × 4 × K × N) | ~8M | ✅ (anneals to one-hot) | fp16 |
| LoRA rank-16 A+B (25 Linears × 2) | ~8M | ✅ | bf16 |
| Layernorms + SSM + conv1d | ~600K | ✅ | bf16/fp32 |
| **Total trainable** | **~17M** | | |
| Frozen (embed_tokens) | ~636M | ❌ | bf16 |

### 2.5 Inference-time model

After training (temperature τ→0):
1. `indices = argmax(index_logits, dim=0)` — hard indices, drop logits
2. `W_merged = W_palettized + LoRA_A @ LoRA_B` — merge LoRA into weights
3. Deploy: 2-bit palettized weights with frozen indices + merged LoRA
4. **No correction layer, no logits, no LoRA at inference** — pure 2-bit palettized model

---

## 3. Reproducible Pipeline (all 8 super-blocks)

### 3.1 Pipeline stages

```
Stage 1: Calibration (offline, once per super-block)
  ├── Load teacher prefix (layers 0 to sb_end-1)
  ├── Capture activations via forward hooks on FineWeb-Edu
  ├── GPTQ + kmeans1d → initial 2-bit indices + palette
  └── Save to palettized/superblock_{sb_idx}/

Stage 2: Training (per super-block, independent)
  ├── Build student (palettized layers + LoRA rank-16 + trainable logits)
  ├── Load teacher prefix (same as student)
  ├── Train with Gumbel-Softmax soft forward + LoRA
  ├── Anneal temperature τ: 1.0 → 0.01 over training
  ├── Eval every 250 steps on held-out set (256 sequences)
  └── Save best model to trained/superblock_{sb_idx}_best/

Stage 3: Merge (after all 8 super-blocks trained)
  ├── For each super-block:
  │   ├── Extract hard indices: argmax(index_logits)
  │   ├── Merge LoRA into palette: palette += LoRA correction
  │   └── Save final 2-bit weights
  └── Assemble full model from 8 super-blocks
```

### 3.2 File structure

```
/root/qwen35_palettize/
├── SPEC.md                           ← This document
├── scripts/
│   ├── qwen_model.py                 ← PalettizedLinear + QwenLoRA + model loading
│   ├── train_qwen.py                 ← Training loop (stage 2)
│   ├── calib_qwen.py                 ← Calibration (stage 1)
│   ├── palettize_core.py             ← 2-bit packing + kmeans1d
│   ├── fused_lut_kernel.cu           ← CUDA kernel (hard + soft variants)
│   └── fused_lut_linear_cuda.py      ← Python wrapper for CUDA kernel
├── palettized/
│   ├── superblock_0/                 ← Stage 1 output: .idx2 + .lut_scalar + metadata.json
│   ├── superblock_1/
│   └── ...
├── trained/
│   ├── superblock_0_best/            ← Stage 2 output: trained weights (.pt files + _resume.json)
│   ├── superblock_1_best/
│   └── ...
├── logs/
│   ├── calib_sb0.log
│   ├── train_sb0.log
│   └── ...
├── eval_tokens.pt                    ← Held-out eval set (shared across super-blocks)
└── hyperparams_qwen.json             ← Live-tunable hyperparameters (/tmp/)
```

### 3.3 Commands (reproducible for any super-block)

```bash
# Stage 1: Calibrate super-block N
python3 scripts/calib_qwen.py --sb_idx N --n_seqs 8192 --seq_len 2048

# Stage 2: Train super-block N
USE_TC_FWD=1 USE_TC_BWD_GX=1 \
python3 scripts/train_qwen.py \
  --sb_idx N \
  --max_steps 5000 \
  --seq_len 128 \
  --batch_size 8 \
  --lora_rank 16 \
  --use_soft_indices 1 \
  --tau_init 1.0 \
  --tau_final 0.01 \
  --tau_anneal_steps 4000

# Stage 3: Merge (after all 8 trained)
python3 scripts/merge_qwen.py --all
```

### 3.4 Hyperparameters

Live-tunable via `/tmp/hyperparams_qwen.json`:
```json
{
  "groups": {
    "palettes": true,
    "lora": true,
    "indices": true,
    "layernorms": true
  },
  "lrs": {
    "palettes": 3e-4,
    "lora": 3e-4,
    "indices": 1e-3,
    "layernorms": 1e-4
  },
  "loss_type": "1-cos+norm_mse",
  "loss_weights": {"cos": 0.5, "mse": 0.5},
  "gradient_clip": 0.3,
  "eval_every": 250,
  "log_every": 50
}
```

### 3.5 Optimizer setup (Dolphin v13 proven, no tricks)

- **Muon** (Newton-Schulz) for 2D non-palette weights (LoRA B matrices)
  - momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0
  - scale_factor=0.2*sqrt(max(A,B)) (default)
  - fp32 master weights (via FP32MasterMuon wrapper)
- **AdamW** for palettes + LoRA A matrices + index_logits + 1D params
  - betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
  - fp32 master weights (via FP32MasterAdamW wrapper)
- **Cosine LR scheduler**, no warmup
- **bf16 model params** (forward/backward), **fp32 master params** (optimizer)

---

## 4. CUDA Kernels

### 4.1 Hard kernel (existing, working)

Files: `fused_lut_kernel.cu` + `fused_lut_linear_cuda.py`

- `fused_lut_linear_fwd` — TC forward with gather (indices → palette → W → matmul)
- `fused_lut_linear_bwd_grad_x` — TC backward (grad_y @ W.T)
- `fused_lut_linear_bwd_grad_palette` — fp32 atomicAdd scatter
- `fused_lut_linear_bwd_grad_bias` — reduction

Status: ✅ 41/41 tests pass, 2× faster than reference

### 4.2 Soft kernel (NEW — awaiting delivery by kernel agent)

- `fused_lut_linear_soft_fwd` — Gumbel-Softmax forward (logits → P → weighted W → matmul)
- `fused_lut_linear_soft_bwd_grad_x` — TC backward (same as hard, W from P+palette)
- `fused_lut_linear_soft_bwd_grad_logits` — softmax Jacobian-vector product
- `fused_lut_linear_soft_bwd_grad_palette` — weighted scatter_add (P-weighted)
- `fused_lut_linear_bwd_grad_bias` — reuse existing

Status: 📝 Work instruction written: `/home/z/my-project/download/gumbel_softmax_kernel_work_instruction.md`

### 4.3 Dual-mode PalettizedLinear

```python
class PalettizedLinear(nn.Module):
    def forward(self, x):
        if self.training and self.use_soft_indices:
            # Soft forward (Gumbel-Softmax) — training only
            y = CUDAFusedLUTLinearSoft.apply(x, self.palette, self.index_logits, ...)
        else:
            # Hard forward — eval or when soft is disabled
            y = CUDAFusedLUTLinear.apply(x, self.palette, self.indices_int8, ...)
        # LoRA correction (always, in both modes)
        if self.lora is not None:
            y = y + self.lora(x)
        return y
```

---

## 5. Temperature Annealing

```
τ = max(τ_final, τ_init * (1 - step / τ_anneal_steps))

Default: τ_init=1.0, τ_final=0.01, τ_anneal_steps=4000

Step 0:    τ=1.0  (soft, exploratory — indices can change freely)
Step 1000: τ=0.75
Step 2000: τ=0.50
Step 3000: τ=0.25
Step 4000: τ=0.01 (nearly hard — indices are committing)
Step 5000: τ=0.01 (hard — equivalent to argmax)
```

At τ=0.01, the soft kernel converges to the hard kernel. Extract final indices: `indices = argmax(index_logits, dim=0)`.

---

## 6. Expected Outcomes

### 6.1 Compression

| Component | Original (bf16) | Palettized (2-bit) | Compression |
|---|---|---|---|
| 32 layers (Linears) | 3.40 GB | 0.48 GB | 7.1× |
| SSM + norms + conv1d | 42 MB | 42 MB | 1× |
| LoRA (merged at inference) | — | 0 MB | merged |
| embed_tokens + lm_head | 1.27 GB | 1.27 GB | 1× |
| **Total** | **4.71 GB** | **1.80 GB** | **2.6×** |

### 6.2 Quality targets

| Metric | Previous (frozen indices + correction) | New (trainable indices + LoRA) |
|---|---|---|
| Super-block output cosine | ~0.93 | **>0.999** |
| Full-model perplexity | ~5% increase | **<2% increase** |
| Generation quality | Minor degradation | **Indistinguishable** |

### 6.3 Training cost

| Resource | Value |
|---|---|
| GPU memory | ~20 GB (logits ~8 GB, but no correction layer saves 225 MB) |
| Step time | ~1.0s (soft kernel +35% overhead vs hard) |
| Steps per super-block | 5000 |
| Time per super-block | ~1.4 hours |
| Total (8 super-blocks) | ~11 hours |

---

## 7. Implementation Status

### Done ✅
- [x] Stage 1 calibration (calib_qwen.py)
- [x] Hard CUDA kernel (fused_lut_kernel.cu) — TC forward + backward, 41/41 tests pass
- [x] PalettizedLinear with hard kernel integration
- [x] QwenLoRA (rank-32, will switch to rank-16)
- [x] Training loop with fp32 master weights (Muon + AdamW)
- [x] Eval set (256 held-out sequences)
- [x] Live JSON hyperparameter tuning
- [x] Resume from checkpoint

### Awaiting kernel agent 📝
- [ ] Soft CUDA kernel (Gumbel-Softmax forward + backward)
- [ ] `CUDAFusedLUTLinearSoft` autograd Function
- [ ] `index_logits` parameter in PalettizedLinear
- [ ] Temperature annealing in training loop
- [ ] Dual-mode forward (soft train / hard eval)

### After kernel delivered 🔲
- [ ] Switch LoRA from rank-32 to rank-16
- [ ] Remove correction layer from build_student_super_block
- [ ] Remove calib_stage2.py (no longer needed)
- [ ] Add merge_qwen.py (extract hard indices + merge LoRA)
- [ ] Run training for all 8 super-blocks
- [ ] Full-model perplexity evaluation

---

## 8. References

- **GSQ** (Gumbel-Softmax for LLM quant): arXiv:2604.18556, https://github.com/IST-DASLab/GSQ
- **LLT** (Learnable Lookup Table): CVPR 2022, https://github.com/SYSU-SAIL/LLT
- **LUT-Q** (Iterative LUT training): arXiv:1811.05355
- **FLUTE** (LUT matmul kernel): arXiv:2407.10960
- **Gumbel-Softmax**: arXiv:1611.01144 (Jang et al.)
- **LSQ** (Learned Step Size): arXiv:1902.08153
- **Knowledge base**: `/home/z/my-project/download/trainable_indices_kb.md`
- **Kernel work instruction**: `/home/z/my-project/download/gumbel_softmax_kernel_work_instruction.md`
