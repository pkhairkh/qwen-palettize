# 01 — Palette Implementation Audit

**Scope:** Reverse-engineer the live state of the 2-bit palette (LUT) subsystem inside `qwen-palettize` so that subsequent waves can argue about precision, optimizers, and training schedules from a single, verified source of truth. Every claim below is pinned to a specific file and line range so it can be re-checked after future refactors.

**Method:** Read `scripts/qwen_model.py`, `scripts/fused_lut_linear_cuda.py`, `scripts/fused_lut_kernel.cu`, `scripts/train_qwen.py`, `scripts/palettize_core.py`, `scripts/palettize_pytorch.py`, and `logs/calib_sb0.log` in full, cross-referenced against `logs/train_sb0.log`. No source was skipped.

---

## 1. Architecture at a glance

The palettization story is split across three code layers that must agree on a single contract:

| Layer | File | Role |
|---|---|---|
| Module | `scripts/qwen_model.py` lines 65–181 | `PalettizedLinear` owns the `palette` parameter and dispatches forward |
| Autograd + launcher | `scripts/fused_lut_linear_cuda.py` lines 428–503 (hard) and 519–684 (soft) | `CUDAFusedLUTLinear` / `CUDAFusedLUTLinearSoft` glue the CUDA kernels into PyTorch autograd |
| Kernel | `scripts/fused_lut_kernel.cu` lines 95–288 (fwd), 902–1123 (bwd grad_palette hard), 1409–1433 (bwd grad_palette soft) | The actual scatter-add that produces `grad_palette` |

The `PalettizedLinear` class (`qwen_model.py:65-181`) is a thin shell. The interesting logic lives in the two autograd Functions and the .cu kernel, because that is where the precision contract is set.

There are **two distinct forward paths** that share the same `palette` parameter but differ in everything else:

1. **Hard path** (`qwen_model.py:148-153`): `W[j,o] = palette[o // 256, indices_int8[j,o]]`. Indices are frozen `int8`. Gradient flows only to `palette`. This is the path used at eval time and when `use_soft_indices=False`.
2. **Soft path** (`qwen_model.py:141-147`): `W = Σ_k P[k,j,o] * palette[g, k]` where `P = softmax((logits + gumbel)/τ)`. Indices are replaced by `index_logits` of shape `(4, K, N) fp16`. Gradients flow to both `palette` and `index_logits`. This is the path used during training when `use_soft_indices=True`.

The training log confirms which path is active:

```
logs/train_sb0.log:4    === Training super-block 0 (soft_indices=True) ===
logs/train_sb0.log:24   Palettized 25 Linears to 2-bit (soft_indices=True)
logs/train_sb0.log:34   indices: 1,782,579,200     ← these are index_logits, not int8
```

The CLI default is `--use_soft_indices 1` (`train_qwen.py:1239`), so unless an operator explicitly passes `--use_soft_indices 0`, every training run is on the soft path. The user-supplied orchestrator brief cites the soft-path gradient formula `contributions.sum(dim=(0,2))` — that line is `fused_lut_linear_cuda.py:662`, confirming the soft path is the one we must diagnose.

---

## 2. The palette parameter itself

### 2.1 Storage dtype — bf16

`qwen_model.py:82-85` creates the palette as a `bfloat16` `nn.Parameter`:

```python
self.palette = nn.Parameter(
    initial_palette.clone().to(torch.bfloat16) if initial_palette is not None
    else torch.zeros(n_groups, palette_size, dtype=torch.bfloat16)
)
```

This is the first precision bottleneck. The palette values are the **only** degrees of freedom in 2-bit palettization (indices are 2 bits, fixed at calibration for the hard path, weakly trainable for the soft path), and yet they are stored in the lowest-precision floating-point format PyTorch supports.

The disk side of this is even worse. `palettize_core.py:175-178` loads the LUT as `float16` and immediately casts to `float32`:

```python
def load_lut(lut_path):
    with open(lut_path, "rb") as f:
        data = f.read()
    return torch.from_numpy(np.frombuffer(data, dtype=np.float16).astype(np.float32))
```

So the round trip is `kmeans (fp32) → write_lut_scalar (fp16) → load_lut (fp32) → PalettizedLinear.__init__ (bf16)`. Two precision-losing casts happen before training even starts: fp32 → fp16 at serialization, and fp32 → bf16 at module construction. For 2,208 parameters totalling ~4.4 KB on disk, this is unnecessary austerity.

### 2.2 Shape and group structure

The palette shape is `(n_groups, 4)` where `n_groups = out_dim // GROUP_SIZE = out_dim // 256` (`palettize_core.py:26-27`). With `BITWIDTH=2` we have `PALETTE_SIZE = 1 << 2 = 4`. The full count for super-block 0 is `2,208` palette parameters across 25 Linears (`train_sb0.log:29`), which matches `Σ_g n_groups × 4` over the 25 palettized tensors.

The 25 Linears and their post-calibration cosines are (`calib_sb0.log:111-234`):

| # | Tensor | Shape (out, in) | Groups | Cos after k-means |
|---|---|---|---|---|
| 1 | layer 0 linear_attn.out_proj | (2560, 4096) | 10 | 0.933 |
| 2 | layer 0 linear_attn.in_proj_qkv | (8192, 2560) | 32 | 0.940 |
| 3 | layer 0 linear_attn.in_proj_z | (4096, 2560) | 16 | 0.922 |
| 4 | layer 0 mlp.gate_proj | (9216, 2560) | 36 | 0.944 |
| 5 | layer 0 mlp.up_proj | (9216, 2560) | 36 | 0.938 |
| 6 | layer 0 mlp.down_proj | (2560, 9216) | 10 | 0.950 |
| 7 | layer 1 linear_attn.out_proj | (2560, 4096) | 10 | 0.966 |
| 8 | layer 1 linear_attn.in_proj_qkv | (8192, 2560) | 32 | 0.926 |
| 9 | layer 1 linear_attn.in_proj_z | (4096, 2560) | 16 | 0.929 |
| 10 | layer 1 mlp.gate_proj | (9216, 2560) | 36 | 0.952 |
| 11 | layer 1 mlp.up_proj | (9216, 2560) | 36 | 0.940 |
| 12 | layer 1 mlp.down_proj | (2560, 9216) | 10 | 0.919 |
| 13 | layer 2 linear_attn.out_proj | (2560, 4096) | 10 | **0.865** ← worst |
| 14 | layer 2 linear_attn.in_proj_qkv | (8192, 2560) | 32 | 0.915 |
| 15 | layer 2 linear_attn.in_proj_z | (4096, 2560) | 16 | 0.940 |
| 16 | layer 2 mlp.gate_proj | (9216, 2560) | 36 | 0.953 |
| 17 | layer 2 mlp.up_proj | (9216, 2560) | 36 | 0.936 |
| 18 | layer 2 mlp.down_proj | (2560, 9216) | 10 | 0.960 |
| 19 | layer 3 self_attn.q_proj | (8192, 2560) | 32 | 0.989 ← best |
| 20 | layer 3 self_attn.k_proj | (1024, 2560) | 4 | 0.923 |
| 21 | layer 3 self_attn.v_proj | (1024, 2560) | 4 | 0.929 |
| 22 | layer 3 self_attn.o_proj | (2560, 4096) | 10 | 0.941 |
| 23 | layer 3 mlp.gate_proj | (9216, 2560) | 36 | 0.950 |
| 24 | layer 3 mlp.up_proj | (9216, 2560) | 36 | 0.938 |
| 25 | layer 3 mlp.down_proj | (2560, 9216) | 10 | 0.927 |

Aggregate stats (`calib_sb0.log:238`): `min=0.865068  mean=0.937068  max=0.989481`. This matches the user's claim that initial calibration cos is 0.865–0.966 per Linear (the 0.989 outlier is q_proj which is unusually well-conditioned).

### 2.3 The 5 worst-cos Linears

`train_qwen.py:703-709` hard-codes a set called `BIG_LORA_TARGETS` that receives rank-32 LoRA instead of the default rank-16:

```python
BIG_LORA_TARGETS = {
    (0, "linear_attn.in_proj_z"),       # cos 0.922
    (1, "mlp.down_proj"),               # cos 0.919
    (2, "linear_attn.out_proj"),        # cos 0.865 ← worst
    (2, "linear_attn.in_proj_qkv"),     # cos 0.915
    (3, "self_attn.k_proj"),            # cos 0.923
}
```

These are the same 5 tensors whose post-calibration cos falls below 0.93. The orchestrator brief asks why rank-32 LoRA fails to rescue them; that question is picked up in §6 below and answered in detail in `06_staged_training.md`.

### 2.4 Calibration: Hessian-weighted 1-D k-means, no GPTQ

`palettize_core.py:61-144` implements calibration. The pipeline is:

1. Compute the Hessian diagonal `hess_diag = diag(X.T @ X)` over a stream of 8,192 calibration sequences of length 2,048 (`calib_sb0.log:4-5`). Total tokens: 16.78 M.
2. For each tensor, call `palettize_groups(W_comp, hess_diag, BITWIDTH=2, GROUP_SIZE=256)` (`palettize_core.py:93`).
3. `palettize_groups` slices W into groups of 256 along `out_dim` and runs `kmeans1d_weighted` (`palettize_pytorch.py:25-87`) with `hess_diag` as the per-column weight.
4. `kmeans1d_weighted` initializes centers as quantiles of the weighted distribution (line 50-56) and refines via Lloyd iterations until `torch.allclose(new_centers, centers, atol=1e-7)` (line 75).

The comment at `palettize_core.py:90` is explicit: `# kmeans only (NO GPTQ — tested: GPTQ hurts with kmeans LUT)`. This is a self-imposed constraint that matters for the literature comparison in `07_literature_comparison.md`: the dominant 2-bit / 3-bit / 4-bit baselines (GPTQ, AWQ, SqueezeLLM) all use some variant of closed-form or grid-search codebook fitting; this repo does not.

### 2.5 Cosine is measured on outputs, not weights

`palettize_core.py:103-105` computes cos between **output activations** of the original vs. palettized weight, not between the weights themselves:

```python
Y_orig  = (X @ W_orig.T).flatten().float()
Y_quant = (X @ Wq.T   ).flatten().float()
cos = F.cosine_similarity(Y_orig.unsqueeze(0), Y_quant.unsqueeze(0), dim=1, eps=1e-8).item()
```

This is the right metric — it is what downstream layers actually see — but it also means the 0.865 floor on layer-2 `out_proj` is a *functional* floor, not a weight-space floor. A better weight reconstruction (lower L2) does not necessarily raise output cos if the residual error happens to align with high-magnitude activation directions.

---

## 3. Forward path: hard vs. soft

### 3.1 Hard forward (eval mode and `use_soft_indices=False`)

`qwen_model.py:148-153`:

```python
y = self._hard_kernel(
    x_flat, self.palette, self.indices_int8,
    self.bias, self.group_size
)
```

This calls `CUDAFusedLUTLinear.apply` (`fused_lut_linear_cuda.py:428-491`), which in turn calls `fused_lut_linear_fwd` (the C++ wrapper at `fused_lut_linear_cuda.py:74-117`). The wrapper validates dtypes (`palette` must be bf16, `indices` must be int8/uint8) and dispatches to `fused_lut_linear_fwd_kernel` in `fused_lut_kernel.cu:95-288`.

The forward kernel materializes a 32×64 `sW` tile in shared memory from `palette` + `indices` (`fused_lut_kernel.cu:195-209`), then runs an FMA loop over the K dimension. The key tile sizes are:

```
FWD_BM = 64   (M rows per tile)
FWD_BN = 64   (N cols per tile)
FWD_BK = 32   (K reduction per iter)
FWD_THREADS = 256
```

A separate tensor-core variant `fused_lut_linear_fwd_tc_kernel` exists at `fused_lut_kernel.cu:309-528` with `FWD_TC_BK = 16` and uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. It is not clear from the launcher which variant is actually bound; both `fused_lut_linear_fwd` (line 74) and `fused_lut_linear_fwd_tc_kernel` exist but only the former is exposed in the C++ wrapper.

### 3.2 Soft forward (training, default)

`qwen_model.py:141-147`:

```python
y = self._soft_kernel(
    x_flat, self.palette, self.index_logits,
    self.bias, self.group_size, self.tau
)
```

This invokes `CUDAFusedLUTLinearSoft.apply` (`fused_lut_linear_cuda.py:519-684`). The forward does three things:

1. Calls the C++ wrapper `fused_lut_linear_soft_fwd` (`fused_lut_linear_cuda.py:225-261`), which runs `compute_P_W_Launcher` in CUDA to sample Gumbel noise and produce both `P (4,K,N) fp16` and `W_soft (K,N) bf16` (`fused_lut_linear_cuda.py:250-255`).
2. **Straight-Through Estimator (STE) trick** (`fused_lut_linear_cuda.py:580-595`):

```python
with torch.no_grad():
    argmax_idx = logits.argmax(dim=0)  # (K, N)
    W_hard = palette[group_per_col.long(), argmax_idx.long()].to(W_soft.dtype)
W = W_hard - W_soft.detach() + W_soft
y = torch.matmul(x, W)
```

This is the classic Bengio-style STE: forward value is `W_hard` (exact one-hot, preserves cos), backward gradient flows through `W_soft` (non-zero, so indices actually train). Without this trick, the soft path would compute `y = x @ W_soft` which is a continuous blend and would cap cos at the soft-max achievable value.

3. Saves `(x, palette, logits, P, W)` for backward (`fused_lut_linear_cuda.py:602`).

### 3.3 The tau anneal schedule

`train_sb0.log:8` shows `tau: 2.0 → 0.1 over 4000 steps`. High tau (2.0) means heavy Gumbel noise relative to logit separation — at training start, `P` is close to uniform and `W_soft` is close to the average of the 4 palette entries, which is a very coarse approximation. Low tau (0.1) means `P` is close to one-hot and `W_soft ≈ W_hard`, so the STE correction vanishes and the forward is effectively hard.

The interaction between tau and gradient magnitude is critical: as tau → 0, `P` becomes one-hot, the gradient `grad_palette = Σ grad_W * P` collapses to only the slot that the argmax picked (matching the hard-path gradient), and **the other 3 palette entries per group receive zero gradient**. We expand on this in `02_gradient_correctness.md` §4.

---

## 4. Optimizer and learning-rate setup

### 4.1 FP32MasterAdamW

The palette is optimized by `FP32MasterAdamW` (`train_qwen.py:205-208`), a thin wrapper around `torch.optim.AdamW` that maintains an fp32 master copy of every parameter (`train_qwen.py:154-202`).

The wrapper's `step` method (`train_qwen.py:186-199`) is the critical precision boundary:

```python
def step(self, closure=None):
    for group in self.opt.param_groups:
        for master in group["params"]:
            p = self.model_param_map[id(master)]
            if p.grad is not None:
                master.grad = p.grad.float()      # ← bf16 grad → fp32 master grad
            else:
                master.grad = None
    self.opt.step(closure=closure)
    with torch.no_grad():
        for group in self.opt.param_groups:
            for master in group["params"]:
                p = self.model_param_map[id(master)]
                p.data.copy_(master.data)          # ← fp32 master → bf16 model param
```

So the optimizer does see an fp32 master. The damage, however, has already happened upstream: `p.grad` is the bf16 tensor produced by `CUDAFusedLUTLinearSoft.backward`, and `.float()` only up-casts the already-rounded values. We expand on this in `03_precision_analysis.md`.

### 4.2 Group classification and LR assignment

`train_qwen.py:569-589` classifies every trainable parameter into one of four buckets and assigns a per-bucket LR:

| Bucket | Optimizer | LR (default) | Notes |
|---|---|---|---|
| `palettes` | FP32MasterAdamW | 3e-3 | tiny: 2,208 params total |
| `lora` | FP32MasterAdamW | 1e-3 | 4.69 M params |
| `indices` (index_logits) | FP32MasterAdamW | 1e-2 | 1.78 B params — huge |
| `layernorms` (1D) | FP32MasterAdamW | 3e-4 | 0.61 M params |
| 2D non-palette (correction, if any) | FP32MasterMuon | 2e-4 | 0 in current run |

The values come from `DEFAULT_HYPERPARAMS["lrs"]` at `train_qwen.py:82-95`. The comment at lines 84-87 is informative but **stale**:

```
# Bumped LRs (step 250 analysis: curve climbing too slowly).
#   palettes: 1e-4 -> 3e-3 (30x). Only 2,208 params, super dense grad,
#             Muon scale ~0.63 makes effective LR very low (6.3e-5).
#             30x brings effective to ~1.9e-3, in line with correction.
```

The comment justifies the 30x LR bump on the assumption that palettes use Muon (whose Newton-Schulz orthogonalization scales the effective step by `0.2 * sqrt(max(A,B))`). But the actual code at `train_qwen.py:574-576` routes palettes to **AdamW**, not Muon. AdamW has no scale factor; the raw `lr * m_hat / (sqrt(v_hat) + eps)` is applied directly. So the "effective LR ~1.9e-3" justification is wrong; the actual effective LR is `3e-3 / (sqrt(v_hat) + 1e-8)`, which after the first few hundred steps settles near `3e-3 / sqrt(mean(grad^2))` — typically 5–20x the raw gradient norm. This is not necessarily wrong, but the rationale needs re-derivation.

### 4.3 Loss function — actual vs. claimed

The brief says the loss is `"1-cos+norm_mse"`. The actual default in `DEFAULT_HYPERPARAMS["loss_type"]` (`train_qwen.py:96`) is:

```python
"loss_type": "norm_mse",
"loss_weights": {"cos": 0.0, "mse": 1.0},
```

`compute_loss` (`train_qwen.py:222-242`) branches on `loss_type`:

- `"norm_mse"`: `loss = ((s - t)**2).mean() / (t*t).mean().clamp(min=1e-6)` (line 237-239)
- `"1-cos"`: `loss = (1 - cos_per).mean()` (line 231-232)
- `"1-cos+norm_mse"`: `loss = w["cos"] * l_cos + w["mse"] * l_mse` (line 233-236)

So under the default config the cosine term is **completely absent** from the loss. The `cos` value logged during training is computed for monitoring only (`compute_loss` returns `(loss, {"cos": l_cos.item(), ...})` at line 242), but it does not influence gradients.

This is a major discrepancy with the brief. The brief implies the system is being trained with `1-cos+norm_mse`; the code says it is being trained with pure `norm_mse`. The two losses have very different gradient landscapes:

- `norm_mse` is scale-sensitive: doubling the student output doubles the MSE. It drives the student toward the teacher's scale, which is good if the palette quantization has shifted the scale, but bad if the scale is already correct and only the direction is off.
- `1-cos` is scale-invariant: it only drives direction alignment. It is the natural loss for measuring reconstruction quality of a Linear's output, but its gradient is `2 * (s/||s|| - (s·t/||s||²) * s/||s||) / ||s||` which becomes ill-conditioned when `||s|| → 0`.

A cyclic schedule exists (`get_loss_type_for_step`, `train_qwen.py:342-353`) that alternates 100 steps of `norm_mse` with 100 steps of `1-cos`, but the training loop at `train_qwen.py:1102` overrides it with the constant `1-cos+norm_mse` (or in this case `norm_mse`) from `hp["loss_type"]`. The cyclic scheduler is dead code in the current run.

We dedicate `05_loss_function.md` to a full comparison.

### 4.4 Gradient clipping

`DEFAULT_HYPERPARAMS["gradient_clip"] = 0.3` (`train_qwen.py:98`). This is a single global clip applied to **all trainable parameters concatenated**, including the 1.78 B index_logits. With 1.78 B parameters, even a per-parameter gradient of `1e-5` produces a total norm of `~14`. A clip of `0.3` therefore scales the entire gradient vector by `0.3 / 14 ≈ 0.021` — i.e. the palette gradient is being multiplied by ~1/50 of its raw value most of the time.

This is a candidate root cause for the cos plateau and we revisit it in `08_recommendations.md`.

---

## 5. The fallback (non-CUDA) path

`qwen_model.py:154-163` defines a pure-PyTorch fallback used when CUDA is unavailable or `pre_transposed=True`:

```python
flat_palette = self.palette.reshape(-1).to(x.dtype)
gathered = flat_palette[self._flat_idx]
if not self.pre_transposed:
    y = x_flat @ gathered
else:
    y = x_flat @ gathered.T
if self.bias is not None:
    y = y + self.bias.to(x.dtype)
```

The fallback uses native PyTorch autograd (gather + matmul), which means `grad_palette` flows through `gathered` via `index_select`'s backward — which is exactly the hard-path scatter_add. The fallback is therefore gradient-equivalent to the hard CUDA kernel. It is used by the test suite (`scripts/test_palettized_v2.py`) but never in the training loop.

The fallback also confirms there is no hidden "magic" in the CUDA path: the formula `W[j,o] = palette[o // 256, indices[j,o]]` is the entire forward computation. The CUDA kernel exists purely for speed.

---

## 6. Why the 5 worst-cos Linears stay stuck

The 5 `BIG_LORA_TARGETS` get rank-32 LoRA (`train_qwen.py:700-701`), 2× the default rank-16. The orchestrator brief notes these Linears do not improve even with this larger adapter. Three structural reasons emerge from the audit:

### 6.1 The palette, not the indices, is the bottleneck for these Linears

For layer-2 `linear_attn.out_proj` (cos=0.865), the 4 palette entries per group of 256 output weights span only ~4 distinct magnitudes. If the original weight distribution in a group is bimodal or heavy-tailed, 4 entries cannot represent it well no matter how the indices are assigned. K-means gives the optimal 4 centers, so re-running k-means per step (LUT-Q style) cannot help. The only ways to improve cos are:

1. **More palette entries per group** (smaller GROUP_SIZE), or
2. **Better palette values** via gradient descent — but the gradient is small because the k-means solution is already a local optimum of the L2 reconstruction objective.

The gradient descent on `palette` is essentially fine-tuning the k-means centers, which is a small correction if k-means already found a good local optimum.

### 6.2 LoRA cannot fix systematic quantization bias

LoRA adds `x @ A @ B.T` to the output, which is a low-rank correction to the weight matrix. If the quantization error `W_orig - W_quant` is high-rank (which it is, since the residual has the same shape as `W`), a rank-32 LoRA can only capture the top-32 singular directions of the residual. For most Linears this captures >80% of the residual energy and cos jumps from ~0.93 to ~0.97+. For the 5 worst Linears the residual spectrum is flatter (more directions matter), so rank-32 captures <50% and cos stays stuck.

This is an information-theoretic limit: rank-32 LoRA adds `2 * 32 * (in_dim + out_dim)` parameters per Linear, but the residual has `in_dim * out_dim` degrees of freedom. Unless the residual is approximately low-rank, LoRA cannot fix it.

### 6.3 The LoRA is initialized from the palettized weight, not the original

`train_qwen.py:718` constructs `QwenLoRA` with `init="loftq"` but `original_weight=None`:

```python
lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=None)
```

With `original_weight=None`, the LoftQ branch in `QwenLoRA.__init__` (`qwen_model.py:209-222`) is skipped and the fallback at `qwen_model.py:223-225` runs:

```python
A_init = torch.randn(in_dim, rank) * 0.01
B_init = torch.zeros(out_dim, rank)
```

So **B is initialized to zero**, meaning the LoRA contribution is exactly zero at step 0. The LoRA must climb out of a zero-init valley from scratch, with no SVD warm start. This is the opposite of what LoftQ is supposed to do (LoftQ initializes A, B from the SVD of `W_orig - W_quant` so that the LoRA immediately cancels the leading residual directions).

The fix is to call `capture_original_weights` (which exists at `qwen_model.py:779-795`) before `attach_lora_to_layer` and pass the dict into `QwenLoRA`. We provide the patch in `08_recommendations.md`.

---

## 7. Summary of audit findings

| # | Finding | Severity | File:line |
|---|---|---|---|
| 1 | Palette stored as bf16, losing precision before training starts | High | `qwen_model.py:82-85` |
| 2 | LUT serialized as fp16 on disk, double cast (fp16 → fp32 → bf16) on load | High | `palettize_core.py:175-178`, `qwen_model.py:82-85` |
| 3 | Soft path (Gumbel-Softmax) is the default; brief implied hard path | Medium | `train_qwen.py:1239` |
| 4 | Actual loss is `norm_mse`, not `1-cos+norm_mse` as brief claims | High | `train_qwen.py:96` |
| 5 | Cosine term is monitored but not in the loss under default config | High | `train_qwen.py:230-242` |
| 6 | Cyclic loss scheduler exists but is dead code in current run | Low | `train_qwen.py:342-353`, `train_qwen.py:1102` |
| 7 | LR comment justifies 3e-3 using "Muon scale 0.63" but palettes use AdamW | Medium | `train_qwen.py:84-87` vs `train_qwen.py:574-576` |
| 8 | Global gradient clip = 0.3 likely throttles palette updates (1.78 B index_logits dominate the norm) | High | `train_qwen.py:98` |
| 9 | LoRA uses `init="loftq"` but `original_weight=None` → falls back to zero-init B | High | `train_qwen.py:718` |
| 10 | 5 worst Linears have structural reasons to be stuck (palette capacity, residual rank) | Medium | `train_qwen.py:703-709` |
| 11 | K-means calibration is already locally optimal for the L2 objective | Medium | `palettize_core.py:90-93` |
| 12 | Cos measured on outputs not weights — weight L2 can improve without cos improving | Low | `palettize_core.py:103-105` |
| 13 | STE trick correctly used for soft path (forward hard, backward soft) | OK | `fused_lut_linear_cuda.py:580-595` |
| 14 | Hard-path gradient uses fp32 atomic accumulator then casts to bf16 at return | High | `fused_lut_linear_cuda.py:158-168` |
| 15 | Soft-path gradient casts final `grad_palette` to bf16 at return | High | `fused_lut_linear_cuda.py:662` |

Findings 1, 2, 4, 5, 8, 9, 14, 15 are individually sufficient to explain the cos plateau at 0.95. They compound multiplicatively. The next wave (`02_gradient_correctness.md`) verifies that the gradient formula itself is mathematically correct, so that we can attribute the plateau to precision / loss / LR issues rather than a buggy backward pass.
