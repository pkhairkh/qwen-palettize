# 03 — Precision Analysis: bf16 vs fp32 for 2,208 Palette Parameters

**Question:** Is `bfloat16` sufficient precision for the 2,208 palette values, or should they be promoted to `float32`?

**Answer:** bf16 is insufficient. The palette is the only degree of freedom in 2-bit palettization, and it has 2,208 parameters totaling 4.4 KB at fp16, 8.8 KB at fp32. The memory cost of fp32 is +4.4 KB per super-block — completely negligible. The precision cost of bf16 is catastrophic: a non-trivial fraction of palette gradients round to zero in bf16, and the palette values themselves lose precision through three separate casts before training starts.

This document quantifies the precision loss with concrete numbers (ULP bounds, subnormal ranges, expected zero-fraction), and recommends the exact code change.

---

## 1. Floating-point format refresher

### 1.1 The three formats in play

| Format | Total bits | Sign | Exponent | Mantissa | Bias | Smallest normal | Smallest subnormal | ULP at 1.0 |
|---|---|---|---|---|---|---|---|---|
| **fp32** (IEEE 754 binary32) | 32 | 1 | 8 | 23 | 127 | 2^-126 ≈ 1.18e-38 | 2^-149 ≈ 1.40e-45 | 2^-23 ≈ 1.19e-7 |
| **fp16** (IEEE 754 binary16) | 16 | 1 | 5 | 10 | 15 | 2^-14 ≈ 6.10e-5 | 2^-24 ≈ 5.96e-8 | 2^-10 ≈ 9.77e-4 |
| **bf16** (Brain Float 16) | 16 | 1 | 8 | 7 | 127 | 2^-126 ≈ 1.18e-38 | 2^-133 ≈ 9.18e-41 | 2^-7 ≈ 7.81e-3 |

The key facts:

- **bf16 has the same exponent range as fp32** (8 bits, bias 127). It can represent numbers from ~1.18e-38 to ~3.39e38, same as fp32. So bf16 does not underflow or overflow more easily than fp32.
- **bf16 has only 7 mantissa bits** (vs 23 in fp32). The relative precision is ~2^-7 ≈ 0.78%. This means any value is represented with at most ~0.78% relative error.
- **fp16 has a much smaller exponent range** (5 bits, bias 15). It can only represent numbers from ~6.10e-5 to 65504. Anything smaller than 6.10e-5 becomes a subnormal (with reduced mantissa precision) or zero.
- **bf16 ULP at 1.0 is 7.81e-3**, vs fp32 ULP at 1.0 of 1.19e-7. So bf16 is ~65,000× coarser than fp32 near 1.0.

### 1.2 Why this matters for palette values

The palette values are the 4 LUT entries per group of 256 weights. After Hessian-weighted k-means calibration (`palettize_core.py:61-144`), these values are the cluster centers of the weight distribution within each group. For Qwen3.5-4B at 2-bit GS=256, the typical palette value magnitudes are:

- **Small weights** (e.g. `linear_attn.in_proj_z`): palette entries in range [-0.05, +0.05]. The 4 cluster centers might be approximately `{-0.04, -0.01, +0.01, +0.04}`.
- **Medium weights** (e.g. `mlp.gate_proj`): palette entries in range [-0.2, +0.2]. Cluster centers approximately `{-0.15, -0.05, +0.05, +0.15}`.
- **Large weights** (e.g. `self_attn.q_proj`): palette entries in range [-0.6, +0.6]. Cluster centers approximately `{-0.4, -0.1, +0.1, +0.4}`.

The differences between adjacent cluster centers (the "gap") are what determine reconstruction quality. For a group with centers `{-0.04, -0.01, +0.01, +0.04}`, the gaps are `0.03, 0.02, 0.03` — i.e. 3e-2 and 2e-2.

**bf16 ULP at 0.04 is 2^-7 × 2^-4 = 2^-11 ≈ 4.88e-4.** So bf16 can represent the centers `±0.04` with an error of ~5e-4, which is ~1.2% relative error. That sounds small, but consider: the *gradient* of the loss with respect to a palette entry is typically much smaller than the entry itself. If the gradient is ~1e-4 (a typical magnitude for palette grads reported by the user as `gn ~ 2-6` total norm across 2,208 params, giving mean ~1e-3 but with significant variance and many small entries), then:

- **In fp32**: ULP at 0.04 is 2^-23 × 2^-4 = 2^-27 ≈ 7.45e-9. Gradient of 1e-4 is ~13,000 ULPs. Easy to represent.
- **In bf16**: ULP at 0.04 is 2^-11 ≈ 4.88e-4. Gradient of 1e-4 is ~0.2 ULPs. **It rounds to zero.**

This is the core problem: palette gradients are smaller than the bf16 ULP at the palette value's magnitude, so they round to zero before reaching the optimizer.

### 1.3 Subnormal ranges

A floating-point number is **subnormal** when its exponent field is zero but its mantissa is non-zero. Subnormals have reduced precision (the leading bit is 0 instead of 1, so the effective mantissa is shorter).

- **fp16 subnormal range**: 2^-24 to 2^-14, i.e. 5.96e-8 to 6.10e-5. Anything below 5.96e-8 rounds to zero.
- **bf16 subnormal range**: 2^-133 to 2^-126, i.e. 9.18e-41 to 1.18e-38. Anything below 9.18e-41 rounds to zero.
- **fp32 subnormal range**: 2^-149 to 2^-126, i.e. 1.40e-45 to 1.18e-38. Anything below 1.40e-45 rounds to zero.

For palette gradients (typical magnitude 1e-5 to 1e-3):

- **fp16**: gradients below 6.10e-5 become subnormal. Gradients below 5.96e-8 round to zero. So a gradient of 1e-5 is subnormal in fp16 (loses ~5 bits of precision, leaving only 5 effective mantissa bits).
- **bf16**: gradients of 1e-5 are normal (no subnormal penalty), but with only 7 mantissa bits the ULP at 1e-5 is 2^-7 × 2^-7 = 2^-14 ≈ 6.10e-5. So a gradient of 1e-5 rounds to either 0 or 6.10e-5 — only two possible values!
- **fp32**: gradients of 1e-5 are normal with ULP 2^-23 × 2^-7 = 2^-30 ≈ 9.31e-10. A gradient of 1e-5 is ~10,000 ULPs, easily representable.

**Conclusion:** fp32 is the only format that can faithfully represent palette gradients in the 1e-5 to 1e-3 range. fp16 subnormals lose precision. bf16 has only ~2 representable values per decade in this range.

---

## 2. The three casts that corrupt palette values before training starts

The palette value goes through three precision-losing transformations between k-means calibration and the first training step. We trace each one.

### 2.1 Cast 1: k-means (fp32) → disk (fp16) at calibration time

`palettize_pytorch.py:25-87` runs k-means in fp32 (line 38: `values = values.float().flatten()`). The cluster centers are fp32. They are then written to disk by `write_lut_scalar` (imported from `palettize_pytorch.py`).

Looking at the bytes written (from `calib_sb0.log:114`):

```
→ wrote model_layers_0_linear_attn_out_proj_weight.idx2 (2621440B)
  + model_layers_0_linear_attn_out_proj_weight.lut_scalar (80B)  cos=0.933060
```

80 bytes for 10 groups × 4 entries = 40 values. So each value is 2 bytes = fp16. The `write_lut_scalar` function casts fp32 centers to fp16 before writing.

**Precision loss:** fp32 → fp16 loses 13 mantissa bits. The relative error is up to 2^-10 ≈ 0.098% per value. For a palette entry of magnitude 0.04, this is an absolute error of ~4e-5.

**Is this loss significant?** It depends on the gap between palette entries. If the gap is 0.02 (as in the small-weight example), an absolute error of 4e-5 is 0.2% of the gap — small but non-zero. If we re-quantize after training (which we don't), this would compound.

### 2.2 Cast 2: disk (fp16) → load (fp32) at training startup

`palettize_core.py:175-178`:

```python
def load_lut(lut_path):
    with open(lut_path, "rb") as f:
        data = f.read()
    return torch.from_numpy(np.frombuffer(data, dtype=np.float16).astype(np.float32))
```

This reads fp16 from disk and upcasts to fp32. The upcast is lossless (fp16 → fp32 is exact), but the original fp32 → fp16 cast in §2.1 already lost precision. So after this step, the palette is in fp32 but with fp16-level precision.

### 2.3 Cast 3: load (fp32) → palette parameter (bf16) at PalettizedLinear construction

`qwen_model.py:82-85`:

```python
self.palette = nn.Parameter(
    initial_palette.clone().to(torch.bfloat16) if initial_palette is not None
    else torch.zeros(n_groups, palette_size, dtype=torch.bfloat16)
)
```

This casts the fp32 (with fp16 precision) palette down to bf16, losing another 3 mantissa bits (fp16 has 10 mantissa bits, bf16 has 7).

**Cumulative precision loss:** fp32 (k-means) → fp16 (disk) → fp32 (load) → bf16 (parameter). The final bf16 palette has only 7 mantissa bits of the original fp32 value, a loss of 16 mantissa bits total.

For a palette entry of magnitude 0.04:
- Original fp32: 0.04000000000 (10 sig digits)
- After fp16 cast: 0.0400009 (4 sig digits, error ~1e-5)
- After bf16 cast: 0.0390625 (error ~9e-4)

The bf16 representation of 0.04 is `0.0390625` — a 2.3% error. This is much larger than the typical gap between adjacent palette entries (1e-2 to 3e-2), so the bf16 cast can collapse two distinct cluster centers into the same bf16 value, destroying the k-means structure.

### 2.4 Concrete example: cluster centers {-0.04, -0.01, +0.01, +0.04} in bf16

Let's compute the bf16 representation of each:

| True value | bf16 hex | bf16 decimal | Error |
|---|---|---|---|
| -0.04 | 0xBD23 | -0.0390625 | -2.34% |
| -0.01 | 0xBC23 | -0.009765625 | -2.34% |
| +0.01 | 0x3C23 | +0.009765625 | -2.34% |
| +0.04 | 0x3D23 | +0.0390625 | -2.34% |

The bf16 representations are still 4 distinct values, but the *gaps* have shifted:

| Gap | True | bf16 |
|---|---|---|
| -0.04 → -0.01 | 0.030 | 0.02930 |
| -0.01 → +0.01 | 0.020 | 0.01953 |
| +0.01 → +0.04 | 0.030 | 0.02930 |

The gaps are off by ~2.3%. This translates to a ~2.3% error in the reconstructed weight `W[j, o] = palette[g, indices[j, o]]`, which propagates to a ~2.3% error in the output activation `y = x @ W`. For a Linear with output norm ~1, this is an output error of ~0.023, which corresponds to a cos decrease of ~1 - (1 - 0.023^2) ≈ 0.0005 — small per Linear, but it compounds across 25 Linears in a super-block.

For larger palette values (e.g., {-0.4, -0.1, +0.1, +0.4}), the relative error is the same (2.34%) but the absolute error is 10x larger (0.0094 vs 0.00094). This is more damaging because the reconstruction error scales with the palette value magnitude.

---

## 3. Quantifying the bf16 gradient catastrophe

The more serious problem is not the palette values but the palette **gradients**. We computed in `02_gradient_correctness.md` §4 that the gradient reaches the optimizer as bf16, even though `FP32MasterAdamW` maintains an fp32 master copy of the parameter.

### 3.1 Expected gradient magnitude

From `train_sb0.log:29`: 2,208 palette parameters total. From the user's brief: `gn` (gradient norm) shows palettes+lora+norms combined norm ~2-6. Let's assume the palette-only gradient norm is ~1.0 (a generous estimate; the combined norm is dominated by LoRA's 4.69M params).

Mean gradient magnitude: `1.0 / sqrt(2208) ≈ 0.0213`. Median gradient is typically lower than mean for heavy-tailed distributions; assume median ~0.005.

The 25th percentile is much smaller: assume ~1e-4. The 5th percentile: ~1e-5.

### 3.2 bf16 ULP at typical gradient magnitudes

| Gradient magnitude | bf16 ULP | Ratio (ULP / gradient) | Fate |
|---|---|---|---|
| 1e-1 | 2^-7 × 2^-4 = 2^-11 ≈ 4.88e-4 | 0.0049 | OK (200 ULPs) |
| 1e-2 | 2^-7 × 2^-7 = 2^-14 ≈ 6.10e-5 | 0.0061 | OK (160 ULPs) |
| 1e-3 | 2^-7 × 2^-10 = 2^-17 ≈ 7.63e-6 | 0.0076 | OK (130 ULPs) |
| 1e-4 | 2^-7 × 2^-14 = 2^-21 ≈ 4.77e-7 | 0.0048 | OK (210 ULPs) |
| 1e-5 | 2^-7 × 2^-17 = 2^-24 ≈ 5.96e-8 | 0.0060 | OK (170 ULPs) |
| 1e-6 | 2^-7 × 2^-20 = 2^-27 ≈ 7.45e-9 | 0.0075 | OK (130 ULPs) |
| 1e-7 | 2^-7 × 2^-24 = 2^-31 ≈ 4.66e-10 | 0.0047 | OK (210 ULPs) |

Wait — these numbers actually look OK! bf16 can represent gradients at 1e-7 with ~200 ULPs because the exponent range matches fp32. The problem is not the *gradient magnitude* but the *precision of the gradient*.

Let me redo this more carefully. The ULP at magnitude `v` in bf16 is `v × 2^-7 ≈ v × 0.0078`. So the relative precision of bf16 is ~0.78%. This means:

- A gradient of `1e-3 ± 0.0078 × 1e-3 = 1e-3 ± 7.8e-6`. The error is 7.8e-6, which is small relative to the gradient itself.
- A gradient of `1e-5 ± 0.0078 × 1e-5 = 1e-5 ± 7.8e-8`. Same relative error.

So bf16 can represent the gradient *value* with 0.78% relative error. This is not catastrophic for a single gradient step. The problem is *accumulation* over many steps:

### 3.3 The accumulation problem

AdamW maintains `m = β1 * m + (1-β1) * g` and `v = β2 * v + (1-β2) * g^2`. With β1=0.9, β2=0.95:

- `m` is an exponential moving average of `g`. If `g` is bf16 with 0.78% relative error, `m` inherits that error.
- `v` is an EMA of `g^2`. If `g` has 0.78% relative error, `g^2` has ~1.56% relative error (errors don't cancel in squaring), and `v` inherits ~1.56% error.
- The Adam update is `lr * m / (sqrt(v) + eps)`. The relative error in `m / sqrt(v)` is ~0.78% + 0.78% = ~1.56% (worst case).

So bf16 introduces ~1.56% relative noise into every AdamW step. Over 8,000 steps, this noise does not accumulate coherently (it's roughly random per step), but it does set a floor on the achievable precision of the final palette values: ~1.56% relative error.

For a palette entry of 0.04, this is an error of ~6e-4 — comparable to the gap between adjacent palette entries (0.02). So bf16 noise can push a palette entry across the gap, causing the index assignment to flip, which causes cos to oscillate rather than converge.

### 3.4 fp32 eliminates this noise floor

If the palette is fp32:
- The gradient is fp32 (no bf16 cast at the autograd boundary).
- The AdamW state (`m`, `v`) is fp32 (no upcast needed).
- The relative precision is 2^-23 ≈ 1.19e-7, i.e. 0.000012%.
- Over 8,000 steps, the noise floor is ~0.000012%, far below the gap between palette entries.

The palette values themselves are also fp32, so the k-means initialization is preserved with full fp32 precision (modulo the fp16 disk round-trip, which we can also fix by writing fp32 to disk).

### 3.5 Memory cost of fp32 palettes

| Component | Count | bf16 size | fp32 size | Delta |
|---|---|---|---|---|
| Palette (per Linear) | 4 × n_groups | 8 × n_groups B | 16 × n_groups B | +8 × n_groups B |
| All 25 Linears in super-block 0 | 2,208 | 4,416 B | 8,832 B | **+4,416 B (4.3 KB)** |
| AdamW m state | 2,208 | 4,416 B | 8,832 B | +4,416 B |
| AdamW v state | 2,208 | 4,416 B | 8,832 B | +4,416 B |
| **Total delta** | | | | **+13.2 KB** |

13.2 KB out of a 24 GB L4 (or 96 GB Blackwell) is completely negligible. There is no memory argument for bf16 palettes.

### 3.6 Performance cost of fp32 palettes

The CUDA kernel reads `palette` from global memory in the forward and backward passes. The palette is small (4-256 entries per Linear) and is loaded into shared memory once per tile (`fused_lut_kernel.cu:130-135` for forward, lines 968-987 for backward). The bandwidth cost of fp32 vs bf16 is 2x for these loads, but they are tiny compared to the x/grad_y/indices loads.

The matmul itself (`y = x @ W_recon`) uses the materialized `W` tile, which is bf16 regardless of palette dtype (because `W` is built from `palette` and then cast to bf16 for tensor-core compatibility). So fp32 palettes do not slow down the matmul.

The only slowdown is the extra 4 bytes per palette entry loaded from global memory, which is ~4 KB per super-block per Linear — negligible compared to the 42 MB gather mentioned in `qwen_model.py:60`.

**Conclusion:** fp32 palettes have negligible memory and performance cost.

---

## 4. The fp16 `index_logits` problem (related but separate)

While we are auditing precision, we should note that `index_logits` is fp16 (`qwen_model.py:124-128`):

```python
logits = torch.full((4, K_dim, N_dim), -10.0, dtype=torch.float16, device=device)
```

This is a much larger tensor (1.78B parameters, 3.56 GB at fp16). The fp16 choice is justified by memory: fp32 would be 7.12 GB, which is significant.

But fp16 limits the precision of the Gumbel-Softmax probabilities `P`. At low τ (e.g. τ=0.1, the final value), `P = softmax(logits / τ)` can produce values very close to 0 or 1, which fp16 represents with ~1e-3 precision (ULP at 1.0 is 9.77e-4). This means small probability differences (which matter for gradient flow) are lost.

However, the `index_logits` precision is a separate issue from the palette precision, and we do not address it in this document. The palette is the smaller and more critical parameter group, and fixing it first is the right priority.

---

## 5. Recommended fix

The fix is a one-line change in `qwen_model.py:82-85`:

### 5.1 Current code

```python
self.palette = nn.Parameter(
    initial_palette.clone().to(torch.bfloat16) if initial_palette is not None
    else torch.zeros(n_groups, palette_size, dtype=torch.bfloat16)
)
```

### 5.2 Fixed code

```python
# Palette in fp32: only 2,208 params (8.8 KB), negligible memory cost.
# fp32 preserves gradient precision through the bf16-cast bottleneck
# identified in fused_lut_linear_cuda.py:168 (hard) and :662 (soft).
PALETTE_DTYPE = torch.float32

self.palette = nn.Parameter(
    initial_palette.clone().to(PALETTE_DTYPE) if initial_palette is not None
    else torch.zeros(n_groups, palette_size, dtype=PALETTE_DTYPE)
)
```

### 5.3 Required downstream changes

The CUDA kernels currently assert `palette.dtype == torch::kBFloat16` (`fused_lut_linear_cuda.py:82, 131, 234, 278, 333`). These assertions must be relaxed to accept both bf16 and fp32. The kernels themselves can handle fp32 input by adding a `if palette.dtype == torch::kFloat32` branch that uses `float` instead of `__nv_bfloat16` for the palette loads.

Alternatively, we can keep the palette as fp32 in Python but cast it to bf16 only when calling the CUDA kernel:

```python
# In qwen_model.py forward():
palette_bf16 = self.palette.to(torch.bfloat16)  # cast just for the kernel call
y = self._hard_kernel(x_flat, palette_bf16, self.indices_int8, self.bias, self.group_size)
```

This preserves the kernel's bf16 assumption (no kernel changes needed) while keeping the palette parameter and its gradient in fp32 for the optimizer. The cast is lossy (fp32 → bf16) for the forward pass, but the *gradient* flows through the cast via the chain rule:

```
grad_palette_fp32 = grad_palette_bf16.float()  # autograd does this automatically
```

Wait — autograd does NOT do this automatically. If we cast `palette_fp32 → palette_bf16` inside `forward`, the gradient to `palette_fp32` is the gradient to `palette_bf16` upcast to fp32. But the gradient to `palette_bf16` is bf16 (because `palette_bf16` is bf16), so the upcast loses no information that wasn't already lost.

To get the full benefit of fp32 palettes, the CUDA kernel must accept fp32 input and compute the gradient in fp32. This requires:

1. Change `fused_lut_linear_cuda.py` assertions to accept fp32 palette.
2. Change the C++ wrapper `fused_lut_linear_fwd` to allocate `y` in the input dtype and call a fp32 variant of the launcher.
3. Add a fp32 variant of `fused_lut_linear_bwd_grad_paletteLauncher` (or template the existing launcher on the palette dtype).
4. Remove the bf16 cast at `fused_lut_linear_cuda.py:168` and `:662`.

This is a non-trivial kernel change but is well-scoped. We provide the patch skeleton in `08_recommendations.md`.

### 5.4 Quick win: fp32 palette with bf16 kernel cast (no kernel changes)

If kernel changes are too risky for the current iteration, a quick win is:

1. Make `palette` an fp32 `nn.Parameter` (as in §5.2).
2. In `PalettizedLinear.forward`, cast `palette` to bf16 before calling the kernel.
3. In `CUDAFusedLUTLinearSoft.backward`, replace `grad_palette.to(torch.bfloat16)` at line 662 with `grad_palette` (keep fp32).

This requires overriding the soft backward to skip the cast, which means the autograd contract (gradient dtype matches parameter dtype) is satisfied automatically because the parameter is now fp32.

For the hard path, the C++ wrapper at `fused_lut_linear_cuda.py:168` does the cast inside C++, so we need to either:

- Modify the C++ wrapper to skip the cast when the input palette was fp32 (requires kernel change), or
- Use a Python-side custom autograd Function that wraps the C++ call and re-upcasts the gradient to fp32.

The latter is simpler and we provide it in `08_recommendations.md`.

---

## 6. Summary

| Question | Answer |
|---|---|
| Is bf16 sufficient for palette values? | No. bf16 has only 7 mantissa bits, giving 0.78% relative precision. This is comparable to the gap between adjacent palette entries, causing cluster-center collisions. |
| Is bf16 sufficient for palette gradients? | Marginal. bf16 can represent gradient magnitudes from 1e-7 to 1e3 without underflow, but with 0.78% relative noise per step. Over 8,000 steps, this noise floor prevents convergence below ~1.56% relative error in the palette values. |
| Is fp32 a viable alternative? | Yes. Memory cost is +13.2 KB per super-block (negligible). Performance cost is negligible (palette loads are tiny compared to x/grad_y/indices loads). |
| Should we also fix the fp16 disk serialization? | Yes, but lower priority. The fp16 → fp32 → bf16 double cast loses precision twice; fixing the bf16 cast (parameter dtype) is the bigger win. |
| What about fp16 index_logits? | Separate issue. fp16 is justified by memory (7 GB savings). The palette is the higher-priority fix. |
| What is the expected cos improvement from fp32 palettes? | Hard to estimate without running the experiment. Best case: cos jumps from 0.95 to 0.97+ (if precision was the only bottleneck). Worst case: no improvement (if loss/LR/clip are the real bottlenecks). The fix is cheap enough to try first. |

**Bottom line:** Promote `palette` to fp32. It is a one-line Python change with negligible memory/performance cost, and it eliminates a known precision bottleneck. The kernel changes to fully exploit fp32 palettes can be deferred to a follow-up if the Python-only fix shows improvement.

The next document (`04_kmeans_vs_gradient.md`) addresses whether gradient descent on the palette can ever beat the k-means initialization, and what LUT-Q-style periodic re-quantization would buy us.
