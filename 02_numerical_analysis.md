# 02 — Numerical Precision Analysis: fp16 / bf16 / fp32 Behavior

**Scope:** Quantify the precision losses at every stage of the soft forward/backward path, identify overflow/underflow risks, and project their effect on cosine-similarity convergence.

---

## 1. Floating-point format recap

| Format | Exp bits | Mantissa bits | Min normal | Max | Eps (relative) |
|--------|----------|---------------|------------|-----|-----------------|
| fp32   | 8        | 23+1          | 1.18e-38   | 3.40e38 | 1.19e-7 |
| tf32   | 8        | 10+1          | 1.18e-38   | 3.40e38 | 4.88e-4 |
| bf16   | 8        | 7+1           | 1.18e-38   | 3.40e38 | 7.81e-3 |
| fp16   | 5        | 10+1          | 6.10e-5    | 65504   | 4.88e-4 |

**Key observations:**
- bf16 has fp32's exponent range (no overflow risk for normal training magnitudes) but only 8-bit mantissa (~2 decimal digits).
- fp16 has 3 more mantissa bits than bf16 (~3 decimal digits) but only 5 exponent bits — max is 65504, and denormals kick in below 6.1e-5.
- The kernel uses BOTH: bf16 for weights/activations/gradients, fp16 for logits/P/grad_logits. This mix is suboptimal in two ways (fp16's narrow range + bf16's low precision).

---

## 2. Forward path precision: hard kernel (SIMD2 FMA, default)

The hard forward kernel (`fused_lut_linear_fwd_kernel`, lines 95–288) computes:

```
y[m, n] = Σ_k x[m, k] * W[k, n] + bias[n]
```

where `W[k, n] = palette[n / GS, indices[k, n]]` (bf16 lookup, bit-exact).

**Precision per stage:**

| Stage | Operation | Dtype in | Dtype out | Error per element |
|-------|-----------|-----------|-----------|------------------|
| Materialize W | palette lookup | bf16 | bf16 | 0 (bit-exact) |
| Load x | smem load | bf16 | bf16 | 0 |
| FMA inner loop | `x0_f * w0_f` (fp32) | bf16→fp32 | fp32 | 0 (no rounding in bf16→fp32) |
| Accumulate | `acc += ...` | fp32 | fp32 | ~ eps_fp32 * sqrt(K) |
| Output | `__float2bfloat16(acc + bias)` | fp32 | bf16 | ≤ 0.5 ULP bf16 (RNE) |

For K=2560:
- Per-multiply error: 0 (bf16→fp32 conversion is exact, fp32 multiply is RNE).
- Accumulator error after K multiplies: `sqrt(2560) * eps_fp32 ≈ 50 * 1.2e-7 ≈ 6e-6` (assuming uncorrelated errors; pessimistic).
- Output rounding: bf16 has 7.8e-3 relative error, so the final bf16 output dominates the error.
- **Total relative error per output element: ~ 7.8e-3** (bf16 quantization dominates).

**Cosine-similarity impact:** For a (1, N) output vector with i.i.d. errors of magnitude 7.8e-3, the cosine similarity between the bf16 output and the fp32 reference is approximately:

```
cos ≈ 1 - (N/2) * (eps/signal)^2 ≈ 1 - 0.5 * (7.8e-3)^2 ≈ 1 - 3e-5
```

So the hard forward kernel alone should give cos > 0.9999 with the fp32 reference — **the hard kernel is NOT the source of the cos gap.**

---

## 3. Forward path precision: TC kernel (`mma.sync.m16n8k16`)

The TC variant (`fused_lut_linear_fwd_tc_kernel`, lines 309–518) uses:

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

This performs 16×8×16 = 2048 multiplies in a single instruction, with:
- A operand: bf16 (16×16 matrix)
- B operand: bf16 (16×8 matrix)
- C accumulator: fp32 (16×8 matrix)

**Critical difference from scalar path:**
- The multiply `a[i,k] * b[k,j]` is performed as `bf16 × bf16` and the result is rounded to fp32 BEFORE accumulation.
- This loses 8 bits of mantissa relative to the scalar path (which does `bf16→fp32 (exact) × fp32 (full precision)`).
- Per-multiply error: `eps_bf16 = 7.8e-3`.

For K=2560 with `FWD_TC_BK=16`:
- Number of mma iterations: 160.
- Accumulator error growth: `sqrt(2560) * eps_bf16 * ||x|| * ||w|| / ||y||`.
- With ||x|| ≈ ||w|| ≈ 1 (post-LayerNorm) and ||y|| ≈ sqrt(N) ≈ 90:
  - Relative error ≈ 50 * 7.8e-3 / 90 ≈ 4.3e-3.
- For the (1, N) output: cos ≈ 1 - 0.5 * (4.3e-3)^2 ≈ 1 - 9e-6.

**Wait — this is still tiny.** The TC variant's error is comparable to bf16 quantization. The reason is that the K-reduction in fp32 averages out the per-multiply bf16 errors. So **the TC kernel is also NOT the dominant source of the cos gap.**

But there's a subtlety: the mma.sync instruction's rounding mode is **NOT round-to-nearest** — it uses a "stochastic" rounding for some operations (depending on the hardware). On Blackwell (sm_120), the mma.sync.bf16.bf16.f32 instruction uses RNE for the multiply-accumulate, but the input bf16 values are quantized from the source fp32 (via the ldmatrix load). So the effective error is:

- Input quantization: bf16 quantization of x and W (already bf16, so 0).
- Multiply rounding: bf16 × bf16 → fp32 RNE.
- Accumulator: fp32 RNE.

This gives the same ~4e-3 relative error as analyzed above. **TC is fine.**

---

## 4. Soft forward: Gumbel-Softmax precision

The soft forward kernel (`fused_lut_linear_soft_compute_P_W_kernel`, lines 1301–1352) computes:

```
n_k = (logit_k + gumbel_k) / tau
P_k = exp(n_k - max(n)) / sum_k exp(n_k - max(n))
W = Σ_k P_k * palette[g, k]
```

### 4.1 Logit precision (fp16 storage)

`logits` is stored as fp16 (5-bit exp, 10-bit mantissa). For logits in [-10, 10]:
- fp16 representation of 10.0 is exact (within mantissa range).
- fp16 representation of -10.0 is exact.
- The minimum difference between adjacent representable values near 10 is `2^(3-10) = 2^-7 ≈ 0.0078`.

So two logits that differ by < 0.0078 are indistinguishable in fp16. This means the model cannot distinguish between "slightly prefer palette entry k" and "strongly prefer palette entry k" when the logits are near 10.

### 4.2 Gumbel sample precision (fp32 throughout)

The Gumbel sampler (lines 1274–1285) produces a fp32 value in approximately [-3, +16]. After `(logit + gumbel) / tau`:
- At tau=2.0: noisy logit range is approximately [-6.5, 13] (logit ±10 + gumbel ±3..16, divided by 2).
- At tau=0.1: noisy logit range is approximately [-130, 260].
- At tau=0.01 (extreme): range is approximately [-1300, 26000].

The fp32 softmax can handle all these magnitudes (max fp32 ≈ 3.4e38).

### 4.3 Softmax stability (fp32 throughout)

```cuda
float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
float e0 = expf(n0 - m);
// ...
float s = e0 + e1 + e2 + e3;
float p0 = e0 / s, ...;
```

- `m` is the max, so `n_k - m ≤ 0` for all k → `e_k ≤ 1`. No overflow.
- `n_k - m` can be as negative as -260 (at tau=0.1) → `expf(-260) = 0` exactly. So the smallest P values are exactly zero in fp32.
- `s = sum of exp values` is in [1, 4]. No precision loss in the sum.
- `p_k = e_k / s` is in [0, 1], fp32 RNE. No issue.

**At tau=0.1 with logits=±10:**
- After adding Gumbel noise and dividing by tau: max ≈ (10 + 16) / 0.1 = 260.
- After softmax with max-subtraction: max entry = exp(0) = 1, others = exp(-130) = 0, exp(-260) = 0, exp(-390) = 0.
- `s ≈ 1`, `P = [0, 0, 1, 0]` or `[0, 1, 0, 0]` etc. — exactly one-hot.

### 4.4 P storage (fp16) — CRITICAL PRECISION LOSS

```cuda
P[0 * plane_size + idx] = __float2half(p0);
```

`P` is stored as **fp16** (5-bit exp, 10-bit mantissa). This has two problems:

**Problem 1: Denormal underflow.** fp16's smallest normal value is `2^-14 ≈ 6.1e-5`. Values below this go into denormal range, which has reduced mantissa precision. Below `2^-24 ≈ 6e-8`, values round to zero.

For a softmax of 4 values with max-subtraction, the smallest P is `exp(-delta) / sum` where `delta = max - min`. When `delta > 18`, `exp(-delta) < 1e-8` and rounds to zero in fp16. This corresponds to a logit gap of `delta * tau = 18 * tau`:
- At tau=2.0: logit gap > 36 (very rare, since logits are clamped to ±20).
- At tau=1.0: logit gap > 18 (rare).
- At tau=0.5: logit gap > 9 (common, since logits can be ±10).
- At tau=0.1: logit gap > 1.8 (almost always).

**At tau ≤ 0.5, the fp16 storage of P loses the non-argmax entries entirely.**

**Problem 2: Mantissa precision.** fp16 has 10 mantissa bits = ~3 decimal digits. The argmax P value, which is close to 1, is stored with relative error ~5e-4. For P values near 1, the absolute error is ~5e-4. This propagates to the gradient as a ~0.05% error per element.

### 4.5 W computation (fp32 → bf16)

```cuda
float W_val = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
W_out[idx] = __float2bfloat16(W_val);
```

`W_val` is computed in fp32 (correct). The output `W_out` is bf16, which has ~8e-3 relative error. **This is the W that goes into the cuBLAS matmul `y = x @ W`** — so the matmul sees a W with ~8e-3 relative noise.

### 4.6 The matmul y = x @ W

The matmul `y = x @ W` is performed by `torch.matmul(x, W)` in Python (line `y = torch.matmul(x, W)` in `CUDAFusedLUTLinearSoft.forward`). Under autocast, this uses bf16 Tensor Cores on Blackwell:
- `x` is bf16 (after autocast), `W` is bf16 (stored as such).
- Result `y` is bf16 (autocast output).
- Accumulator: fp32 (TC default).

Per-multiply error: bf16 × bf16 → fp32, ~8e-3 relative.
Accumulator error: `sqrt(K) * 8e-3 ≈ 50 * 8e-3 ≈ 0.4` relative — but this is for i.i.d. errors; in practice the bias partially cancels and the relative error on `y` is ~1-2%.

**This is the dominant source of forward error in the soft path.** The `W` from Gumbel-Softmax has ~8e-3 relative error (bf16 quantization), and the matmul propagates this with another ~1-2% error.

---

## 5. STE: forward W_hard vs backward W_soft

The Python STE (line `W = W_hard - W_soft.detach() + W_soft`) means:
- Forward: `y = x @ W_hard` (exact one-hot reconstruction, no Gumbel noise).
- Backward: gradient flows through `W_soft` (the Gumbel-Softmax blend).

**Forward precision of `y = x @ W_hard`:**
- `W_hard` is `palette[g, argmax]` (bf16 lookup, bit-exact).
- `x` is bf16.
- The matmul `torch.matmul(x, W_hard)` under autocast uses bf16 TC, ~1-2% relative error per output element.
- Cosine impact: `1 - 0.5 * 0.02^2 ≈ 1 - 2e-4` — i.e., cos > 0.9998 for the matmul alone.

But the actual training-step cos is ~0.95. This means the ~5% cos gap is NOT from the matmul precision; it's from the **palette quantization error** (the W_hard approximation to the true fp16 weight).

The 2-bit palettization itself produces cos ≈ 0.937 (from calibration log mean). After 8000 steps of LoRA + palette + (attempted) index training, cos improves to 0.946. The improvement is only 0.009 — the LoRA is mostly compensating for the residual quantization error of the W_hard path, not improving the indices.

---

## 6. Backward path precision (Python implementation)

The Python backward (`CUDAFusedLUTLinearSoft.backward`) computes:

### 6.1 grad_x = grad_y @ W.T

```python
grad_x = torch.matmul(grad_y, W.T)
```

`grad_y` and `W` are both bf16. The matmul produces a bf16 result under autocast (with fp32 accumulator). Per-element error ~8e-3 (bf16 W quantization) + ~1% (matmul). **Same precision as the forward matmul.**

### 6.2 grad_W = x.T @ grad_y

```python
grad_W = torch.matmul(x.T, grad_y)  # (K, N) bf16
```

**CRITICAL: `grad_W` is computed and stored as bf16.** This loses 16 bits of mantissa relative to fp32.

For typical magnitudes:
- `x` ~ bf16, magnitude ~1 (post-LayerNorm).
- `grad_y` ~ bf16, magnitude ~1e-2 (cos loss / dim).
- Per-element multiply: bf16 × bf16 → fp32 (TC accum), ~8e-3 relative error.
- After K=2560 accumulation: relative error ~50 * 8e-3 / sqrt(N) ≈ 4e-3.
- Stored as bf16: rounding error ~8e-3 relative.
- **Total relative error in grad_W: ~1%.**

For grad_W magnitudes of ~1e-3 (small gradient), the bf16 representation has absolute error ~8e-6 — which is the noise floor for downstream gradient computations.

### 6.3 grad_palette = sum(grad_W * P, dim=(K, GS))

```python
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)
```

- `grad_W` is bf16, `P_kno` is fp16.
- Product dtype: under autocast, **bf16 × fp16 → bf16** (autocast prefers bf16 on Blackwell).
- The product has relative error ~5e-3 (bf16 precision).
- Sum over K*GS = 655360 values: the sum grows to magnitude ~655360 * (1e-3 * 0.25) ≈ 164.
- The fp32 accumulator (from autocast) handles this fine.
- Final `.to(torch.bfloat16)`: relative error ~8e-3.

**Final grad_palette magnitude: ~164 with ~8e-3 relative error ≈ 1.3 absolute.** This is large compared to the palette values themselves (which are ~1e-2). The optimizer (FP32MasterAdamW) then casts this to fp32, so the fp32 master gradient has ~8e-3 relative error from the bf16 cast.

### 6.4 grad_logits = grad_W * P * (c - W)

```python
grad_W_f = grad_W.float()  # cast bf16 → fp32 (no precision gain!)
P_kno_f = P.permute(1, 2, 0).float()  # cast fp16 → fp32 (no precision gain!)
g_idx = torch.arange(N, device=x.device) // GS
pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
W_val = (P_kno_f * pal_pos).sum(dim=-1)
grad_logits = (
    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))
).to(torch.float16).permute(2, 0, 1).contiguous()
```

**Issues:**
1. `grad_W_f = grad_W.float()` — this is a no-op for precision. The bf16 grad_W already lost its precision; casting to fp32 just widens the representation.
2. `P_kno_f = P.permute(1, 2, 0).float()` — same, fp16 P already lost precision.
3. The product `grad_W_f * P_kno_f * (pal_pos - W_val)` has magnitude ~1e-3 * 0.25 * 1e-2 = 2.5e-6.
4. **Stored as fp16**: fp16's smallest denormal is ~6e-8, smallest normal is ~6e-5. The product 2.5e-6 is in the denormal range of fp16 — stored with reduced precision (4 mantissa bits) or rounded to zero.

**At low tau where P is one-hot:**
- `P[argmax] = 1.0`, others = 0.
- `W_val = c[argmax]` (since `P[argmax] = 1` and others = 0).
- `pal_pos[argmax] - W_val = c[argmax] - c[argmax] = 0`.
- `pal_pos[k] - W_val = c[k] - c[argmax]` for `k != argmax`.
- `grad_logits[argmax] = grad_W * 1 * 0 = 0`.
- `grad_logits[k != argmax] = grad_W * 0 * (c[k] - c[argmax]) = 0`.
- **All gradients are zero.**

This is the **vanishing-gradient at low tau** — not a numerical bug, but a mathematical property of the Gumbel-Softmax at low temperature.

---

## 7. Overflow / underflow risk matrix

| Tensor | Dtype | Magnitude (typical) | Min representable | Risk |
|--------|-------|---------------------|-------------------|------|
| `x` (input activations) | bf16 | ~1 (post-LayerNorm) | 1.2e-38 | None |
| `palette` (LUT entries) | bf16 | ~1e-2 to ~1 | 1.2e-38 | None |
| `logits` (Gumbel input) | fp16 | ±10 (clamped to ±20) | 6.1e-5 (norm) | LOW (clamp at ±20 prevents overflow) |
| `P` (softmax probs) | fp16 | 0 to 1 | 6.1e-5 (norm) | **HIGH** — non-argmax entries underflow |
| `W_soft` (Gumbel blend) | bf16 | ~1e-2 to ~1 | 1.2e-38 | None |
| `W_hard` (one-hot lookup) | bf16 | ~1e-2 to ~1 | 1.2e-38 | None |
| `y` (output) | bf16 | ~1 to ~10 | 1.2e-38 | None (inf only on NaN) |
| `grad_y` (loss gradient) | bf16 | ~1e-2 to ~1 | 1.2e-38 | None |
| `grad_W` (chain rule) | **bf16** | ~1e-3 to ~1 | 1.2e-38 | **MEDIUM** — small grads lose precision |
| `grad_palette` | bf16 (after sum) | ~1 to ~100 | 1.2e-38 | None (large enough) |
| `grad_logits` | **fp16** | ~1e-6 to ~1e-2 | 6e-8 (denorm) | **HIGH** — small grads underflow |
| `grad_bias` | bf16 | ~1 to ~10 | 1.2e-38 | **MEDIUM** — large sums lose precision |
| `grad_x` | bf16 | ~1e-2 to ~1 | 1.2e-38 | None |
| `index_logits` (master) | fp32 | ±20 (clamped) | 1.2e-38 | None |
| `index_logits` (model) | fp16 | ±20 | 65504 | None (within range) |
| `gumbel_sample` | fp32 | [-3, +16] | 1.2e-38 | None |

**Two CRITICAL underflow risks:**

1. **P in fp16**: At tau ≤ 0.5, non-argmax P values underflow to zero. This is by design (we want one-hot at low tau), but it means the gradient signal for "which palette entry should be argmax" is dead.

2. **grad_logits in fp16**: Even at moderate tau (e.g., 1.0), grad_logits values are ~1e-6, in the denormal range of fp16. These get stored with only 4 bits of mantissa precision (vs 10 for normal fp16). At low tau, they round to zero entirely.

---

## 8. The accumulation-order problem

The Python backward computes grad_palette as:

```python
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)
```

The `sum(dim=(0, 2))` reduces K*GS = 655360 values into a single fp32 accumulator. PyTorch's default reduction is sequential (not pairwise), which means:

- Accumulator grows linearly with K*GS.
- Relative error grows as `eps_fp32 * K * GS` (worst case) or `eps_fp32 * sqrt(K * GS)` (typical).
- For K*GS = 655360: typical relative error ~`1.2e-7 * 810 ≈ 1e-4`. Acceptable.
- BUT the accumulator magnitude is ~655360 * (1e-3 * 0.25) ≈ 164. Adding a value of magnitude 1e-3 to 164 loses ~7 digits of precision (164 has 3 digits, so we have 6 - 3 = 3 digits of useful precision). This is the "swamping" problem.

**Mitigation**: Use `torch.sum` with `dtype=torch.float32` (already done implicitly via autocast), and consider Welford-style accumulation or block reduction for high-precision sum.

---

## 9. Numerical experiments: what cos can the kernels support?

Based on the precision analysis, the **theoretical cos ceiling** for each path is:

| Path | Per-element relative error | Theoretical cos ceiling |
|------|---------------------------|-------------------------|
| Hard forward (SIMD2 fp32 FMA, default) | ~8e-3 (bf16 output quantization) | 0.99996 |
| Hard forward (TC mma.sync) | ~5e-3 | 0.99999 |
| Soft forward (W_soft + matmul, moderate tau) | ~2e-2 (W quantization + matmul) | 0.9998 |
| Soft forward + STE (W_hard + matmul) | ~1e-3 (just matmul, W_hard is exact palette) | 0.9999995 |
| Calibration (2-bit palettization only) | — | **0.937** (measured) |
| Training plateau (LoRA + palette + soft indices) | — | **0.946** (measured) |

**Key insight:** The kernel itself supports cos > 0.9999. The measured plateau at 0.946 is **NOT** due to kernel precision — it's due to the 2-bit palettization's representation limit (cos 0.937 from calibration) plus the LoRA's limited correction capacity.

The soft indices training is supposed to push cos above 0.937 by refining which palette entry each weight uses. But:
1. The STE forward uses `W_hard` (cos 0.937 limit inherent to the palette).
2. The gradient flows through `W_soft`, which is a noisy approximation.
3. At low tau, `W_soft ≈ W_hard` so the gradient is correct in direction, but **zero in magnitude** (vanishing gradient).
4. The LoRA can only compensate for the residual error of `W_hard` vs the true weight; it cannot change which palette entry each weight maps to.

**Therefore, the cos plateau at 0.946 is fundamentally a REPRESENTATION problem (2-bit palettization's expressiveness limit), not a numerical precision problem.** The kernel numerics are sufficient for cos > 0.999; the bottleneck is the choice of indices (frozen after calibration) and the LoRA's rank.

---

## 10. Quantified precision recommendations

| Issue | Current | Recommended | Expected improvement |
|-------|---------|-------------|----------------------|
| `P` storage | fp16 | bf16 or fp32 | Recover small gradient signal at moderate tau |
| `grad_logits` storage | fp16 | fp32 | Eliminate underflow at low tau |
| `grad_W` dtype | bf16 | fp32 | 16× more mantissa for small gradients |
| `grad_palette` reduction | bf16 sum | fp32 sum + bf16 cast | Reduce swamping error |
| `grad_bias` output | bf16 | fp32 | Avoid precision loss on large sums |
| Autocast in backward | bf16 | fp32 for backward | Eliminate all autocast-induced bf16 reductions |
| Hard kernel OOB sentinel | 0 (valid idx) | 0xFF + guard | Remove systematic palette[0] bias |

**Implementing all of these would NOT push cos above 0.95** — the plateau is dominated by the 2-bit representation limit, not by kernel precision. But it WOULD allow the Gumbel-Softmax indices to actually train (rather than being a no-op as they currently are), which could potentially push cos above 0.96 if combined with better palette initialization (k-means re-quantization after each super-block).

---

## 11. Summary

The CUDA kernels themselves are numerically sound for the hard forward path (cos > 0.999 achievable). The soft path has multiple precision issues:

1. **P stored as fp16** → vanishing non-argmax probabilities at tau ≤ 0.5.
2. **grad_logits stored as fp16** → underflow in the denormal range.
3. **grad_W computed/stored as bf16** → 16 bits of mantissa lost.
4. **grad_palette reduction in bf16** → swamping error for large reductions.
5. **grad_bias output as bf16** → precision loss on large sums.

The cos plateau at 0.946 is **NOT primarily a kernel numerics issue** — it's the 2-bit palettization representation limit (calibration cos 0.937) plus the LoRA's limited correction capacity. The kernel precision issues prevent the Gumbel-Softmax indices from training effectively, which is why "training the indices" is a no-op. **Fixing the kernel precision is necessary but not sufficient** for breaking the plateau — the palette/indices training algorithm itself needs to be redesigned (e.g., k-means re-quantization, Gumbel-Softmax with higher minimum tau, or a differentiable top-k relaxation).
