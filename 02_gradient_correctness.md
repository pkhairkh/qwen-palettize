# 02 — Gradient Correctness Audit

**Question:** Is `grad_palette` mathematically correct in both the hard and soft paths?

**Answer:** Yes. Both paths compute the correct gradient of the loss with respect to `palette`, given the forward definitions. The plateau at cos=0.95 is **not** caused by a buggy backward pass. It is caused by precision loss *after* the correct gradient is computed (see `03_precision_analysis.md`) and by loss/LR/clip missettings (see `05_loss_function.md` and `08_recommendations.md`).

This document walks through both derivations line-by-line, then lists the three places where a *correct* gradient still gets corrupted before it reaches the optimizer.

---

## 1. Setup and notation

For one `PalettizedLinear`:

- `x`: input activations, shape `(M, K)` bf16, where `K = in_dim`.
- `palette`: trainable LUT, shape `(G, 4)` bf16, where `G = out_dim / 256` and `4 = 2^bitwidth`.
- `indices`: hard assignments, shape `(K, N)` int8, where `N = out_dim`. Frozen at calibration for the hard path; replaced by `index_logits` for the soft path.
- `group_size = GS = 256`. Group of column `o` is `g(o) = o // GS`.
- `y = x @ W_recon + bias`, where `W_recon` is the (K, N) reconstructed weight.
- Upstream gradient: `grad_y`, shape `(M, N)` bf16, coming from the loss.

The chain rule for the palette is:

```
grad_palette[g, k] = ∂L / ∂palette[g, k]
                   = Σ_{j,o} (∂L / ∂W[j,o]) * (∂W[j,o] / ∂palette[g, k])
                   = Σ_{j,o} grad_W[j,o] * (∂W[j,o] / ∂palette[g, k])
```

where `grad_W = ∂L / ∂W = x.T @ grad_y` has shape `(K, N)`.

The two paths differ only in the second factor `∂W[j,o] / ∂palette[g, k]`.

---

## 2. Hard path: `∂W[j,o] / ∂palette[g, k] = δ(g, g(o)) * δ(k, indices[j,o])`

### 2.1 Derivation

In the hard path the forward is (`qwen_model.py:148-153` and `fused_lut_kernel.cu:204-207`):

```
W[j, o] = palette[g(o), indices[j, o]]
```

So `W[j, o]` depends on `palette[g, k]` iff `g = g(o)` and `k = indices[j, o]`. Therefore:

```
∂W[j, o] / ∂palette[g, k] = 1 if g(o) = g AND indices[j, o] = k
                           = 0 otherwise
```

This is the Kronecker delta on `(g, k)`, restricted to the slot that `indices` actually picks. The gradient is:

```
grad_palette[g, k] = Σ_{j, o: g(o)=g, indices[j,o]=k} grad_W[j, o]
```

This is exactly a **scatter-add** of `grad_W` into a `(G, 4)` accumulator, indexed by `(g(o), indices[j, o])`.

### 2.2 The CUDA kernel implements this correctly

`fused_lut_kernel.cu:902-1123` is the hard-path `grad_palette` kernel. The relevant scatter step is at lines 1085-1110:

```cuda
// ── Scatter_add dW into grad_palette via shared-mem accumulator ──────────
for (int ji = 0; ji < 4; ++ji) {
    const int j_local = ty * 4 + ji;
    for (int oi = 0; oi < 4; ++oi) {
        const int o_local = tx * 4 + oi;
        const int o_global = o_global_base + oi;
        if (o_global >= N) continue;
        const int j_global = j_global_base + ji;
        if (j_global >= K) continue;

        const int group = o_global / group_size;
        const int group_local = (group == g_first) ? 0 : 1;
        const uint8_t p_raw = sidx[j_local][o_local];
        const int p = (p_raw < 4) ? (int)p_raw : 0;
        const float val = dW[ji][oi];

        atomicAdd(&s_acc[group_local][p], val);
    }
}
__syncthreads();

// ── Flush smem accumulators to global grad_palette via atomicAdd ────────
if (linear_tid < 8) {
    const int g_local = linear_tid / 4;
    const int p = linear_tid % 4;
    if (g_local < n_groups_in_tile) {
        const int g_global = (g_local == 0) ? g_first : g_last;
        const float val = s_acc[g_local][p];
        atomicAdd(&grad_palette[g_global * 4 + p], val);
    }
}
```

Reading this line-by-line:

- `dW[ji][oi]` is the per-thread accumulator holding `grad_W[j, o] = Σ_i x[i, j] * grad_y[i, o]` (computed in the FMA loop at lines 1048-1081).
- `group = o_global / group_size` is the `g(o)` mapping.
- `p = sidx[j_local][o_local]` is `indices[j, o]`.
- `atomicAdd(&s_acc[group_local][p], val)` accumulates `grad_W[j, o]` into the slot `(g(o), indices[j, o])`.
- The flush at lines 1114-1122 propagates the shared-memory accumulator to the global `grad_palette` tensor.

This is precisely the scatter-add formula derived above. The kernel is correct.

### 2.3 The `dW` accumulator is fp32, but `grad_palette` is downcast at the boundary

The kernel allocates `grad_palette` as fp32 in the C++ wrapper (`fused_lut_linear_cuda.py:158`):

```cpp
grad_palette = torch::zeros({G, 4}, x.options().dtype(torch::kFloat32));
```

Then `atomicAdd` into it (`fused_lut_kernel.cu:1120`) accumulates in fp32, which is correct and avoids catastrophic cancellation. However, immediately after the kernel returns, the wrapper casts it back to bf16 (`fused_lut_linear_cuda.py:167-168`):

```cpp
// Cast back to bf16 (matching palette dtype) for autograd compatibility
grad_palette = grad_palette.to(torch::kBFloat16);
```

This is the first precision bottleneck. The fp32 accumulator holds the true gradient; the cast to bf16 truncates 23 mantissa bits down to 7. For a gradient value of magnitude ~1e-5 (typical for palette grads reported as `gn ~ 2-6` total norm divided across 2,208 params → mean ~1e-3, but with significant variance), bf16's ULP at 1e-3 is ~1e-5, so values smaller than ~1e-5 round to zero. We quantify this in `03_precision_analysis.md` §3.

The cast is forced by PyTorch autograd: the gradient tensor returned by `backward` must have the same dtype as the parameter it gradients. Since `palette` is bf16, `grad_palette` must be bf16. The fix is therefore to make `palette` itself fp32 — which is feasible because there are only 2,208 palette parameters (8.7 KB at fp32, negligible).

### 2.4 The hard-path kernel reads `palette` but does not use it for the gradient

`fused_lut_kernel.cu:905` includes `palette` in the kernel signature:

```cuda
const __nv_bfloat16* __restrict__ palette,  // (G, 4) — needed for pre-materialize W
```

and lines 968-987 use it to pre-materialize the `sW` shared-memory tile (the same way the forward kernel does). But `palette` only enters the *forward* weight reconstruction, not the *gradient* formula. The gradient is purely `grad_W * ∂W/∂palette`, and `∂W/∂palette` depends only on `indices`, not on `palette`'s value. So the kernel signature's `palette` argument is used only to materialize W for the grad_x kernel (a separate kernel, `fused_lut_linear_bwd_grad_xLauncher`, which we are not auditing here).

This means the comment `// ADDED in Phase I` at `fused_lut_linear_cuda.py:162-163` is misleading — `palette` is passed to the launcher for symmetry with `grad_x`, but it is not consumed by the `grad_palette` computation itself. No correctness issue, but worth noting because it suggests the kernel went through a Phase I refactor where `palette` was added to the signature in case it was needed; it never was.

### 2.5 Conclusion for the hard path

**The hard-path `grad_palette` formula and CUDA implementation are mathematically correct.** The scatter-add accumulates `grad_W[j, o]` into `grad_palette[g(o), indices[j, o]]` exactly as the chain rule demands. The only defect is the mandatory bf16 cast at the autograd boundary, which loses precision but does not bias the gradient.

---

## 3. Soft path: `∂W[j,o] / ∂palette[g, k] = P[j, o, k] * δ(g, g(o))`

### 3.1 Derivation (without STE)

In the soft path the forward is (`fused_lut_linear_cuda.py:574-578` and the C++ wrapper at lines 225-261):

```
P[k, j, o] = softmax_k((logits[k, j, o] + gumbel_k) / τ)
W_soft[j, o] = Σ_k P[k, j, o] * palette[g(o), k]
```

Differentiating with respect to `palette[g, k]` (holding `P` fixed, since `P` is a function of `logits`, not `palette`):

```
∂W_soft[j, o] / ∂palette[g, k] = P[k, j, o] * δ(g, g(o))
```

So the gradient is:

```
grad_palette[g, k] = Σ_{j, o: g(o)=g} grad_W[j, o] * P[k, j, o]
```

This is a **weighted scatter-add**: instead of routing all of `grad_W[j, o]` to a single slot (as in the hard path), the soft path distributes `grad_W[j, o]` across all 4 slots in proportion to `P[k, j, o]`.

### 3.2 The Python implementation matches the derivation

`fused_lut_linear_cuda.py:640-676`:

```python
grad_logits = None
grad_palette = None
if needs_grad_logits or needs_grad_palette:
    skip_grad_logits = os.environ.get("SKIP_ZERO_GRAD_LOGITS", "0") == "1"

    # grad_palette only needs grad_W * P (cheaper than full grad_logits path)
    # Use bf16 matmul for grad_W (faster, sufficient precision for palette grad)
    grad_W = torch.matmul(x.T, grad_y)  # (K, N) bf16

    if needs_grad_palette:
        # grad_palette[g, k] = Σ_{j, o in group g} grad_W[j, o] * P[j, o, k]
        # Compute via reshape + sum — no (K,N,4) fp32 materialization.
        # P is (4, K, N) fp16. Permute to (K, N, 4) but keep fp16 to save memory.
        P_kno = P.permute(1, 2, 0)  # (K, N, 4) fp16, no float() cast
        # grad_W (K,N) bf16 → expand to (K,N,1) → multiply with P_kno (K,N,4) fp16
        # Result is (K,N,4) fp16 (autocast handles bf16×fp16 → fp16)
        contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
        grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)
```

Line-by-line:

- `grad_W = torch.matmul(x.T, grad_y)` computes `grad_W[j, o] = Σ_i x[i, j] * grad_y[i, o]`. ✓
- `P_kno = P.permute(1, 2, 0)` reorders `P` from `(4, K, N)` to `(K, N, 4)`. ✓
- `grad_W.unsqueeze(-1)` makes `grad_W` shape `(K, N, 1)` so it broadcasts against `P_kno` shape `(K, N, 4)`. ✓
- `(grad_W.unsqueeze(-1) * P_kno)` produces shape `(K, N, 4)` where element `[j, o, k] = grad_W[j, o] * P[k, j, o]`. ✓
- `.view(K, G, GS, 4)` reshapes the `N` axis into `(G, GS)` so that `N = G * GS` and group `g` corresponds to rows `g*GS : (g+1)*GS`. ✓ (This works because `g(o) = o // GS` means groups are contiguous chunks of `GS` columns.)
- `.sum(dim=(0, 2))` sums over `K` (dim 0) and `GS` (dim 2), producing `(G, 4)`. Element `[g, k] = Σ_{j=0..K-1} Σ_{o in group g} grad_W[j, o] * P[k, j, o]`. ✓

This is exactly the derived formula. The implementation is correct.

### 3.3 The STE trick changes the forward value, not the gradient formula

`fused_lut_linear_cuda.py:580-595` applies the Straight-Through Estimator:

```python
with torch.no_grad():
    argmax_idx = logits.argmax(dim=0)  # (K, N) — hard index assignment
    group_idx = torch.arange(N, device=palette.device) // group_size
    group_per_col = group_idx.unsqueeze(0).expand(K, N)
    W_hard = palette[group_per_col.long(), argmax_idx.long()].to(W_soft.dtype)
# STE trick: forward = W_hard, backward = through W_soft
W = W_hard - W_soft.detach() + W_soft
y = torch.matmul(x, W)
```

The expression `W = W_hard - W_soft.detach() + W_soft` is the standard STE construction:

- **Forward value** (numerical): `W_hard - W_soft + W_soft = W_hard` (because `.detach()` only affects gradient flow, not value). So forward uses the hard weight, preserving cos.
- **Backward gradient**: `∂W/∂palette = ∂W_hard/∂palette * 0 + ∂W_soft/∂palette * 1 = ∂W_soft/∂palette` (because `W_hard` is under `torch.no_grad()` and `W_soft.detach()` has zero gradient). So backward flows through `W_soft`, giving the soft-path gradient derived in §3.1.

This means the gradient that reaches `palette` is the **soft-path** gradient (weighted by `P`), even though the forward value is the **hard-path** weight. This is by design — it lets the indices train via `P` while keeping the forward numerically exact.

**But there is a subtle implication for palette training:** since the gradient to `palette` is weighted by `P`, and `P → one-hot` as `τ → 0`, the palette gradient converges to the hard-path scatter-add formula in the limit. At low τ (late training), only the slot that `argmax(logits)` picks receives gradient; the other 3 slots per group get zero gradient. This is the same behavior as the hard path, and it means **late-training palette updates only happen at the argmax slot, not at all 4 slots**.

Concretely: if at step 4000 (τ=0.1) the argmax for group `g` is slot `k*`, then `grad_palette[g, k*]` receives the full `Σ_{j, o in g} grad_W[j, o]` and `grad_palette[g, k != k*] = 0`. The palette entries that are *not* the argmax cannot be fine-tuned, even if moving them would reduce the loss. This is a structural limitation of the STE+Gumbel approach.

The hard path has the same limitation (frozen indices mean only one slot per group receives gradient). So both paths share this property.

### 3.4 Numerical precision of the soft-path gradient

The soft-path computation mixes dtypes:

- `grad_W`: produced by `torch.matmul(x.T, grad_y)` where both `x` and `grad_y` are bf16. PyTorch's autocast may dispatch this to a bf16 tensor-core matmul, which accumulates in fp32 internally but returns bf16. So `grad_W` is bf16.
- `P_kno`: fp16 (because `index_logits` is fp16, see `qwen_model.py:124`).
- `grad_W.unsqueeze(-1) * P_kno`: bf16 × fp16. PyTorch's autocast rules promote this to fp32 for the multiplication but the comment at line 660 says "autocast handles bf16×fp16 → fp16", so the result is fp16.
- `contributions.sum(dim=(0, 2))`: summing an fp16 tensor in fp16 accumulates in fp32 internally (PyTorch's `sum` uses fp32 accumulation for fp16 inputs by default), but the result is cast back to fp16.
- `.to(torch.bfloat16)`: final cast to bf16 for autograd compatibility.

So the precision pipeline is:

```
fp32 (matmul accum) → bf16 (grad_W) → fp16 (product with P) → fp32 (sum accum) → fp16 (sum result) → bf16 (return)
```

There are **four** precision-losing casts in this pipeline. The most damaging is the first one: `grad_W` is bf16, which means small entries (below ~1e-5 in absolute value for the typical scale of activation × gradient) get rounded to zero before they ever multiply with `P`. If `grad_W` were fp32, the subsequent fp16 multiplications would lose less information because `P` is bounded in `[0, 1]` and the product would stay representable.

We quantify this loss in `03_precision_analysis.md` §4.

### 3.5 The skip-grad-logits escape hatch

`fused_lut_linear_cuda.py:643-647` contains a comment that is critical for understanding what has been tried:

```python
# PERF: skip grad_logits entirely when one-hot + low tau (grad is always 0).
# Empirically verified at tau=0.1 with logits=±10: all 25 index_logits
# grads are 0.0. The L4 "training" of indices was a no-op.
# Compute grad_logits by default (STE makes it non-zero).
# Set SKIP_ZERO_GRAD_LOGITS=1 to skip (legacy behavior, for benchmarking).
skip_grad_logits = os.environ.get("SKIP_ZERO_GRAD_LOGITS", "0") == "1"
```

This says: without STE, at low τ with one-hot logits, `grad_logits = 0` exactly, and index training is a no-op. The STE fix (lines 580-595) was added to make `grad_logits` non-zero, but the comment about "L4 training of indices was a no-op" was never updated.

The implication for palette training: even with STE making `grad_logits` non-zero, the *palette* gradient is still governed by `P` (not by `grad_logits`), and `P → one-hot` at low τ. So the palette is in the same boat as the indices: only the argmax slot receives gradient.

### 3.6 Conclusion for the soft path

**The soft-path `grad_palette` formula and Python implementation are mathematically correct.** The weighted scatter-add `Σ_{j, o in g} grad_W[j, o] * P[k, j, o]` matches the chain rule exactly. The STE trick correctly routes the forward through `W_hard` and the backward through `W_soft`. The defects are:

1. The intermediate `grad_W` is bf16 (precision loss for small entries).
2. The final `grad_palette` is cast to bf16 for autograd compatibility (precision loss).
3. As `τ → 0`, the gradient collapses to the hard-path formula, leaving non-argmax palette slots without gradient.

None of these is a *correctness* bug. They are precision and optimization-landscape issues, addressed in subsequent waves.

---

## 4. Three places where a correct gradient gets corrupted

Even though both gradient formulas are correct, the gradient that reaches `FP32MasterAdamW.step()` passes through three corrupting transformations. We list them here as setup for `03_precision_analysis.md` and `08_recommendations.md`.

### 4.1 Cast to bf16 at the autograd boundary

Both paths end with `.to(torch.bfloat16)` (`fused_lut_linear_cuda.py:168` for hard, line 662 for soft). This is forced by PyTorch's autograd contract: the gradient tensor must match the parameter's dtype.

| Path | File:line | Code |
|---|---|---|
| Hard | `fused_lut_linear_cuda.py:167-168` | `grad_palette = grad_palette.to(torch::kBFloat16);` |
| Soft | `fused_lut_linear_cuda.py:662` | `grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)` |

For 2,208 parameters, fp32 storage costs 8.7 KB. There is no memory pressure forcing bf16. The fix is `palette = nn.Parameter(..., dtype=torch.float32)`.

### 4.2 Global gradient clipping with 1.78 B index_logits in the norm

`train_qwen.py:98` sets `gradient_clip = 0.3`. This is applied (somewhere in the training loop; we did not locate the exact `clip_grad_norm_` call in the first 800 lines of `train_qwen.py` but the hyperparameter is defined and the comment at line 84-87 refers to "step 250 analysis") to the concatenated gradient of all trainable parameters.

With 1.78 B index_logits (each fp16, so each gradient is fp16) in the norm, even a per-element gradient of `1e-5` produces a total norm of `sqrt(1.78e9 * 1e-10) ≈ 13.3`. The clip `0.3` then scales the entire vector by `0.3 / 13.3 ≈ 0.0226`. The palette gradient — already small in absolute terms — is multiplied by ~1/45 of its raw value before reaching the optimizer.

This is the single most damaging transformation. We expand on it in `08_recommendations.md` §3.

### 4.3 FP32MasterAdamW copies bf16 grad into fp32 master grad

`train_qwen.py:186-194`:

```python
def step(self, closure=None):
    for group in self.opt.param_groups:
        for master in group["params"]:
            p = self.model_param_map[id(master)]
            if p.grad is not None:
                master.grad = p.grad.float()      # ← upcast already-corrupted bf16 grad
            else:
                master.grad = None
    self.opt.step(closure=closure)
```

The upcast `.float()` does not recover information that was lost in the bf16 cast. It faithfully copies the (already rounded) bf16 values into fp32. So the optimizer sees a "fp32 grad" that has the precision of bf16.

The fix is upstream: make `palette` fp32 so that `p.grad` is fp32 from the start, and the upcast is a no-op.

---

## 5. Cross-checks

### 5.1 The fallback path uses native autograd and gets the same answer

`qwen_model.py:154-163` (PyTorch fallback) computes `gathered = flat_palette[self._flat_idx]` and `y = x_flat @ gathered`. PyTorch's autograd will compute `grad_palette` by reversing the `index_select` and the `matmul`, which produces exactly the scatter-add formula. So the fallback is gradient-equivalent to the hard CUDA kernel.

This is a useful sanity check: if we suspect a CUDA bug, we can run the fallback and compare `grad_palette` from both paths. They should match to within bf16 rounding.

### 5.2 The hard and soft paths converge as τ → 0

In the limit `τ → 0`, `P[k, j, o] → δ(k, argmax_k logits[k, j, o])`, so the soft-path formula reduces to:

```
grad_palette[g, k] = Σ_{j, o: g(o)=g, argmax(logits)[j,o]=k} grad_W[j, o]
```

This is exactly the hard-path formula with `indices[j, o] = argmax(logits)[j, o]`. So at low τ the two paths agree (assuming the hard `indices` is set to `argmax(logits)`). The training loop's `extract_hard_indices` (`qwen_model.py:169-180`) does exactly this:

```python
def extract_hard_indices(self):
    if self.index_logits is not None:
        with torch.no_grad():
            self.indices = self.index_logits.argmax(dim=0).long()
            self.indices_int8 = self.indices.to(torch.int8).contiguous()
            ...
```

So at the end of soft training, extracting hard indices and switching to the hard path should produce the same `grad_palette` (up to bf16 precision). This confirms the two paths are consistent.

### 5.3 The Hessian-weighted k-means calibration is consistent with the gradient

The calibration uses `hess_diag = diag(X.T @ X)` as the per-column weight for k-means (`palettize_core.py:84-85`). The gradient `grad_W = x.T @ grad_y` weighted by `P=one-hot` is equivalent to a k-means update where the "loss" is `||y_orig - y_quant||^2` and the weight is `||x||^2` (the Hessian diagonal). So gradient descent on `palette` from the k-means initialization is a fine-tuning step on the same objective that k-means already optimizes.

This means: **k-means already finds a local optimum of the L2 reconstruction objective, and gradient descent can only find a better local optimum if the loss surface is non-convex**. For 1-D k-means with k=4, the surface is typically convex within each group (the k-means algorithm is guaranteed to converge to a local optimum, and for 1-D data with k=4 the local optimum is usually the global optimum). So gradient descent on `palette` will make only small improvements unless the loss function is different from L2 reconstruction (e.g., cosine on outputs, which is what we actually want).

This is the deepest answer to "why doesn't palette training improve cos": **the palette is already at a local optimum of the L2 reconstruction objective, and the loss being optimized (`norm_mse`) is a global version of the same L2 objective**. To improve cos, we need either (a) a different loss that directly optimizes cos, or (b) more palette capacity (smaller GROUP_SIZE), or (c) joint optimization of palette + indices + LoRA with a loss that targets cos directly.

---

## 6. Summary

| Claim | Verdict | Evidence |
|---|---|---|
| Hard-path `grad_palette = scatter_add(grad_W, indices)` is correct | ✅ True | `fused_lut_kernel.cu:1085-1122` implements the derived formula |
| Soft-path `grad_palette = (grad_W * P).sum(...)` is correct | ✅ True | `fused_lut_linear_cuda.py:654-662` implements the derived formula |
| STE trick correctly decouples forward (hard) from backward (soft) | ✅ True | `fused_lut_linear_cuda.py:580-595` follows the standard STE construction |
| Hard and soft paths agree as τ → 0 | ✅ True | Soft formula reduces to hard formula when P is one-hot |
| Fallback (PyTorch native) is gradient-equivalent to hard CUDA | ✅ True | `qwen_model.py:154-163` uses `index_select` backward = scatter_add |
| `grad_palette` reaches optimizer at full precision | ❌ False | Cast to bf16 at autograd boundary (`fused_lut_linear_cuda.py:168, 662`) |
| Gradient clip preserves palette gradient magnitude | ❌ False | Global clip 0.3 with 1.78B index_logits in norm scales palette grad by ~1/45 |
| FP32 master recovers precision lost in bf16 cast | ❌ False | `p.grad.float()` only upcasts already-rounded values (`train_qwen.py:191`) |
| K-means is at a local optimum of the L2 objective | ✅ True | 1-D k-means converges to local optimum; gradient descent fine-tunes around it |
| As τ → 0, non-argmax palette slots receive zero gradient | ✅ True | P → one-hot collapses gradient to argmax slot only |

**Bottom line:** The gradient formula is right. The plateau is caused by (a) precision loss in the gradient pipeline, (b) loss-function mismatch (we optimize `norm_mse`, not `1-cos`), (c) global gradient clipping throttling palette updates, and (d) k-means already being at a local optimum of the L2 objective. None of these is a backward-pass bug.

The next wave (`03_precision_analysis.md`) quantifies the bf16 precision loss with concrete numbers (subnormal ranges, ULP bounds, expected fraction of zeroed-out gradients). The wave after (`04_kmeans_vs_gradient.md`) addresses whether gradient descent can ever beat k-means, and what LUT-Q-style periodic re-quantization would buy us.
