# 07 — Concrete Recommendations: Code Patches to Make Indices Train

**Scope:** This document provides ready-to-apply code patches for the five fixes identified in `00_overview.md` §4. Each patch includes the file, line numbers, the exact `old_str` → `new_str` replacement, the rationale, and the expected impact. Patches are ordered by expected impact (highest first) and can be applied independently.

**Base commit:** `b82a6be` of `qwen-palettize`.

---

## Fix 1: Replace linear τ schedule with polynomial decay

**File:** `scripts/train_qwen.py`
**Lines:** 1034-1040
**Impact:** 62% boost in cumulative gradient signal; indices remain trainable throughout the 8300-step run.

### Current code (`train_qwen.py:1034-1040`)

```python
        # Temperature annealing for Gumbel-Softmax
        if use_soft_indices:
            tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
            # Update tau on all PalettizedLinear modules
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau
```

### Proposed patch

```python
        # Temperature annealing for Gumbel-Softmax
        # PROPOSED: piecewise warmup + quadratic polynomial decay.
        # Rationale: linear 2.0->0.1 spends 25% of window at tau<0.5 where
        # P[loser] < 0.016 and gradients are <9% of peak. Polynomial decay
        # with alpha=2 front-loads the high-tau regime, keeping gradients
        # strong for 66% of training. See 04_tau_schedule.md for derivation.
        if use_soft_indices:
            T_WARMUP = 500
            T_ANNEAL = 6000  # anneal over 6000 steps, then hold at tau_final
            if global_step < T_WARMUP:
                tau = tau_init  # 2.0 — warmup at high tau
            elif global_step < T_WARMUP + T_ANNEAL:
                progress = (global_step - T_WARMUP) / T_ANNEAL
                tau = max(tau_final, tau_init * (1.0 - progress) ** 2)  # alpha=2 quadratic
            else:
                tau = tau_final  # 0.5 — hold (NOT 0.1, which zero gradients)
            # Update tau on all PalettizedLinear modules
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau
```

### Also update the CLI defaults (`train_qwen.py:1241-1246`)

```python
    ap.add_argument("--tau_init", type=float, default=2.0,
                    help="Initial Gumbel-Softmax temperature. Default 2.0 (high tau = soft = gradients flow).")
    ap.add_argument("--tau_final", type=float, default=0.5,  # was 0.1
                    help="Final Gumbel-Softmax temperature. Default 0.5 (above 0.1 to keep gradients flowing).")
    ap.add_argument("--tau_anneal_steps", type=int, default=6000,  # was 4000
                    help="Steps over which to anneal temperature from tau_init to tau_final.")
```

### Verification

After applying, train for 1000 steps and check the log:
- `tau` should be `2.000` for steps 0-500 (warmup).
- `tau` should decrease quadratically from `2.0` to `0.5` over steps 500-6500.
- `tau` should hold at `0.500` for steps 6500-8300.
- `gn=[indices=X.XX]` should remain non-zero (target: `> 1e-3`) throughout training, not drop to `~0` after step 4000.

---

## Fix 2: Switch from Gumbel-ST to deterministic-ST (LLT pattern)

**File:** `scripts/fused_lut_kernel.cu`
**Lines:** 1323-1328
**Impact:** Reduces gradient variance (sign flips), allowing AdamW's `m` to accumulate constructively.

### Current code (`fused_lut_kernel.cu:1323-1328`)

```cpp
    // Add Gumbel noise + divide by tau
    float inv_tau = 1.0f / tau;
    float n0 = (l0 + gumbel_sample(step_seed, idx * 4 + 0)) * inv_tau;
    float n1 = (l1 + gumbel_sample(step_seed, idx * 4 + 1)) * inv_tau;
    float n2 = (l2 + gumbel_sample(step_seed, idx * 4 + 2)) * inv_tau;
    float n3 = (l3 + gumbel_sample(step_seed, idx * 4 + 3)) * inv_tau;
```

### Proposed patch

```cpp
    // Divide by tau (NO Gumbel noise — deterministic-ST, à la LLT CVPR 2022).
    // Rationale: Gumbel noise adds gradient variance without clear benefit at K=4.
    // The LCG sampler (gumbel_sample) is statistically weak and the marginal-
    // correctness guarantee is not needed for our deterministic argmax forward.
    // Removing the noise reduces sign flips in grad_logits, allowing AdamW's m
    // to accumulate constructively. See 02_ste_correctness.md §5.3, 05_literature_comparison.md §2.
    float inv_tau = 1.0f / tau;
    float n0 = l0 * inv_tau;
    float n1 = l1 * inv_tau;
    float n2 = l2 * inv_tau;
    float n3 = l3 * inv_tau;
```

### Also remove the unused `step_seed` parameter

The `step_seed` parameter is no longer used. For minimal disruption, leave the function signature unchanged (the parameter is just ignored). For a cleaner codebase, remove it from:

- `fused_lut_kernel.cu:1301` (kernel signature)
- `fused_lut_kernel.cu:1439-1452` (launcher)
- `fused_lut_linear_cuda.py:225-261` (C++ wrapper)
- `fused_lut_linear_cuda.py:510-516, 572, 576-578` (Python `_next_soft_step_seed` and call sites)

### Verification

After applying, the forward output should be **identical** for the same `logits` and `palette` (no Gumbel noise → deterministic). The gradient should be smoother across steps (lower variance in `gn=[indices=X.XX]`).

---

## Fix 3: Tighten logit clamp from ±20 to ±5

**File:** `scripts/train_qwen.py`
**Lines:** 1145-1153
**Impact:** Prevents logit saturation; keeps `P[loser]` non-zero throughout training.

### Current code (`train_qwen.py:1145-1153`)

```python
        # CRITICAL: clamp index_logits to safe fp16 range after step.
        # Gumbel-Softmax grad at low tau can push fp32 master to ±1e6,
        # which overflows fp16 (max 65504) → inf → NaN on next forward.
        # Clamp to ±20 (softmax(20/0.1) is already numerically one-hot).
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        par.data.clamp_(-20.0, 20.0)
```

### Proposed patch

```python
        # CRITICAL: clamp index_logits to safe fp16 range after step.
        # Gumbel-Softmax grad at low tau can push fp32 master to ±1e6,
        # which overflows fp16 (max 65504) → inf → NaN on next forward.
        # PROPOSED: clamp to ±5 (was ±20). At ±20, softmax is numerically
        # one-hot at any tau > 0.01, zeroing all loser gradients. At ±5,
        # softmax(5/0.5) = softmax(10) still gives P[winner]≈0.99995, but
        # the winner logit can't run away to ±20. This keeps P[loser] non-zero
        # throughout training, preserving gradient flow. See 03_gradient_flow_analysis.md §7.
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        par.data.clamp_(-5.0, 5.0)
```

### Verification

After applying, check that `index_logits` values stay in `[-5, 5]` throughout training (add a log line printing `logits.abs().max()` every 50 steps). The clamp should rarely trigger after the first 100 steps (once the logits have stabilized).

---

## Fix 4: Implement LLT's `1/√(N_i)` per-group gradient rescaling

**File:** `scripts/fused_lut_linear_cuda.py`
**Lines:** 664-676 (the `grad_logits` computation in the PyTorch fallback path)
**Impact:** Near-unit effect for balanced k-means init, but becomes important if collapse occurs. Safety net against codebook collapse.

### Current code (`fused_lut_linear_cuda.py:664-676`)

```python
            if needs_grad_logits and not skip_grad_logits:
                # Full grad_logits computation (only if not skipping)
                grad_W_f = grad_W.float()
                P_kno_f = P.permute(1, 2, 0).float()
                g_idx = torch.arange(N, device=x.device) // GS
                pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
                W_val = (P_kno_f * pal_pos).sum(dim=-1)
                grad_logits = (
                    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))
                ).to(torch.float16).permute(2, 0, 1).contiguous()
            elif needs_grad_logits:
                # Skip — return zero grad (matches actual behavior at low tau)
                grad_logits = torch.zeros_like(logits)
```

### Proposed patch

```python
            if needs_grad_logits and not skip_grad_logits:
                # Full grad_logits computation (only if not skipping)
                grad_W_f = grad_W.float()
                P_kno_f = P.permute(1, 2, 0).float()
                g_idx = torch.arange(N, device=x.device) // GS
                pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
                W_val = (P_kno_f * pal_pos).sum(dim=-1)

                # PROPOSED: LLT-style 1/sqrt(N_i) per-group gradient rescaling.
                # N_i[g, k] = count of (j, o) in group g with argmax(logits) == k.
                # Rescaling grad_logits by 1/sqrt(N_i + 1) balances gradient across
                # codebook entries, preventing collapse to a single dominant entry.
                # See 03_gradient_flow_analysis.md §4 for derivation.
                argmax_idx = logits.argmax(dim=0)  # (K, N)
                one_hot = F.one_hot(argmax_idx, num_classes=4).float()  # (K, N, 4)
                N_ik = torch.zeros(G, 4, device=x.device, dtype=torch.float32)
                N_ik.scatter_add_(
                    0,
                    g_idx.unsqueeze(0).expand(K, N).long(),
                    one_hot
                )  # (G, 4)
                rescale = 1.0 / (N_ik + 1.0).sqrt()  # (G, 4) — +1 avoids div by zero
                rescale_kno = rescale[g_idx.long()].unsqueeze(0).expand(K, N, 4)  # (K, N, 4)

                grad_logits = (
                    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1)) * rescale_kno
                ).to(torch.float16).permute(2, 0, 1).contiguous()
            elif needs_grad_logits:
                # Skip — return zero grad (matches actual behavior at low tau)
                grad_logits = torch.zeros_like(logits)
```

### Also add the import at the top of `fused_lut_linear_cuda.py`

```python
import torch.nn.functional as F
```

### Verification

After applying, log `N_ik` statistics every 50 steps:
- `N_ik.mean()` should be `~K_dim · group_size / 4 = 163840` (balanced).
- `N_ik.std()` should be `~350` (small, indicating balanced init).
- If `N_ik.std()` grows over training (indicating collapse), the rescaling will kick in.

---

## Fix 5: Add Hessian-weighted gradient (SqueezeLLM pattern)

**File:** new file `scripts/precompute_hessian.py` (one-time precompute) + `scripts/fused_lut_linear_cuda.py` (use in backward)
**Impact:** ~2× improvement in convergence speed on the most important positions.

### Step 1: Precompute `H_diag` once (new script)

Create `scripts/precompute_hessian.py`:

```python
"""Precompute Hessian diagonal H_diag[j, o] = 2 * sum_i x[i, j]^2 for each
PalettizedLinear module, from calibration activations.

Usage:
    python precompute_hessian.py --calib_dir <dir> --output <hessian.pt>
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

def precompute_hessian(calib_tokens_path, model, sb_idx, output_path):
    """Compute H_diag for each PalettizedLinear in the super-block.

    H_diag[j, o] = 2 * sum_i x[i, j]^2
    where x is the input activation to the PalettizedLinear.
    """
    from qwen_model import PalettizedLinear, SUPER_BLOCKS, is_full_attn_layer, is_gated_delta_layer

    tokens = torch.load(calib_tokens_path)
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    # Forward through embed + layers up to sb_end, collecting inputs to each PalettizedLinear
    h_diag_dict = {}
    with torch.no_grad():
        h = model.model.embed_tokens(tokens)
        for layer_idx in range(sb_end):
            layer = model.model.layers[layer_idx]
            # Hook to capture inputs
            inputs = {}
            def make_hook(name):
                def hook(module, inp, out):
                    inputs[name] = inp[0].detach()
                return hook
            handles = []
            for name, mod in layer.named_modules():
                if isinstance(mod, PalettizedLinear):
                    handles.append(mod.register_forward_hook(make_hook(name)))
            # Forward
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
            # Accumulate H_diag
            for name, x in inputs.items():
                # x shape: (B*S, K) or (B, S, K)
                x_flat = x.reshape(-1, x.shape[-1]).float()
                h_diag = 2.0 * (x_flat ** 2).sum(dim=0)  # (K,)
                # Broadcast to (K, N) — N is determined by the PalettizedLinear's out_features
                # We need to find the module to get N
                for n, m in layer.named_modules():
                    if n == name and isinstance(m, PalettizedLinear):
                        N = m.out_features
                        h_diag_full = h_diag.unsqueeze(1).expand(-1, N).contiguous()
                        key = f"layers.{layer_idx}.{name}"
                        if key in h_diag_dict:
                            h_diag_dict[key] += h_diag_full
                        else:
                            h_diag_dict[key] = h_diag_full
                        break
            for handle in handles:
                handle.remove()

    torch.save(h_diag_dict, output_path)
    print(f"Saved H_diag for {len(h_diag_dict)} modules to {output_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_tokens", type=str, default="cached_tokens.pt")
    ap.add_argument("--sb_idx", type=int, default=0)
    ap.add_argument("--output", type=str, default="hessian_diag.pt")
    args = ap.parse_args()
    # Load model (simplified — would need full build_student_super_block)
    print("Run this via train_qwen.py with --precompute_hessian flag")
```

### Step 2: Use `H_diag` in the backward pass (`fused_lut_linear_cuda.py:664-676`)

Add `H_diag` as a buffer in `PalettizedLinear.__init__` (`qwen_model.py:65-130`):

```python
        # Hessian diagonal for gradient weighting (SqueezeLLM pattern)
        # Precomputed from calibration activations; loaded via load_hessian().
        self.register_buffer("h_diag", torch.ones(K_dim, N_dim, dtype=torch.float32))
```

Add a `load_hessian` method:

```python
    def load_hessian(self, h_diag):
        """Load precomputed Hessian diagonal (K, N) fp32."""
        assert h_diag.shape == (self.in_features, self.out_features), \
            f"h_diag shape {h_diag.shape} != ({self.in_features}, {self.out_features})"
        self.h_diag = h_diag.to(torch.float32).to(self.palette.device)
```

Modify the forward to pass `h_diag` to the soft kernel (this requires updating the `CUDAFusedLUTLinearSoft.forward` signature, or — simpler — store `h_diag` on the module and access it via a closure). The cleanest approach is to pass `h_diag` through `ctx`:

In `fused_lut_linear_cuda.py:602`, add `h_diag` to saved tensors:

```python
        ctx.save_for_backward(x, palette, logits, P, W, h_diag)
```

In `fused_lut_linear_cuda.py:664-676`, multiply `grad_logits` by `h_diag`:

```python
            if needs_grad_logits and not skip_grad_logits:
                grad_W_f = grad_W.float()
                P_kno_f = P.permute(1, 2, 0).float()
                g_idx = torch.arange(N, device=x.device) // GS
                pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
                W_val = (P_kno_f * pal_pos).sum(dim=-1)

                # PROPOSED: Hessian-weighted gradient (SqueezeLLM pattern).
                # h_diag[j, o] = 2 * sum_i x[i, j]^2 (precomputed).
                # Weighting grad_logits by h_diag focuses gradient on sensitive
                # weights, improving convergence speed ~2x on important positions.
                # See 05_literature_comparison.md §4.
                h_diag_f = h_diag.float().unsqueeze(0).expand(K, N, 1)  # (K, N, 1)

                # (also include LLT 1/sqrt(N_i) rescaling from Fix 4)
                # ... (Fix 4 code here) ...

                grad_logits = (
                    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1)) * rescale_kno * h_diag_f
                ).to(torch.float16).permute(2, 0, 1).contiguous()
```

### Verification

After applying, the gradient on high-activation channels should be larger. Log `grad_logits.abs().mean()` weighted by `h_diag` vs unweighted; the ratio should be `~2×` (indicating the Hessian weighting is focusing gradient on the right positions).

---

## Combined patch application order

Apply the fixes in this order (each is independent, but this order minimizes risk):

1. **Fix 3** (logit clamp `±20` → `±5`): one-line change, immediate safety benefit.
2. **Fix 1** (τ schedule): one-block change, highest expected impact.
3. **Fix 2** (deterministic-ST): CUDA kernel change, requires recompile but no Python changes.
4. **Fix 4** (`1/√(N_i)` rescaling): ~10 lines in `fused_lut_linear_cuda.py`, safety net.
5. **Fix 5** (Hessian weighting): requires precompute script + module changes, most involved.

After each fix, run a 1000-step training test and check:
- `cos` should not decrease (forward correctness preserved by STE).
- `gn=[indices=X.XX]` should remain non-zero throughout (target: `> 1e-3` at step 1000).
- `argmax(logits)` flip rate should be `> 0.01%` per step (indices are actually moving).

If any fix causes regression, roll it back and proceed to the next.

---

## The radical alternative: LUT-Q pattern

If all five fixes don't break the plateau, switch to the LUT-Q pattern (Cardinaux et al. 2018). This is a structural change, not a patch.

### Concept

Replace `index_logits` with an FP shadow weight matrix `W_shadow` (same shape as `W_hard`). Train `W_shadow` with standard STE (`grad_W_shadow = grad_W`). Every N steps, recompute `indices = argmin_k |W_shadow - palette[g, k]|` via k-means.

### Implementation sketch

In `qwen_model.py:65-180`, replace the `index_logits` parameter with:

```python
        # LUT-Q pattern: FP shadow weight, k-means reassignment every N steps
        if use_soft_indices and not pre_transposed:
            K_dim, N_dim = indices.shape
            # Initialize W_shadow from palette + indices (one-hot reconstruction)
            W_shadow = torch.zeros(K_dim, N_dim, dtype=torch.float32, device=device)
            for k in range(4):
                mask = (indices == k)
                W_shadow[mask] = palette[..., k][mask]  # gather palette values
            self.W_shadow = nn.Parameter(W_shadow)
            self.kmeans_interval = 10  # reassign indices every 10 steps
            self._kmeans_counter = 0
        else:
            self.W_shadow = None
```

In `forward`, replace the soft kernel call with:

```python
            if self.training and self.use_soft_indices and self.W_shadow is not None:
                # LUT-Q: forward uses hard indices (recomputed from W_shadow via k-means)
                if self._kmeans_counter % self.kmeans_interval == 0:
                    self._recompute_indices()  # k-means step
                self._kmeans_counter += 1
                # STE: forward = palette[g, indices], backward = through W_shadow
                W_hard = self.palette[..., self.indices_int8.long()]  # (K, N) bf16
                W = W_hard - self.W_shadow.detach() + self.W_shadow  # STE bridge
                y = torch.matmul(x_flat, W)
                if self.bias is not None:
                    y = y + self.bias
            else:
                y = self._hard_kernel(...)  # existing hard path
```

### Trade-offs

**Pros:**
- Eliminates Gumbel-Softmax gradient damping entirely (`grad_W_shadow = grad_W ~ 1e-3`, vs `grad_logits ~ 2.6e-6`).
- Conceptually simpler (no τ, no softmax, no Gumbel noise).
- Empirically more stable (LUT-Q's track record since 2018).

**Cons:**
- `W_shadow` is `K · N · 4B = 26 MB` per Linear × 25 Linears = 650 MB additional memory (acceptable).
- k-means reassignment every 10 steps adds ~7B FLOPs (negligible vs matmul).
- Requires rethinking the optimizer (AdamW on `W_shadow` instead of `index_logits`).

---

## Summary

The five patches (Fix 1-5) are the recommended path. They are independently applicable, each with clear theoretical justification and minimal code changes. The expected cumulative effect is to break the `cos = 0.95` plateau by allowing indices to continue training throughout the 8300-step run, rather than freezing at step 4000.

If the patches don't break the plateau, the LUT-Q pattern (§"The radical alternative") is the fallback — it eliminates the Gumbel-Softmax formulation entirely and replaces it with the simpler, more stable FP-shadow + k-means + STE pattern.

---

## References

1. Jang, E. et al. *Categorical Reparameterization with Gumbel-Softmax.* [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
2. Wang, L. et al. *Learnable Lookup Table for Neural Network Quantization (LLT).* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html)
3. Cardinaux, F. et al. *Iteratively Training Look-Up Tables for Network Quantization (LUT-Q).* [arXiv:1811.05355](https://arxiv.org/abs/1811.05355)
4. Kim, S. et al. *SqueezeLLM: Dense-and-Sparse Quantization.* [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
5. Nagel, M. et al. *Overcoming Oscillations in Quantization-Aware Training.* [arXiv:2203.11086](https://arxiv.org/abs/2203.11086)
6. Bengio, Y. et al. *Estimating or Propagating Gradients Through Stochastic Neurons (STE).* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

*Code citations refer to commit `b82a6be` of `qwen-palettize`.*
