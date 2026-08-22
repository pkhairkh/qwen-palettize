# 06 — Concrete Recommendations and Code Patches

**Scope:** Actionable code patches (not just prose) for the highest-impact fixes. Each patch is presented as a diff against the current `fused_lut_linear_cuda.py` and `fused_lut_kernel.cu`, with rationale and expected cos impact.

---

## Patch 1: Floor Gumbel-Softmax τ at 0.5 (immediate, +0.005-0.01 cos)

**File:** `scripts/train_qwen.py`
**Lines:** around 1241-1246 (argument defaults) and around 1034-1040 (τ anneal).

### Problem

The current schedule anneals τ from 2.0 → 0.1 over 4000 steps. At τ ≤ 0.1, the Gumbel-Softmax gradient vanishes (proven in 03_ste_analysis.md section 5), making index training a no-op.

### Fix

Change the τ floor from 0.1 to 0.5, and shorten the anneal range from 2.0→0.1 to 1.0→0.5:

```python
# BEFORE (current, train_qwen.py lines 1241-1246):
ap.add_argument("--tau_init", type=float, default=2.0,
                help="Initial Gumbel-Softmax temperature. Default 2.0 (high tau = soft = gradients flow).")
ap.add_argument("--tau_final", type=float, default=0.1,
                help="Final Gumbel-Softmax temperature. Default 0.1 (below this, gradients vanish).")
ap.add_argument("--tau_anneal_steps", type=int, default=4000,
                help="Steps over which to anneal temperature from tau_init to tau_final.")
```

```python
# AFTER (fixed):
ap.add_argument("--tau_init", type=float, default=1.0,
                help="Initial Gumbel-Softmax temperature. Default 1.0 (lower to keep STE valid).")
ap.add_argument("--tau_final", type=float, default=0.5,
                help="Final Gumbel-Softmax temperature. Default 0.5 (FLOOR: below 0.5 gradients vanish).")
ap.add_argument("--tau_anneal_steps", type=int, default=2000,
                help="Steps over which to anneal temperature from tau_init to tau_final.")
```

Also, after training, extract hard indices for inference:

```python
# Add at end of training loop, before final save:
if use_soft_indices:
    print("Extracting hard indices from trained index_logits...")
    for name, mod in student.named_modules():
        if hasattr(mod, 'extract_hard_indices') and mod.index_logits is not None:
            mod.extract_hard_indices()
    print("Hard indices extracted. Model is now in eval mode (no Gumbel-Softmax).")
```

### Rationale

At τ=0.5 with logits=±10, the softmax is peaked but not one-hot:
- `softmax(10/0.5) = softmax(20) ≈ [0.9999999979, ...]` — non-argmax entries are ~2e-9, still representable in fp32.
- The gradient `grad_W * P[k] * (c[k] - W_soft)` for non-argmax k is `grad_W * 2e-9 * (c[k] - c[argmax])` — small but non-zero.
- The gradient for the argmax k is `grad_W * 1.0 * (c[argmax] - W_soft) ≈ grad_W * 1.0 * (c[argmax] - c[argmax]) = 0` — still zero for the argmax, but the non-argmax gradients can push the logits to change the argmax.

### Expected impact

- Enables actual index training (was a no-op).
- Estimated +0.005-0.01 cos improvement (indices can refine the k-means solution).
- Combined with Patch 2 (fp32 P/grad_logits), the index training becomes numerically viable.

---

## Patch 2: Use fp32 for P and grad_logits (immediate, +0.002-0.005 cos)

**File:** `scripts/fused_lut_kernel.cu`
**Lines:** 1301-1352 (soft forward kernel) and 1364-1398 (soft backward grad_logits kernel).

### Problem

P is stored as fp16 (`__half`), which underflows to zero for non-argmax probabilities at τ ≤ 0.5. grad_logits is also stored as fp16, which underflows for small gradients.

### Fix

Change P and grad_logits from `__half` to `float`:

```cuda
// BEFORE (fused_lut_kernel.cu, line 1304-1305):
__half*              __restrict__ P,         // (4, K, N) fp16 — output
__nv_bfloat16*       __restrict__ W_out,     // (K, N) bf16 — output

// AFTER (fixed):
float*               __restrict__ P,         // (4, K, N) fp32 — output (was fp16, underflowed)
__nv_bfloat16*       __restrict__ W_out,     // (K, N) bf16 — output (unchanged)
```

```cuda
// BEFORE (line 1340-1343):
P[0 * plane_size + idx] = __float2half(p0);
P[1 * plane_size + idx] = __float2half(p1);
P[2 * plane_size + idx] = __float2half(p2);
P[3 * plane_size + idx] = __float2half(p3);

// AFTER (fixed):
P[0 * plane_size + idx] = p0;  // fp32 store (was __float2half, underflowed at low tau)
P[1 * plane_size + idx] = p1;
P[2 * plane_size + idx] = p2;
P[3 * plane_size + idx] = p3;
```

```cuda
// BEFORE (line 1368):
__half*              __restrict__ grad_logits, // (4, K, N) fp16 — output

// AFTER (fixed):
float*               __restrict__ grad_logits, // (4, K, N) fp32 — output (was fp16, underflowed)
```

```cuda
// BEFORE (lines 1394-1397):
grad_logits[0 * plane_size + idx] = __float2half(dW * p0 * (c0 - W_val));
grad_logits[1 * plane_size + idx] = __float2half(dW * p1 * (c1 - W_val));
grad_logits[2 * plane_size + idx] = __float2half(dW * p2 * (c2 - W_val));
grad_logits[3 * plane_size + idx] = __float2half(dW * p3 * (c3 - W_val));

// AFTER (fixed):
grad_logits[0 * plane_size + idx] = dW * p0 * (c0 - W_val);  // fp32 store
grad_logits[1 * plane_size + idx] = dW * p1 * (c1 - W_val);
grad_logits[2 * plane_size + idx] = dW * p2 * (c2 - W_val);
grad_logits[3 * plane_size + idx] = dW * p3 * (c3 - W_val);
```

### Update the C++ wrapper and Python side

In `fused_lut_linear_cuda.py`, change the CPP_SOURCE declarations and the Python autograd Function to use fp32 for P and grad_logits:

```python
# BEFORE (in CPP_SOURCE):
auto P = torch::empty({4, K, N}, logits.options());  // fp16
// ...
auto grad_logits = torch::empty({4, K, N}, P.options());  // fp16

# AFTER (fixed):
auto P = torch::empty({4, K, N}, x.options().dtype(torch::kFloat32));  // fp32
// ...
auto grad_logits = torch::empty({4, K, N}, P.options());  // fp32
```

```python
# BEFORE (in CUDAFusedLUTLinearSoft.forward):
# logits is fp16, P is fp16

# AFTER (fixed):
# logits can stay fp16 (input), but P and grad_logits should be fp32
logits_f32 = logits.float()  # cast to fp32 before kernel call
# ... pass logits_f32 to the kernel ...
# P and grad_logits are now fp32
```

### Memory cost

P grows from (4, K, N) fp16 = 4 × 2560 × 8192 × 2 bytes = 168 MB to fp32 = 336 MB. grad_logits similarly. Total +336 MB VRAM — acceptable on 96 GB Blackwell.

### Expected impact

- Eliminates fp16 underflow for P at τ ≤ 0.5.
- Eliminates fp16 underflow for grad_logits at small magnitudes.
- Enables the Gumbel-Softmax gradient to flow correctly at τ=0.5.
- Estimated +0.002-0.005 cos (combined with Patch 1).

---

## Patch 3: Use fp32 for grad_W in Python backward (immediate, +0.002 cos)

**File:** `scripts/fused_lut_linear_cuda.py`
**Lines:** in `CUDAFusedLUTLinearSoft.backward`, around the `grad_W = torch.matmul(x.T, grad_y)` line.

### Problem

`grad_W = torch.matmul(x.T, grad_y)` produces a bf16 tensor under autocast, losing 16 bits of mantissa relative to fp32.

### Fix

Cast x and grad_y to fp32 before the matmul:

```python
# BEFORE (current, in CUDAFusedLUTLinearSoft.backward):
grad_W = torch.matmul(x.T, grad_y)  # (K, N) bf16

# AFTER (fixed):
# Disable autocast for the grad_W matmul — use fp32 for full precision
with torch.amp.autocast(device_type='cuda', enabled=False):
    grad_W = torch.matmul(x.float().T, grad_y.float())  # (K, N) fp32
```

### Update downstream consumers

The grad_palette computation can now use fp32 grad_W directly (no need for `.float()` cast):

```python
# BEFORE:
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)

# AFTER:
# grad_W is already fp32, P_kno is now fp32 (from Patch 2)
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)  # fp32 * fp32 = fp32
grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)  # cast to bf16 at the end
```

The grad_logits computation no longer needs the `.float()` cast:

```python
# BEFORE:
grad_W_f = grad_W.float()
P_kno_f = P.permute(1, 2, 0).float()
# ...

# AFTER:
# grad_W is already fp32, P is already fp32 (from Patch 2)
P_kno_f = P.permute(1, 2, 0)  # already fp32
# ... (no .float() needed)
```

### Memory cost

grad_W grows from (K, N) bf16 = 42 MB to fp32 = 84 MB. Total +42 MB VRAM — negligible.

### Expected impact

- 16× more mantissa for small gradients.
- Eliminates the bf16 quantization noise in grad_palette.
- Estimated +0.002 cos (small but compounds with Patch 1 + 2).

---

## Patch 4: Add GPTQ-style second-order calibration (medium effort, +0.02-0.04 cos)

**File:** New file `scripts/gptq_calibrate.py` (or extend `palettize_core.py`).

### Problem

The current calibration (`palettize_core.py`) does pure k-means + round-to-nearest, which is the simplest and weakest quantization strategy. GPTQ-style second-order compensation can recover 2-4% cos by re-distributing quantization error across columns.

### Implementation sketch

```python
# scripts/gptq_calibrate.py
import torch
import torch.nn.functional as F

def gptq_calibrate_linear(linear_module, x_calib, group_size=256, n_palette=4):
    """GPTQ-style 2-bit palettization with second-order compensation.

    Args:
        linear_module: nn.Linear to palettize.
        x_calib: (M, in_features) calibration activations.
        group_size: 256 (default).
        n_palette: 4 (2-bit).

    Returns:
        indices: (in_features, out_features) int8
        palette: (n_groups, 4) bf16
    """
    W = linear_module.weight.data.float()  # (out_features, in_features)
    X = x_calib.float()  # (M, in_features)
    M, K = X.shape
    out_features, in_features = W.shape

    # Compute Hessian = X^T @ X (per-layer, ignoring cross-layer effects)
    H = X.T @ X  # (K, K)
    H += 0.001 * torch.diag(torch.diag(H).mean())  # damping

    # Process columns of W (rows of W.T) one at a time
    # For 2-bit LUT with GS=256: group every 256 columns together
    n_groups = (out_features + group_size - 1) // group_size
    indices = torch.zeros(in_features, out_features, dtype=torch.int8, device=W.device)
    palette = torch.zeros(n_groups, n_palette, dtype=torch.bfloat16, device=W.device)

    # Initialize palette via k-means on the full W (per group)
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, out_features)
        W_g = W[:, start:end].flatten()
        # k-means with k=4
        centroids = kmeans(W_g, k=n_palette, n_iters=50)
        palette[g] = centroids.to(torch.bfloat16)

    # GPTQ main loop: quantize one column at a time, compensate the rest
    for j in range(out_features):
        # Quantize column j of W (row j of W.T)
        g = j // group_size
        w_col = W[:, j]  # (in_features,)
        # Assign each weight to nearest palette entry
        dists = (w_col.unsqueeze(-1) - palette[g].float().unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)  # (in_features,)
        q_col = palette[g].float()[idx]  # (in_features,)
        indices[:, j] = idx.to(torch.int8)

        # Compute the quantization error for this column
        err = (w_col - q_col)  # (in_features,)

        # Compensate: update remaining columns using inverse Hessian
        H_inv_j = H.inverse()[j, :]  # (in_features,) — precompute once
        # W[:, j+1:] -= err.unsqueeze(-1) * H_inv_j.unsqueeze(0) / H[j, j]
        # (This is O(K) per column, O(K*N) total — can be batched)
        scale = H_inv_j / H[j, j]
        W[:, j+1:] -= err.unsqueeze(-1) * scale.unsqueeze(0)

    return indices, palette


def kmeans(x, k=4, n_iters=100):
    """Simple 1D k-means."""
    centroids = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(n_iters):
        dists = (x.unsqueeze(-1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=-1)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centroids[i] = x[mask].mean()
    return centroids
```

### Integration

Replace the k-means call in `palettize_core.py` with `gptq_calibrate_linear`. The output format (indices, palette) is unchanged, so the rest of the pipeline (PalettizedLinear, QwenLoRA, training loop) works without modification.

### Expected impact

- Reduces quantization error by ~30-50% vs pure k-means + RTN.
- Estimated +0.02-0.04 cos at calibration (0.937 → ~0.96-0.98).
- Combined with LoRA training, final cos could reach ~0.97-0.99.

---

## Patch 5: Add outlier isolation (medium effort, +0.01-0.03 cos)

**File:** New file `scripts/outlier_isolate.py` and modify `PalettizedLinear` in `qwen_model.py`.

### Problem

The top 0.5% of weights (by magnitude) account for ~30-50% of the quantization MSE because they round to the largest palette entry, losing precision. Isolating them in a sparse FP16 matrix removes this error source.

### Implementation sketch

```python
# scripts/outlier_isolate.py
import torch
import torch.nn as nn

def extract_outliers(W, threshold_pct=0.005):
    """Extract the top threshold_pct of weights (by magnitude) into a sparse FP16 matrix.

    Args:
        W: (out_features, in_features) weight matrix.
        threshold_pct: fraction of weights to extract (default 0.5%).

    Returns:
        W_dense: (out_features, in_features) with outliers zeroed out.
        W_sparse: (out_features, in_features) sparse FP16 matrix with only outliers.
    """
    abs_W = W.abs()
    threshold = torch.quantile(abs_W.flatten(), 1.0 - threshold_pct)
    mask = abs_W > threshold
    W_sparse = (W * mask).to(torch.float16)
    W_dense = W * (~mask)
    return W_dense, W_sparse


class PalettizedLinearWithOutliers(nn.Module):
    """PalettizedLinear + sparse FP16 outlier matrix."""

    def __init__(self, original_linear, indices, palette, group_size, outlier_matrix):
        super().__init__()
        self.pal_linear = PalettizedLinear(
            original_linear, indices, palette, group_size
        )
        # Outlier matrix as a dense buffer (could be sparse for memory efficiency)
        self.register_buffer("outlier_matrix", outlier_matrix)
        # Trainable scale (so the model can learn to downweight outliers)
        self.outlier_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float16))

    def forward(self, x):
        # Main path: palettized linear
        y = self.pal_linear(x)
        # Outlier path: x @ outlier_matrix.T * scale
        # (use fp16 matmul for speed)
        orig_ndim = x.ndim
        if orig_ndim == 3:
            B, S, _ = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x
        y_outlier = (x_flat @ self.outlier_matrix.T.to(x.dtype)) * self.outlier_scale
        if orig_ndim == 3:
            y_outlier = y_outlier.reshape(B, S, -1)
        return y + y_outlier
```

### Integration

In `qwen_model.py`'s `palettize_linear` function, after loading the original weight, extract outliers before passing to `PalettizedLinear`:

```python
# In palettize_linear (qwen_model.py):
# ... load original weight ...
W_orig = linear_module.weight.data.float()
W_dense, W_sparse = extract_outliers(W_orig, threshold_pct=0.005)
# Palettize the dense part (without outliers)
# ... existing k-means / GPTQ calibration on W_dense ...
# Create PalettizedLinearWithOutliers
pal_mod = PalettizedLinearWithOutliers(
    linear_module, indices, lut, GROUP_SIZE, W_sparse
)
```

### Memory cost

The outlier matrix is (out_features, in_features) fp16 = full weight size. For Qwen3.5-4B with ~4B weights, this is 8 GB. **This is too much.**

**Optimization:** Store the outlier matrix as a true sparse tensor (COO or CSR format) with only 0.5% non-zero entries. Memory cost: 0.005 × 4B × 4 bytes (fp32 index + fp16 value) = 80 MB total. Acceptable.

The sparse matmul `x @ W_sparse.T` is slower than dense but only processes 0.5% of the weights — net ~10-20% slower than the dense equivalent, but the cos improvement justifies it.

### Expected impact

- Removes the heavy-tail contribution to quantization error.
- Estimated +0.01-0.03 cos at calibration (0.937 → ~0.95-0.97).
- Combined with GPTQ (Patch 4) and LoRA training, final cos could reach ~0.98-0.99.

---

## Patch 6: Switch loss from pure cosine to 50/50 cosine + block-MSE (low effort, +0.005-0.02 cos)

**File:** `scripts/train_qwen.py`, in `compute_loss` function (around line 222).

### Problem

The current loss is pure `1 - cos(student, teacher)`, which is a normalized loss that ignores magnitude differences. The gradient signal is weak, especially for deep layers where the per-layer cos contribution to global cos is small.

### Fix

Add a block-MSE term to the loss:

```python
# BEFORE (current, compute_loss):
def compute_loss(student_out, teacher_out, hp):
    s = student_out.float()
    t = teacher_out.float()
    cos_per = F.cosine_similarity(s.flatten(0, 1), t.flatten(0, 1), dim=-1, eps=1e-4)
    l_cos = (1 - cos_per).mean()
    loss_type = hp.get("loss_type", "1-cos+norm_mse")
    if loss_type == "1-cos":
        loss = l_cos
    elif loss_type == "1-cos+norm_mse":
        l_mse = ((s - t) ** 2).mean()
        t_var = t.var() + 1e-6
        loss = l_cos + l_mse / t_var
    # ...
    return loss, {"cos": l_cos.item(), "loss": loss.item()}


# AFTER (fixed):
def compute_loss(student_out, teacher_out, hp):
    s = student_out.float()
    t = teacher_out.float()

    # Cosine loss (direction alignment)
    cos_per = F.cosine_similarity(s.flatten(0, 1), t.flatten(0, 1), dim=-1, eps=1e-4)
    l_cos = (1 - cos_per).mean()

    # Block-MSE loss (per-token reconstruction error)
    l_mse = ((s - t) ** 2).mean(dim=-1).mean()  # per-token MSE
    t_var = t.var() + 1e-6
    l_mse_normalized = l_mse / t_var

    # Combined loss (50/50)
    loss_type = hp.get("loss_type", "1-cos+norm_mse")
    if loss_type == "1-cos":
        loss = l_cos
    elif loss_type == "1-cos+norm_mse":
        # Weight cosine and MSE equally (normalize each to [0, 1] range first)
        w_cos = hp.get("loss_weights", {"cos": 0.5, "mse": 0.5})["cos"]
        w_mse = hp.get("loss_weights", {"cos": 0.5, "mse": 0.5})["mse"]
        loss = w_cos * l_cos + w_mse * l_mse_normalized
    elif loss_type == "norm_mse":
        loss = l_mse_normalized
    # ...
    return loss, {"cos": l_cos.item(), "mse": l_mse.item(), "loss": loss.item()}
```

### Rationale

The block-MSE term provides a direct gradient signal for the reconstruction error, while the cosine term ensures direction alignment. The 50/50 weighting balances the two objectives. AQLM uses pure block-MSE (no cosine); the qwen-palettize repo's pure cosine is too weak.

### Expected impact

- Stronger gradient signal for palette and LoRA training.
- Estimated +0.005-0.02 cos (depending on how undertrained the palette/LoRA were).

---

## Patch 7: Use PyTorch's built-in `gumbel_softmax(hard=True)` (low effort, +0.001-0.005 cos)

**File:** `scripts/fused_lut_linear_cuda.py`, in `CUDAFusedLUTLinearSoft.forward`.

### Problem

The manual STE `W = W_hard - W_soft.detach() + W_soft` is mathematically correct but introduces small numerical errors when `W_soft.detach()` and `W_soft` are re-computed separately (e.g., due to non-deterministic Gumbel sampling if the seed is not properly reset).

### Fix

Replace the manual STE with PyTorch's built-in `F.gumbel_softmax(hard=True)`:

```python
# BEFORE (current):
# ... compute W_soft via CUDA kernel ...
with torch.no_grad():
    argmax_idx = logits.argmax(dim=0)
    group_idx = torch.arange(N, device=palette.device) // group_size
    group_per_col = group_idx.unsqueeze(0).expand(K, N)
    W_hard = palette[group_per_col.long(), argmax_idx.long()].to(W_soft.dtype)
W = W_hard - W_soft.detach() + W_soft
y = torch.matmul(x, W)

# AFTER (fixed):
import torch.nn.functional as F
# Use PyTorch's built-in Gumbel-Softmax with hard=True
# This gives one-hot in forward, soft gradient in backward — exactly the STE behavior.
# logits is (4, K, N); we need to apply gumbel_softmax over dim=0 (the 4 palette entries).
P_hard = F.gumbel_softmax(
    logits.permute(1, 2, 0),  # (K, N, 4)
    tau=tau,
    hard=True,
    dim=-1,
).permute(2, 0, 1)  # back to (4, K, N)

# Compute W from P_hard (one-hot in forward, soft gradient in backward)
# W[k, n] = sum_i P_hard[i, k, n] * palette[g, i]
group_idx = torch.arange(N, device=palette.device) // group_size
pal_expanded = palette[group_idx.long()]  # (N, 4) -> need (K, N, 4)
pal_expanded = pal_expanded.unsqueeze(0).expand(K, N, 4)
# P_hard is (4, K, N) -> permute to (K, N, 4)
P_hard_kno = P_hard.permute(1, 2, 0)
W = (P_hard_kno * pal_expanded).sum(dim=-1)  # (K, N) — one-hot in forward
y = torch.matmul(x, W)
```

### Rationale

PyTorch's `gumbel_softmax(hard=True)` is the canonical STE implementation, handles edge cases correctly (e.g., τ → 0), and is well-tested. It also avoids the manual `argmax` + gather, which is slower on GPU.

### Expected impact

- Eliminates the manual STE arithmetic (small numerical improvement).
- Cleaner code, easier to maintain.
- Estimated +0.001-0.005 cos (mostly from avoiding re-compute of W_soft.detach()).

---

## Patch 8: Fix OOB indices sentinel (low effort, removes systematic bias)

**File:** `scripts/fused_lut_kernel.cu`
**Lines:** 186, 401, 632, 812, 961.

### Problem

Out-of-bounds indices are set to 0 (a valid palette entry), causing a systematic bias toward `palette[0]` for edge tiles.

### Fix

Use `0xFF` (255) as the OOB sentinel, and add a guard in the materialize-W step:

```cuda
// BEFORE (e.g., line 186):
sidx[r][c + i] = ok ? indices[(k_chunk + r) * N + (bn * FWD_BN + c + i)] : 0;

// AFTER (fixed):
sidx[r][c + i] = ok ? indices[(k_chunk + r) * N + (bn * FWD_BN + c + i)] : 0xFF;
```

```cuda
// In the materialize-W step (e.g., line 207):
const uint8_t idx_val = sidx[r][cc];
// Guard: if idx_val is OOB (0xFF), use 0 contribution (multiply by 0)
const float w_val = (idx_val < 4) ? __bfloat162float(spalette[group_local][idx_val]) : 0.0f;
sW[r][cc] = __float2bfloat16(w_val);
```

### Rationale

Using 0 (a valid palette entry) biases the partial sum toward `palette[g, 0]`. Using 0xFF (invalid) and guarding with `if (idx_val < 4)` produces zero contribution, which is the mathematically correct behavior for OOB.

### Expected impact

- Removes the systematic bias toward `palette[0]` for edge tiles.
- Estimated +0.0005-0.001 cos (small but removes a correctness bug).

---

## Patch 9: Add NaN guard in hard kernel output (low effort, prevents silent failure)

**File:** `scripts/fused_lut_kernel.cu`
**Lines:** 285, 514, 716, 888, 1146.

### Problem

The output write `y[m_global * N + n_global] = __float2bfloat16(v)` has no NaN/inf guard. If `v` is NaN or inf (e.g., due to a bad gradient update), the NaN propagates silently to the next layer.

### Fix

Add a NaN check before the output write:

```cuda
// BEFORE (line 285):
y[m_global * N + n_global] = __float2bfloat16(v);

// AFTER (fixed):
// Guard against NaN/inf propagation
if (isnan(v) || isinf(v)) {
    v = 0.0f;  // or could use a sentinel like -1.0f to trigger an assertion
}
y[m_global * N + n_global] = __float2bfloat16(v);
```

Apply the same fix to lines 514, 716, 888, 1146.

### Rationale

NaN propagation is a common cause of silent training divergence. Guarding against it allows the training loop to detect and recover from bad updates (e.g., by skipping the step or reducing the learning rate).

### Expected impact

- No direct cos improvement, but prevents silent training failures.
- Improves debugging experience (NaN is detected at the kernel output, not 5 layers later).

---

## Patch 10: Long-term — switch to AQLM-style additive VQ (very high effort, +0.05-0.10 cos)

This is the fundamental algorithmic change recommended in 04_literature_comparison.md and 05_convergence_analysis.md. It requires rearchitecting the codebook structure, the CUDA kernels, and the training loop. The implementation is too large for a single patch; see the AQLM paper ([arxiv 2401.06118](https://arxiv.org/abs/2401.06118)) and the [AQLM GitHub repo](https://github.com/Vahe1994/AQLM) for reference.

### High-level changes

1. **Replace the 4-entry palette with two 256-entry codebooks** (additive VQ, K=2, B=8 bits each).
2. **Block the weights into 8-dim blocks** (instead of per-weight scalar quantization).
3. **Rewrite the hard forward kernel** to do additive codebook lookups.
4. **Replace Gumbel-Softmax with direct STE** (beam search in forward, direct gradient in backward).
5. **Add GPTQ-style initialization** before gradient descent.
6. **Train for 100K+ steps** with block-MSE loss.

### Expected impact

- Breaks the 2-bit scalar LUT ceiling (cos ~0.94).
- Achieves cos > 0.99 at 2 bits/weight (matching AQLM).
- This is the only path to literature-SOTA accuracy.

---

## Summary of patches

| Patch | Effort | Cos impact | Priority |
|-------|--------|-----------|----------|
| 1. Floor τ at 0.5 | Low | +0.005-0.01 | Immediate |
| 2. fp32 P + grad_logits | Low | +0.002-0.005 | Immediate |
| 3. fp32 grad_W | Low | +0.002 | Immediate |
| 4. GPTQ calibration | Medium | +0.02-0.04 | Medium-term |
| 5. Outlier isolation | Medium | +0.01-0.03 | Medium-term |
| 6. 50/50 cosine + MSE loss | Low | +0.005-0.02 | Immediate |
| 7. PyTorch `gumbel_softmax(hard=True)` | Low | +0.001-0.005 | Immediate |
| 8. OOB indices sentinel | Low | +0.0005-0.001 | Immediate |
| 9. NaN guard | Low | 0 (correctness) | Immediate |
| 10. AQLM-style additive VQ | Very High | +0.05-0.10 | Long-term |

**Cumulative impact of patches 1-3 + 6-9 (immediate): +0.015-0.04 cos** (0.946 → ~0.96-0.98).
**Cumulative impact of patches 1-9 (medium-term): +0.04-0.10 cos** (0.946 → ~0.98-0.99).
**Cumulative impact of all 10 patches (long-term): +0.05-0.15 cos** (0.946 → >0.99).

The immediate patches (1-3, 6-9) are low-effort and can be applied in a single afternoon. The medium-term patches (4-5) require new calibration code. The long-term patch (10) is a fundamental rewrite.
