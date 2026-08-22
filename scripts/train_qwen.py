#!/usr/bin/env python3
"""train_qwen.py — Per-super-block distillation training for Qwen3.5-4B.

Architecture per super-block:
  - 4 original layers: 2-bit palettized Linears (palettes TRAINABLE) + everything else fp16
  - 1 correction layer: dense GatedDeltaNet (fp16, trainable)
  - LoRA rank-32 ONLY on the correction layer's Linears (fp16, trainable)

Live JSON control (/tmp/hyperparams_qwen.json):
  - "groups": { "palettes": true, "lora": true, "correction": true, "layernorms": true }
    → Set to false to freeze that group LIVE (no restart)
  - "lrs": { "palettes": 1e-4, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4 }
    → Set LR per group LIVE
  - "loss_type": "1-cos" | "norm_mse" | "1-cos+norm_mse"
  - "loss_weights": { "cos": 0.5, "mse": 0.5 }
  - "gradient_clip": 0.3
  - "eval_every": 250
  - "log_every": 50

NO FP32 weights. NO pre-transpose. Shared embeddings.
"""
import os, sys, json, time, argparse, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import BITWIDTH, GROUP_SIZE, PALETTE_SIZE, load_indices, load_lut
from qwen_model import (
    SUPER_BLOCKS, PalettizedLinear, QwenLoRA,
    load_qwen_super_block_only, load_qwen_model,
    insert_correction_layers, attach_lora_to_layer,
    should_palettize, palettize_linear, capture_original_weights,
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
PALETTIZED_BASE = "/root/qwen35_palettize/palettized"
TRAINED_BASE = "/root/qwen35_palettize/trained"
LOGS_DIR = "/root/qwen35_palettize/logs"
HYPERPARAMS_FILE = "/tmp/hyperparams_qwen.json"

# Auto-redirect stdout + stderr to logs/train_sb{N}.log (OVERWRITE mode).
# Set FORCE_STDOUT=1 env var to disable (e.g., for IDE debugging).
if os.environ.get("FORCE_STDOUT") != "1":
    os.makedirs(LOGS_DIR, exist_ok=True)
    # We'll re-set the path once sb_idx is known — for now point to a default.
    class _DualStream:
        """Write to both the original stdout and the log file (tee)."""
        def __init__(self, log_path):
            self._stdout = sys.stdout
            self._file = open(log_path, "w", buffering=1)  # line-buffered, OVERWRITE
            self._log_path = log_path
        def write(self, s):
            self._stdout.write(s)
            self._file.write(s)
        def flush(self):
            self._stdout.flush()
            self._file.flush()
        def rewire(self, log_path):
            self._file.close()
            self._file = open(log_path, "w", buffering=1)
            self._log_path = log_path
    _dual = _DualStream(os.path.join(LOGS_DIR, "train_sb0.log"))
    sys.stdout = _dual
    sys.stderr = _dual
    # Helper to re-point the log file once sb_idx is parsed
    def _rewire_log(sb_idx):
        new_path = os.path.join(LOGS_DIR, f"train_sb{sb_idx}.log")
        _dual.rewire(new_path)
        print(f"[log] writing to {new_path} (overwrite mode)", flush=True)

DEFAULT_HYPERPARAMS = {
    "groups": {
        "palettes": True,
        "lora": True,
        "correction": True,
        "layernorms": True,
    },
    "lrs": {
        # Bumped LRs (step 250 analysis: curve climbing too slowly).
        #   palettes: 1e-4 -> 3e-3 (30x). Only 2,208 params, super dense grad,
        #             Muon scale ~0.63 makes effective LR very low (6.3e-5).
        #             30x brings effective to ~1.9e-3, in line with correction.
        #   lora:     3e-4 -> 1e-3 (3x).  Standard LoRA range on AdamW.
        #   correction: 1e-4 -> 2e-4 (2x).  Muon scale ~18 makes effective LR
        #             already ~1.8e-3, so modest bump.
        #   layernorms: 1e-4 -> 3e-4 (3x).  Norms can take 3x safely.
        "palettes": 3e-3,
        "lora": 1e-3,
        "correction": 2e-4,
        "layernorms": 3e-4,
    },
    "loss_type": "norm_mse",
    "loss_weights": {"cos": 0.0, "mse": 1.0},
    "gradient_clip": 0.3,
    "eval_every": 250,
    "log_every": 50,
    "save_every": 2000,
}


# ─── Muon optimizer ─────────────────────────────────────────────────────
class Muon(torch.optim.Optimizer):
    """Muon optimizer (Newton-Schulz orthogonalized momentum).

    Reverted to Dolphin v13's clean setup:
      - No per-group weight_decay (always 0.0)
      - No per-group scale_factor override (always 0.2*sqrt(max(A,B)))
      - Operates on fp32 master params (via FP32MasterMuon wrapper)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            mu = group['momentum']; lr = group['lr']; wd = group.get('weight_decay', 0.0)
            ns_steps = group['ns_steps']; nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim < 2:
                    if wd > 0: p.mul_(1 - lr * wd)
                    p.add_(g, alpha=-lr)
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(mu).add_(g)
                g_eff = g.add(buf, alpha=mu) if nesterov else buf
                o = self._newton_schulz(g_eff, steps=ns_steps)
                A, B = g.shape[-2], g.shape[-1]
                scale = 0.2 * (max(A, B) ** 0.5)
                if wd > 0: p.mul_(1 - lr * wd)
                p.add_(o, alpha=-lr * scale)

    @torch.no_grad()
    def _newton_schulz(self, X, steps=5, eps=1e-7):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = X / (X.norm() + eps)
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X


# ─── FP32 Master optimizer wrappers ────────────────────────────────────
class FP32MasterOptimizer:
    """Base class for optimizers with fp32 master weights.

    Maintains fp32 copies of all params. At step():
      1. Copy bf16 model grads → fp32 master grads
      2. Step on fp32 masters
      3. Copy fp32 masters back → bf16 model params
    """

    def __init__(self, param_groups, opt_class, **opt_kwargs):
        self.model_param_map = {}  # id(fp32_master) → bf16 model param
        self.master_grads = {}     # id(fp32_master) → pre-allocated fp32 grad buffer
        new_param_groups = []
        for group in param_groups:
            new_group = dict(group)
            new_params = []
            for p in group["params"]:
                master = p.data.float().clone()
                master.requires_grad_(True)
                self.model_param_map[id(master)] = p
                # Pre-allocate fp32 grad buffer (avoids per-step allocation)
                self.master_grads[id(master)] = torch.empty_like(master)
                new_params.append(master)
            new_group["params"] = new_params
            new_param_groups.append(new_group)
        self.opt = opt_class(new_param_groups, **opt_kwargs)

    @property
    def param_groups(self):
        return self.opt.param_groups

    @property
    def state(self):
        return self.opt.state

    def step(self, closure=None):
        # Copy bf16 grads → pre-allocated fp32 master grads (no allocation here)
        for group in self.opt.param_groups:
            for master in group["params"]:
                p = self.model_param_map[id(master)]
                if p.grad is not None:
                    grad_buf = self.master_grads[id(master)]
                    grad_buf.copy_(p.grad)  # bf16 → fp32 in-place (single copy_ kernel)
                    master.grad = grad_buf
                else:
                    master.grad = None
        self.opt.step(closure=closure)
        # Copy fp32 masters → bf16 model params (in-place copy_, no allocation)
        with torch.no_grad():
            for group in self.opt.param_groups:
                for master in group["params"]:
                    p = self.model_param_map[id(master)]
                    p.data.copy_(master.data)

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)


class FP32MasterAdamW(FP32MasterOptimizer):
    """AdamW with fp32 master weights."""
    def __init__(self, param_groups, **adamw_kwargs):
        super().__init__(param_groups, torch.optim.AdamW, **adamw_kwargs)


class FP32MasterMuon(FP32MasterOptimizer):
    """Muon with fp32 master weights."""
    def __init__(self, param_groups, **muon_kwargs):
        super().__init__(param_groups, Muon, **muon_kwargs)


# ─── Loss ──────────────────────────────────────────────────────────────
def normalize_weights(w):
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else w

def compute_loss(student_out, teacher_out, hp):
    """Compute loss. fp32 math. Handles zero-norm student output gracefully."""
    s = student_out.float()
    t = teacher_out.detach().float()
    w = normalize_weights(hp.get("loss_weights", {"cos": 0.5, "mse": 0.5}))
    # Use larger eps to handle zero-norm student output (correction layer zero-init)
    cos_per = F.cosine_similarity(s.flatten(0, 1), t.flatten(0, 1), dim=-1, eps=1e-4)
    l_cos = (1 - cos_per).mean()
    loss_type = hp.get("loss_type", "1-cos+norm_mse")
    if loss_type == "1-cos":
        loss = l_cos
    elif loss_type == "1-cos+norm_mse":
        t_var = (t * t).mean().clamp(min=1e-6)
        l_mse = ((s - t) ** 2).mean() / t_var
        loss = w["cos"] * l_cos + w["mse"] * l_mse
    elif loss_type == "norm_mse":
        t_var = (t * t).mean().clamp(min=1e-6)
        loss = ((s - t) ** 2).mean() / t_var
    else:
        loss = l_cos
    return loss, {"cos": l_cos.item(), "loss": loss.item()}


# ─── Solution A: Iterative palette freezing ────────────────────────────
def freeze_settled_palettes(model, sb_idx, prev_snapshot, curr_snapshot, verbose=True):
    """Freeze palette entries whose index assignment hasn't changed between
    prev_snapshot and curr_snapshot (separated by FREEZE_MIN_STEPS steps).

    This implements Nagel et al. ICML 2022 "Overcoming Oscillations in QAT" —
    weights that keep flipping between two grid points never converge, and freezing
    them stops the oscillation at its source.

    Args:
        model: the student model
        sb_idx: super-block index
        prev_snapshot: dict {param_name: index_tensor} from FREEZE_MIN_STEPS ago
        curr_snapshot: dict {param_name: index_tensor} from current step

    Returns:
        n_frozen: number of palette entries frozen
    """
    from qwen_model import PalettizedLinear
    n_frozen = 0
    n_checked = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        if name not in prev_snapshot or name not in curr_snapshot:
            continue

        n_checked += 1
        recent = curr_snapshot[name]
        older = prev_snapshot[name]
        if recent.shape != older.shape:
            continue

        gs = mod.group_size
        si, so = recent.shape
        n_groups = so // gs
        if n_groups == 0:
            continue

        # Reshape to (in_dim, n_groups, gs) and check if all elements in each group match
        recent_g = recent.reshape(si, n_groups, gs)
        older_g = older.reshape(si, n_groups, gs)
        group_unchanged = (recent_g == older_g).all(dim=(0, 2))  # (n_groups,)

        # Freeze palette entries for unchanged groups by zeroing their grad
        if mod.palette.grad is not None:
            for g_idx in range(n_groups):
                if group_unchanged[g_idx]:
                    mod.palette.grad[g_idx].zero_()
                    n_frozen += 1

    if verbose and n_checked > 0:
        print(f"  [freeze] checked {n_checked} palettes, froze {n_frozen} groups", flush=True)
    return n_frozen


def snapshot_palette_indices(model):
    """Take a snapshot of current palette index assignments for freeze tracking.
    Returns int8 tensors (not int64) to save memory — indices are 0-3 for 2-bit."""
    from qwen_model import PalettizedLinear
    snapshots = {}
    for name, mod in model.named_modules():
        if isinstance(mod, PalettizedLinear):
            # Store as int8 (indices are 0-3 for 2-bit, fits in int8)
            snapshots[name] = mod.indices.to(torch.int8).clone()
    return snapshots


# ─── Solution D: Gradient noise injection ───────────────────────────────
def inject_gradient_noise(model, noise_scale=1e-4, min_grad_mean=1e-6, verbose=False):
    """Add small Gaussian noise to gradients that are too flat (mean abs < min_grad_mean).

    This helps escape flat basins where the gradient is washed out by averaging
    (e.g., palette gradients averaged over 640 weights).

    Noise scale is relative to the gradient's own std, so it scales appropriately.
    Only applied to params with near-zero gradients (flat regions).
    """
    n_injected = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        g_mean = g.abs().mean().item()
        if g_mean < min_grad_mean:
            g_std = g.std().item()
            if g_std > 0:
                noise = torch.randn_like(g) * noise_scale * g_std
                p.grad.add_(noise)
                n_injected += 1
    if verbose and n_injected > 0:
        print(f"  [noise] injected noise into {n_injected} flat gradients", flush=True)
    return n_injected


# ─── Solution F: Cyclic loss schedule ───────────────────────────────────
def get_loss_type_for_step(global_step, cycle_len=100):
    """Cyclic loss: 100 steps pure norm_mse, then 100 steps pure 1-cos, repeat.

    norm_mse pushes magnitude alignment (fixes scale issues from quantization).
    1-cos pushes direction alignment (fixes feature representation).
    Alternating prevents either objective from dominating and getting stuck.
    """
    phase = (global_step // cycle_len) % 2
    if phase == 0:
        return "norm_mse"
    else:
        return "1-cos"


# ─── Eval set ──────────────────────────────────────────────────────────
EVAL_CACHE_PATH = "/root/qwen35_palettize/eval_tokens.pt"
EVAL_N_SEQS = 256
EVAL_SEQ_LEN = 512


def prepare_eval_set(tokenizer, device="cuda"):
    """Load or create a held-out eval set of 256 sequences × 512 tokens.

    These are DIFFERENT sequences from training (no overlap) and are cached
    so every eval uses the same data for fair comparison.
    """
    if os.path.exists(EVAL_CACHE_PATH):
        eval_tokens = torch.load(EVAL_CACHE_PATH, weights_only=False)
        print(f"  Loaded eval set: {eval_tokens.shape} from {EVAL_CACHE_PATH}", flush=True)
        return eval_tokens.to(device)

    print(f"  Creating eval set: {EVAL_N_SEQS} seqs × {EVAL_SEQ_LEN} tokens...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Skip the first 2000 training sequences to avoid overlap
    ds_iter = iter(ds)
    for _ in range(2000):
        try: next(ds_iter)
        except StopIteration: break

    seqs = []
    for ex in ds_iter:
        text = ex.get("text", "")
        if not text or len(text) < 100: continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=EVAL_SEQ_LEN, return_tensors="pt")["input_ids"].squeeze(0)
        if ids.numel() < 64: continue
        if ids.numel() < EVAL_SEQ_LEN:
            pad = torch.full((EVAL_SEQ_LEN - ids.numel(),), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:EVAL_SEQ_LEN]
        seqs.append(ids)
        if len(seqs) >= EVAL_N_SEQS: break

    eval_tokens = torch.stack(seqs)
    torch.save(eval_tokens, EVAL_CACHE_PATH)
    print(f"  Saved eval set: {eval_tokens.shape} → {EVAL_CACHE_PATH}", flush=True)
    return eval_tokens.to(device)


@torch.no_grad()
def evaluate(student, teacher, eval_tokens, sb_idx, hp, max_batches=8):
    """Run proper evaluation on held-out data.

    Computes cos + norm_mse on the eval set (not training data).
    Uses the teacher's prefix to get h_in, then runs student forward and compares.

    Returns: dict with "cos", "loss", "n_seqs"
    """
    from qwen_model import SUPER_BLOCKS
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    student.eval()
    total_cos = 0.0
    total_loss = 0.0
    n_batches = 0

    batch_size = eval_tokens.shape[0] // max_batches
    for i in range(max_batches):
        batch_ids = eval_tokens[i*batch_size:(i+1)*batch_size]
        if batch_ids.shape[0] == 0: continue

        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            # Teacher prefix
            h = teacher.model.embed_tokens(batch_ids)
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
            pos_emb = None
            if hasattr(teacher.model, 'rotary_emb') and teacher.model.rotary_emb is not None:
                pos_emb = teacher.model.rotary_emb(h, position_ids)
            for layer_idx in range(sb_end):
                if layer_idx < len(teacher.model.layers):
                    layer = teacher.model.layers[layer_idx]
                    out = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
                    h = out[0] if isinstance(out, tuple) else out
            h_out = h.detach()

            # Student forward
            s_h = student.model.embed_tokens(batch_ids)
            s_pos_emb = None
            if hasattr(student.model, 'rotary_emb') and student.model.rotary_emb is not None:
                s_pos_emb = student.model.rotary_emb(s_h, position_ids)
            for layer_idx in range(sb_end + 1):
                if layer_idx < len(student.model.layers):
                    layer = student.model.layers[layer_idx]
                    out = layer(s_h, position_embeddings=s_pos_emb) if s_pos_emb is not None else layer(s_h)
                    s_h = out[0] if isinstance(out, tuple) else out
            student_out = s_h

        loss, comps = compute_loss(student_out, h_out, hp)
        total_cos += (1.0 - comps["cos"])
        total_loss += comps["loss"]
        n_batches += 1
        del batch_ids, h_out, student_out, loss, comps

    student.train()
    if n_batches == 0:
        return {"cos": -1.0, "loss": float('inf'), "n_seqs": 0}
    return {
        "cos": total_cos / n_batches,
        "loss": total_loss / n_batches,
        "n_seqs": n_batches * batch_size,
    }


# ─── Hyperparams ────────────────────────────────────────────────────────
def write_default_hyperparams():
    if not os.path.exists(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE, "w") as f:
            json.dump(DEFAULT_HYPERPARAMS, f, indent=2)

def load_hyperparams():
    if not os.path.exists(HYPERPARAMS_FILE):
        write_default_hyperparams()
    with open(HYPERPARAMS_FILE, "r") as f:
        return json.load(f)


# ─── Classify params into groups ───────────────────────────────────────
def classify_param(name, sb_idx, n_correction=1):
    """Classify a parameter into a group for freeze/unfreeze + LR control.
    Returns: one of "palettes", "lora", "indices", "layernorms", "frozen"

    Groups:
      - palettes: palette values (G, 4) — trainable LUT entries
      - lora: LoRA A/B matrices — rank-16 adapters on all Linears
      - indices: index_logits (4, K, N) — Gumbel-Softmax logits for trainable indices
      - layernorms: norms, biases, SSM params, conv1d — small params in original layers
      - frozen: embed_tokens and everything outside the super-block
    """
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    parts = name.split(".")

    # Parse layer index
    layer_idx = None
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        layer_idx = int(parts[2])
    elif len(parts) >= 2 and parts[0] == "layers":
        layer_idx = int(parts[1])

    # Check if it's index_logits (trainable indices via Gumbel-Softmax)
    if "index_logits" in name:
        return "indices"

    # Check if it's a palette
    if "palette" in name:
        return "palettes"

    # Check if it's LoRA
    if "lora_A" in name or "lora_B" in name:
        return "lora"

    # Check if in super-block's original layers [sb_start, sb_end)
    if layer_idx is not None and sb_start <= layer_idx < sb_end:
        return "layernorms"

    # Everything else: embed_tokens, final norm, etc.
    return "frozen"


# ─── Apply freeze/unfreeze from JSON ───────────────────────────────────
def apply_groups(model, hp, sb_idx):
    """Set requires_grad based on hp["groups"]. Returns param counts per group.

    Groups dict from JSON controls which groups are trainable:
      {"palettes": True, "lora": True, "correction": True, "layernorms": True}
    The "frozen" group (embed_tokens, etc.) is ALWAYS frozen — not controllable
    from JSON, because training embeddings would break distillation.
    """
    groups = hp.get("groups", {"palettes": True, "lora": True, "correction": True, "layernorms": True})
    counts = {"palettes": 0, "lora": 0, "correction": 0, "layernorms": 0, "frozen": 0}
    for name, p in model.named_parameters():
        group = classify_param(name, sb_idx)
        # "frozen" group is always frozen; other groups default to True if not in JSON
        if group == "frozen":
            should_train = False
        else:
            should_train = groups.get(group, True)
        p.requires_grad_(should_train)
        if should_train:
            counts[group] = counts.get(group, 0) + p.numel()
        else:
            counts["frozen"] = counts.get("frozen", 0) + p.numel()
    return counts


# ─── FP32 Master AdamW ─────────────────────────────────────────────────
# ─── Build optimizers with per-group LR ────────────────────────────────
def build_optimizers(model, hp, sb_idx):
    """Build Muon (2D non-palette) + AdamW (palettes + LoRA + 1D).

    Reverted to Dolphin v13's proven setup:
      - Muon for 2D non-palette weights (correction layer dense weights)
      - AdamW for palettes + LoRA + 1D params (palettes are tiny vectors, Muon's
        NS orthogonalization is inappropriate for them)
      - Both optimizers use fp32 master weights
      - No weight_decay (0.0)
      - AdamW beta2=0.95 (Dolphin's value)
      - Muon scale_factor=0.2*sqrt(max(A,B)) (default, no per-group override)
    """
    lrs = hp.get("lrs", {"palettes": 1e-4, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4})
    muon_groups = []
    adamw_groups = []
    plain_adamw_groups = []  # for index_logits — no fp32 master (saves 7 GB)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = classify_param(name, sb_idx)
        lr = lrs.get(group, 1e-4)
        if group == "palettes":
            # Palettes → AdamW with fp32 master (tiny, benefits from precision)
            adamw_groups.append({"params": [p], "lr": lr, "name": name, "group": group})
        elif group == "lora":
            # LoRA → AdamW with fp32 master
            adamw_groups.append({"params": [p], "lr": lr, "name": name, "group": group})
        elif group == "indices":
            # index_logits → plain AdamW WITHOUT fp32 master (too large: 1.78B params)
            # Using bf16/fp16 AdamW directly saves 7 GB of fp32 master memory
            plain_adamw_groups.append({"params": [p], "lr": lr, "name": name, "group": group})
        elif p.ndim >= 2:
            # 2D non-palette non-indices → Muon with fp32 master
            muon_groups.append({"params": [p], "lr": lr, "name": name, "group": group})
        else:
            # 1D params → AdamW with fp32 master
            adamw_groups.append({"params": [p], "lr": lr, "name": name, "group": group})

    # FP32 master optimizers for palettes, LoRA, layernorms, Muon params
    opt_muon = FP32MasterMuon(muon_groups, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0) if muon_groups else None
    opt_adamw = FP32MasterAdamW(adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) if adamw_groups else None
    # index_logits optimizer: bitsandbytes AdamW8bit (Patch 8, Wave 1).
    # Replaces FP32MasterAdamW (113 ms/step, 21.4 GB VRAM) with bnb.optim.AdamW8bit
    # (~20 ms/step, 3.56 GB VRAM). 8-bit state (m, v) handles fp16 index_logits
    # natively — no fp32 master copy needed.
    #
    # eps=1e-6 (Wave 3 defensive bump from 1e-8): 8-bit quantization introduces
    # ~1/256 state-range noise on dequantization. For tiny Gumbel-Softmax grads
    # at low τ, sqrt(v)+eps with eps=1e-8 risks underflow → NaN (the original
    # FP32MasterAdamW had a CRITICAL warning about this for fp16 state). 1e-6 is
    # the bitsandbytes-recommended floor for 8-bit state and matches the Wave 3
    # eps=1e-6 for 8-bit state stability
    # convergence; benefit: NaN safety without an fp32 master copy. If NaN still
    # appears in real training, the existing skip-and-continue guard at line ~1161
    # (if not torch.isfinite(loss): ... continue) plus the clamp_(-20, 20) at
    # line ~1209 will keep training stable.
    # See research-filter-consolidation/03_optimizer_speedup.md §Patch 8 (Option B),
    # docs/papers/1412.6980_Adam_Kingma2015.pdf + 1711.05101_AdamW_Loshchilov2019.pdf.
    import bitsandbytes as bnb
    opt_indices = bnb.optim.AdamW8bit(
        plain_adamw_groups, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.0
    ) if plain_adamw_groups else None

    # Per-group counts for visibility
    muon_by = {}
    for g in muon_groups:
        k = g.get("group", "?")
        muon_by[k] = muon_by.get(k, 0) + 1
    adamw_by = {}
    for g in adamw_groups:
        k = g.get("group", "?")
        adamw_by[k] = adamw_by.get(k, 0) + 1

    print(f"  Muon  groups: {len(muon_groups):3d}  — {muon_by}  (fp32 master)", flush=True)
    print(f"  AdamW groups: {len(adamw_groups):3d}  — {adamw_by}  (fp32 master, beta2=0.95)", flush=True)
    return opt_muon, opt_adamw, opt_indices


# ─── Update LRs live from JSON ────────────────────────────────────────
def update_lrs(opt_muon, opt_adamw, hp, sb_idx, sched_muon=None, sched_adamw=None,
                opt_indices=None, sched_indices=None):
    """Update per-group LRs from JSON (live, no restart).

    CRITICAL: also update scheduler.base_lrs, otherwise the cosine
    scheduler's .step() will overwrite g["lr"] back to the original
    base_lr * lambda(step) on the next step, silently ignoring our update.

    Set any group LR to 0 in JSON to freeze it (e.g., palettes: 0 to keep
    the trained palettes intact while tuning other groups).
    """
    lrs = hp.get("lrs", {"palettes": 3e-3, "lora": 1e-3, "indices": 1e-2, "layernorms": 3e-4})
    if opt_muon:
        for i, g in enumerate(opt_muon.param_groups):
            group = g.get("group", "layernorms")
            new_lr = lrs.get(group, 1e-4)
            g["lr"] = new_lr
            g["initial_lr"] = new_lr  # so scheduler picks it up
            if sched_muon is not None and i < len(sched_muon.base_lrs):
                sched_muon.base_lrs[i] = new_lr
    if opt_adamw:
        for i, g in enumerate(opt_adamw.param_groups):
            group = g.get("group", "lora")
            new_lr = lrs.get(group, 1e-4)
            g["lr"] = new_lr
            g["initial_lr"] = new_lr
            if sched_adamw is not None and i < len(sched_adamw.base_lrs):
                sched_adamw.base_lrs[i] = new_lr
    if opt_indices:
        # opt_indices is bnb.optim.AdamW8bit (Patch 8) — NOT an FP32MasterOptimizer
        # wrapper, so we access .param_groups DIRECTLY (not via .opt.param_groups).
        # Round 1 fix: verified this matches scheduler init at line ~994 which
        # uses LambdaLR(opt_indices, ...) — also no .opt wrapper.
        for i, g in enumerate(opt_indices.param_groups):
            group = g.get("group", "indices")
            new_lr = lrs.get(group, 1e-2)
            g["lr"] = new_lr
            g["initial_lr"] = new_lr
            if sched_indices is not None and i < len(sched_indices.base_lrs):
                sched_indices.base_lrs[i] = new_lr


# ─── Build student super-block ──────────────────────────────────────────
def build_student_super_block(sb_idx, lora_rank=16, lora_alpha=32, use_soft_indices=False):
    """Build student for one super-block.

    Architecture (trainable indices + LoRA-only):
      1. Load prefix model (embed_tokens + layers 0 to sb_end-1)
      2. Replace ALL palettizable Linears with PalettizedLinear (2-bit, soft indices)
      3. Attach LoRA rank-16 on ALL palettized Linears (compensates for residual quant error)
      No correction layer. No stage-2 palettization.
    """
    print(f"\n=== Building student super-block {sb_idx} (soft_indices={use_soft_indices}) ===", flush=True)

    palettized_dir = os.path.join(PALETTIZED_BASE, f"superblock_{sb_idx}")
    if not os.path.isdir(palettized_dir):
        print(f"  ERROR: palettized dir not found: {palettized_dir}", flush=True)
        return None, None

    # Load prefix model
    model, tokenizer = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    # Replace original layers' Linears with PalettizedLinear (layers sb_start to sb_end-1)
    n_palettized = 0
    for layer_idx in range(sb_start, sb_end):
        layer = model.model.layers[layer_idx]
        linears_to_palettize = []
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Strip .base suffix if present (PalettizedLinear is wrapped as .base)
                clean_name = name.replace(".base", "")
                full_name = f"model.layers.{layer_idx}.{clean_name}.weight"
                if should_palettize(full_name, module.weight):
                    linears_to_palettize.append((name, module, full_name))
        for name, module, full_name in linears_to_palettize:
            pal_mod = palettize_linear(module, full_name, palettized_dir, device=DEVICE,
                                       use_soft_indices=use_soft_indices)
            if pal_mod is not None:
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], pal_mod)
                n_palettized += 1
    print(f"  Palettized {n_palettized} Linears to 2-bit (soft_indices={use_soft_indices})", flush=True)

    # Capture original fp16 weights from the HF checkpoint for LoftQ SVD init.
    # Patch 2 (LoftQ): QwenLoRA init="loftq" needs the ORIGINAL pre-palettization
    # weight to compute R = W_orig - W_quantized and take its SVD. Previously
    # this was passed as None, causing QwenLoRA to fall back to zero-init B
    # (qwen_model.py:223-225), wasting the first ~1000 steps climbing out of
    # the zero-init valley. See docs/papers/2305.14314_QLoRA_Dettmers2023.pdf §3.2.
    #
    # We capture from the HF checkpoint (not from the loaded wrapper) because
    # the wrapper's nn.Linear modules have already been replaced by
    # PalettizedLinear above — the original weights are gone from `model`.
    # The checkpoint is the only source of truth for pre-palettization weights.
    #
    # Weights are placed on DEVICE (= "cuda") explicitly so the SVD inside
    # QwenLoRA.__init__ (qwen_model.py:217-222) runs on-GPU directly without
    # a host→device copy per-tensor. This matches the function default but is
    # passed explicitly for clarity at the call site.
    from qwen_model import capture_original_weights_from_checkpoint
    original_weights = capture_original_weights_from_checkpoint(sb_idx, device=DEVICE)

    # Attach LoRA on ALL palettized Linears.
    # Use rank-32 for the 5 worst-cosine Linears (from calib_sb0.log),
    # rank-16 for the rest. These 5 had cos < 0.93 after calibration.
    total_lora = 0
    BIG_LORA_RANK = lora_rank * 2  # 32 if lora_rank=16
    BIG_LORA_ALPHA = lora_alpha * 2  # 64 if lora_alpha=32
    # Format: (layer_idx, attr_path) — relative to layer
    BIG_LORA_TARGETS = {
        (0, "linear_attn.in_proj_z"),
        (1, "mlp.down_proj"),
        (2, "linear_attn.out_proj"),
        (2, "linear_attn.in_proj_qkv"),
        (3, "self_attn.k_proj"),
    }
    for layer_idx in range(sb_start, sb_end):
        layer = model.model.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, (PalettizedLinear, nn.Linear)) and not isinstance(module, QwenLoRA):
                # Check if this module is in BIG_LORA_TARGETS
                is_big = (layer_idx, name) in BIG_LORA_TARGETS
                rank = BIG_LORA_RANK if is_big else lora_rank
                alpha = BIG_LORA_ALPHA if is_big else lora_alpha
                # Look up the original pre-palettization weight for LoftQ SVD init.
                # full_name format matches capture_original_weights_from_checkpoint keys:
                #   "model.layers.{idx}.{submodule.path}.weight"
                # Strip .base suffix if present (PalettizedLinear is wrapped as .base)
                clean_name = name.replace(".base", "")
                full_name = f"model.layers.{layer_idx}.{clean_name}.weight"
                orig_w = original_weights.get(full_name)
                if orig_w is None:
                    # Should not happen — every palettized Linear has a corresponding
                    # original weight. Log and fall back to zero-init if it does.
                    print(f"    WARNING: no original weight for {full_name} — LoRA falls back to zero-init", flush=True)
                lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=orig_w)
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora_mod)
                total_lora += 1
    print(f"  Attached LoRA (rank-{lora_rank} + rank-{BIG_LORA_RANK} on 5 worst-cos) to {total_lora} Linears across {sb_end - sb_start} layers", flush=True)

    # Free the original weights — they're no longer needed after LoRA init.
    # QwenLoRA has already consumed them for SVD; keeping them would waste
    # ~1-2 GB of CPU RAM for the rest of build_student_super_block.
    del original_weights
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cast to dtype
    model.to(DTYPE)
    # Restore index_logits to fp16 (model.to(bf16) converts them to bf16,
    # but the soft CUDA kernel requires fp16 logits)
    # PartialWrapper is now an nn.Module (Patch 9), so model.apply() would
    # work, but we keep the explicit layer loop for parity with the training
    # loop's streaming forward pattern (which calls layers directly).
    for layer in model.model.layers:
        for submod in layer.modules():
            if isinstance(submod, PalettizedLinear) and submod.index_logits is not None:
                submod.index_logits.data = submod.index_logits.data.to(torch.float16)
    print(f"  Cast all params to {DTYPE} (index_logits kept fp16)", flush=True)

    return model, tokenizer


# ─── Save / load state — uses .idx2 + .lut_scalar format (same as palettized/) ─
def save_state(model, sb_idx, step, cos, loss, out_dir):
    """Save trained state using the palettized format:
      - .idx2 for indices (2-bit packed, argmax of index_logits if soft)
      - .lut_scalar for palettes
      - .pt for LoRA + layernorms (small)
    """
    from palettize_core import pack_idx2, write_lut_scalar, sanitize_name
    from qwen_model import PalettizedLinear

    os.makedirs(out_dir, exist_ok=True)
    n_idx2 = 0
    n_lut = 0
    n_pt = 0

    # Save indices (.idx2) + palettes (.lut_scalar) from PalettizedLinear modules
    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        # Indices: argmax of logits if soft, else frozen indices
        if mod.index_logits is not None:
            indices = mod.index_logits.argmax(dim=0)  # (K, N) — trained assignment
        else:
            indices = mod.indices  # (K, N) — frozen
        indices_int8 = indices.to(torch.uint8).cpu()

        san = sanitize_name(name)
        # Pack as 2-bit .idx2
        packed = pack_idx2(indices_int8)
        with open(os.path.join(out_dir, f"{san}.idx2"), "wb") as f:
            f.write(packed)
        n_idx2 += 1

        # Palette as .lut_scalar
        write_lut_scalar(os.path.join(out_dir, f"{san}.lut_scalar"), mod.palette.detach().cpu())
        n_lut += 1

    # Save LoRA + layernorms + other small trainable params as .pt
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "index_logits" in name:
            continue  # saved as .idx2 above
        if "palette" in name:
            continue  # saved as .lut_scalar above
        safe_name = name.replace(".", "_")
        torch.save(p.detach().cpu(), os.path.join(out_dir, f"{safe_name}.pt"))
        n_pt += 1

    with open(os.path.join(out_dir, "_resume.json"), "w") as f:
        json.dump({"step": step, "cos": cos, "loss": loss, "sb_idx": sb_idx}, f)
    print(f"  Saved {n_idx2} .idx2 + {n_lut} .lut_scalar + {n_pt} .pt "
          f"(step={step}, cos={cos:.6f})", flush=True)


def load_state(model, resume_dir):
    """Load trained state from .idx2 + .lut_scalar + .pt format."""
    from palettize_core import load_indices, load_lut, sanitize_name
    from qwen_model import PalettizedLinear

    if not os.path.isdir(resume_dir):
        print(f"  Resume dir not found: {resume_dir}", flush=True)
        return 0, -1.0, float('inf')
    resume_json = os.path.join(resume_dir, "_resume.json")
    if not os.path.exists(resume_json):
        print(f"  _resume.json not found in {resume_dir}", flush=True)
        return 0, -1.0, float('inf')
    with open(resume_json, "r") as f:
        meta = json.load(f)
    step = meta.get("step", 0)
    cos = meta.get("cos", -1.0)
    loss = meta.get("loss", float('inf'))

    n_loaded = 0
    n_missing = 0

    # Load indices (.idx2) + palettes (.lut_scalar) into PalettizedLinear modules
    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        san = sanitize_name(name)
        idx_path = os.path.join(resume_dir, f"{san}.idx2")
        lut_path = os.path.join(resume_dir, f"{san}.lut_scalar")

        if os.path.exists(idx_path):
            K, N = mod.indices.shape
            indices = load_indices(idx_path, N, K).to(mod.indices.device)
            mod.indices = indices.long()
            mod.indices_int8 = indices.to(torch.int8).contiguous()
            # Update index_logits — SOFT one-hot (±1, not ±10) so gradients flow.
            # STE (in fused_lut_linear_cuda.py) keeps forward=hard (cos preserved).
            # Soft logits make P non-degenerate → grad_logits is non-zero.
            # At ±10: P[non-argmax]≈0.00003 → grad≈0 (indices frozen).
            # At ±1:   P[non-argmax]≈0.09    → grad flows (indices train).
            if mod.index_logits is not None:
                with torch.no_grad():
                    mod.index_logits.data.fill_(-1.0)
                    for k in range(4):
                        mask = (indices == k)
                        mod.index_logits.data[k][mask] = 1.0
                    mod.index_logits.data = mod.index_logits.data.to(torch.float16)
            n_loaded += 1
        else:
            n_missing += 1

        if os.path.exists(lut_path):
            lut = load_lut(lut_path).reshape(mod.n_groups, mod.palette_size).to(mod.palette.device)
            mod.palette.data.copy_(lut.to(mod.palette.dtype))
            n_loaded += 1
        else:
            n_missing += 1

    # Load LoRA + layernorms from .pt
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "index_logits" in name:
            continue  # loaded from .idx2 above
        if "palette" in name:
            continue  # loaded from .lut_scalar above
        safe_name = name.replace(".", "_")
        pt_path = os.path.join(resume_dir, f"{safe_name}.pt")
        if os.path.exists(pt_path):
            saved = torch.load(pt_path, map_location=p.device, weights_only=True)
            if saved.shape == p.data.shape:
                p.data.copy_(saved.to(p.dtype))
                n_loaded += 1
            else:
                # Shape mismatch (e.g., rank-16 saved → rank-32 model)
                # Copy what fits, leave the rest zero-init (LoRA B-row extension).
                min_dims = tuple(min(s, m) for s, m in zip(saved.shape, p.data.shape))
                slices = tuple(slice(0, d) for d in min_dims)
                p.data.zero_()
                p.data[slices].copy_(saved[slices].to(p.dtype))
                n_loaded += 1
                if n_loaded <= 5:  # only print first 5 mismatches
                    print(f"    shape mismatch for {name}: saved {list(saved.shape)} → model {list(p.data.shape)}, copied {min_dims}", flush=True)
        else:
            n_missing += 1

    print(f"  Resumed {n_loaded} params from {resume_dir} (step={step}, cos={cos:.6f})", flush=True)
    if n_missing > 0:
        print(f"  WARNING: {n_missing} params not found in resume dir "
              f"(will use fresh init for those)", flush=True)
    return step, cos, loss


# ─── Streaming training data ───────────────────────────────────────────
def stream_training_data(tokenizer, n_seqs, seq_len, device="cuda", batch_size=8):
    """Stream batches of tokenized sequences from FineWeb-Edu.

    Each yielded batch has shape (batch_size, seq_len) on `device`.
    - seq_len is small (128) so we can afford a larger batch_size
      to keep the L4 GPU busy despite the small per-seq FLOPs.
    - batch_size is the # of independent sequences per training step.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buf = []
    produced = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 100: continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"].squeeze(0)
        if ids.numel() < 64: continue
        if ids.numel() < seq_len:
            pad = torch.full((seq_len - ids.numel(),), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:seq_len]
        buf.append(ids)
        while len(buf) >= batch_size and produced < n_seqs:
            batch = torch.stack(buf[:batch_size]).to(device)
            del buf[:batch_size]
            produced += batch_size
            yield batch
        if produced >= n_seqs: return


# ─── Main training loop ────────────────────────────────────────────────
def train_super_block(sb_idx, max_steps, lora_rank=16, lora_alpha=32, seq_len=128, batch_size=8,
                      resume_from=None, use_soft_indices=False,
                      tau_init=2.0, tau_final=0.5, tau_anneal_steps=6000,
                      shutdown_on_done=False):
    _rewire_log(sb_idx)
    print(f"\n{'='*70}")
    print(f"=== Training super-block {sb_idx} (soft_indices={use_soft_indices}) ===")
    print(f"  max_steps: {max_steps}  lora_rank: {lora_rank}")
    print(f"  seq_len: {seq_len}  batch_size: {batch_size}")
    print(f"  tokens/step: {seq_len * batch_size}")
    if use_soft_indices:
        print(f"  tau: {tau_init} → {tau_final} over {tau_anneal_steps} steps")
    if resume_from:
        print(f"  resume_from: {resume_from}")
    print(f"{'='*70}")

    write_default_hyperparams()
    hp = load_hyperparams()

    # Build student (no correction layer, LoRA on all Linears, soft indices)
    student, tokenizer = build_student_super_block(sb_idx, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                                     use_soft_indices=use_soft_indices)
    if student is None: return

    # Apply initial groups (everything trainable)
    counts = apply_groups(student, hp, sb_idx)
    print(f"\n=== Param counts ===", flush=True)
    for g, c in counts.items():
        print(f"  {g}: {c:,}", flush=True)

    # Resume if requested (BEFORE building optimizers, so params are loaded first)
    # NOTE: best_cos/best_step/best_loss are initialized AFTER this block (below),
    # so we only capture resume_* values here; they get promoted to best_* later.
    resume_step = 0
    resume_cos = -1.0
    resume_loss = float('inf')
    if resume_from:
        resume_step, resume_cos, resume_loss = load_state(student, resume_from)

    # Build optimizers (after resume, so loaded params are in place)
    print(f"\n=== Building optimizers ===", flush=True)
    opt_muon, opt_adamw, opt_indices = build_optimizers(student, hp, sb_idx)

    # Cosine LR scheduler with linear warmup (100 steps) — ABSOLUTE step.
    # Without warmup, STE + full LR causes index flips that destroy the model.
    # Warmup lets the model adapt to STE before full LR kicks in.
    WARMUP_STEPS = 100
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP_STEPS) / max(max_steps - WARMUP_STEPS, 1)))
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon.opt, lr_lambda) if opt_muon else None
    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw.opt, lr_lambda) if opt_adamw else None
    # opt_indices is bnb.optim.AdamW8bit (Patch 8) — no .opt wrapper indirection.
    # opt_muon / opt_adamw still use FP32MasterOptimizer wrapper, so they keep .opt.
    sched_indices = torch.optim.lr_scheduler.LambdaLR(opt_indices, lr_lambda) if opt_indices else None

    # If resuming, advance the LR scheduler to the resumed step
    if resume_step > 0:
        if sched_muon: sched_muon.step(epoch=resume_step)
        if sched_adamw: sched_adamw.step(epoch=resume_step)
        if sched_indices: sched_indices.step(epoch=resume_step)
        print(f"  Advanced LR scheduler to step {resume_step}", flush=True)

    # Load teacher — SAME PREFIX as student (embed_tokens + layers 0 to sb_end-1)
    # NO full model! We only need the prefix to compute teacher's super-block output.
    print(f"\n=== Loading teacher (prefix only, same as student) ===", flush=True)
    teacher, _ = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    # Freeze teacher (it's a PartialWrapper, not nn.Module — iterate manually)
    for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
    for layer in teacher.model.layers:
        for p in layer.parameters(): p.requires_grad_(False)

    # Share embeddings: student uses teacher's embed_tokens
    student.model.embed_tokens = teacher.model.embed_tokens

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    # Set student to train mode
    student.train()

    # Training loop
    print(f"\n=== Training ===", flush=True)
    print(f"  Loss: {hp['loss_type']}  weights: {hp['loss_weights']}", flush=True)
    print(f"  LRs: {hp['lrs']}", flush=True)
    print(f"  Groups: {hp['groups']}", flush=True)

    # Initialize best trackers. If resumed, preserve the resumed best so we
    # don't overwrite a better save from a previous run with a worse one from
    # the current run (e.g., after an LR bump causes a temporary dip).
    if resume_from and resume_step > 0:
        best_cos = resume_cos
        best_step = resume_step
        best_loss = resume_loss
        print(f"  Resumed best: step={best_step} cos={best_cos:.6f} (saves will only trigger if beaten)", flush=True)
    else:
        best_cos = -1.0
        best_step = 0
        best_loss = float('inf')
    global_step = resume_step
    start_time = time.time()
    n_nan_skip = 0
    last_hp_check = 0
    last_hp_sig = json.dumps(hp, sort_keys=True)

    # Prepare held-out eval set (Solution: proper eval)
    print(f"\n=== Preparing eval set ===", flush=True)
    eval_tokens = prepare_eval_set(tokenizer, device=DEVICE)

    data_stream = stream_training_data(tokenizer, n_seqs=10**12, seq_len=seq_len, device=DEVICE, batch_size=batch_size)

    # ─── Patch 6 (Wave 2): persistent teacher stream + double-buffered h_out ───
    # Eliminates per-step stream creation (~5µs alloc/destroy × 25 Linears =
    # 125 µs wasted per step) AND overlaps the 68 ms teacher forward with the
    # 260 ms student backward by preparing batch N+1's teacher output while
    # the student trains on batch N. Steady-state step time: 530 ms → 461 ms
    # (~13 % speedup; teacher fwd fully hidden behind student bwd).
    #
    # Synchronization invariants (research-kernel-efficiency/06_stream_overlap.md §3.1):
    #   1. Teacher never overwrites a buffer the student is reading —
    #      teacher's stream_t.wait_event(event_s[buf_idx]) at iter start.
    #   2. Student never reads a buffer the teacher is writing —
    #      student's current_stream().wait_event(event_t[buf_idx]) before loss.
    #   3. event_s[buf_idx] is recorded IMMEDIATELY after compute_loss
    #      (before backward) — the student only needs the buffer for the
    #      loss; this lets the NEXT iter's teacher start as soon as loss is
    #      computed, enabling overlap with the current iter's backward.
    stream_t = torch.cuda.Stream()                                  # allocated ONCE
    h_out_buf = [None, None]                                         # ping-pong buffers
    event_t = [torch.cuda.Event(), torch.cuda.Event()]               # teacher-done events
    event_s = [torch.cuda.Event(), torch.cuda.Event()]               # student-done events
    buf_idx = 0                                                      # ping-pong index

    # ─── Patch 21 + 22 (Wave 4): CUDA Graph capture infrastructure ────────
    # Research: research-kernel-efficiency/08_recommendations.md §11
    #           research-kernel-efficiency/06_stream_overlap.md §6
    #
    # WHAT: Capture the full training step as TWO CUDA Graphs (teacher_graph
    #       on stream_t + student_graph on default stream), replayed per step.
    #       Eliminates ~500 kernel launch dispatches × 5µs = 2.5ms of CPU-side
    #       dispatch overhead per step. Patch 22 integrates with the existing
    #       Patch 6 stream double-buffer via event-based synchronization
    #       between the two graphs (teacher fwd fully hidden behind student
    #       compute, made deterministic by graph replay).
    #
    # REQUIREMENTS (08_recommendations.md §11):
    #   1. Static input tensors — batch_ids copied into static_batch_ids
    #      before each replay.
    #   2. No dynamic shapes — batch_size, seq_len fixed at capture time.
    #   3. No data-dependent control flow inside the graph (NaN check,
    #      hyperparameter updates, LR scheduler step are OUTSIDE the graph).
    #   4. 3-5 warmup steps before capture (settles Triton autotuner configs).
    #
    # RECAPTURE POLICY:
    #   tau (Gumbel-Softmax temperature) and per-group LR are baked into the
    #   captured kernels as Python-float kernel args. We re-capture every
    #   CUDA_GRAPH_RECAPTURE_INTERVAL steps to pick up these changes. Over a
    #   6000-step training run, this gives ~60 re-captures × ~50ms each =
    #   ~3 seconds total — negligible vs. the ~600s of training time saved.
    #   We ALSO re-capture when the live-JSON hyperparameters change (forces
    #   immediate re-capture with the new freeze/LR configuration).
    #
    # OFFLINE CONSTRAINT (RULES.md §"Offline Constraint"):
    #   CUDA Graph capture requires a GPU. The code below is syntactically
    #   validated but NOT runtime-tested. On the server, capture failure sets
    #   graph_capture_failed=True and silently falls back to the existing
    #   eager path (training continues uninterrupted — the eager path is
    #   preserved verbatim below the graph-replay branch).

    # Static input buffer — shape fixed at capture time. The data pipeline
    # yields (batch_size, seq_len) torch.long tensors on DEVICE.
    static_batch_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=DEVICE)

    # Static loss + cos tensors — read via .item() AFTER replay (the .item()
    # sync forces CPU-GPU synchronization, which is forbidden inside capture).
    # Two copies for ping-pong (the captured graphs bake in buf_idx-specific
    # event/buffer addresses, so we need a separate graph pair per buf_idx).
    static_loss  = [torch.zeros((), dtype=torch.float32, device=DEVICE) for _ in range(2)]
    static_l_cos = [torch.zeros((), dtype=torch.float32, device=DEVICE) for _ in range(2)]

    # Graph objects (one per buf_idx). Captured lazily after warmup.
    teacher_graph = [None, None]   # Patch 22: captured on stream_t
    student_graph = [None, None]   # Patch 21: captured on default stream

    # Re-capture tracking.
    # last_capture_step[buf_idx] = step at which this buf_idx's graph pair was
    # last captured. -1 = never captured (force immediate capture next eligible
    # step). Reset to -1 after NaN recovery or HP changes to force re-capture.
    last_capture_step = [-1, -1]
    # HP signature (json.dumps, sorted) baked into the captured graph. If the
    # live-JSON hyperparams change, last_hp_sig (in the for-loop below) is
    # updated and differs from this — forcing a re-capture.
    last_capture_hp_sig = [None, None]

    # Tunables
    CUDA_GRAPH_WARMUP_STEPS = 3          # 3-5 eager steps before first capture
    CUDA_GRAPH_RECAPTURE_INTERVAL = 100  # periodic re-capture for LR/tau pickup
    graph_capture_failed = False         # if True, never try graph replay again

    # ─── Graph-friendly helpers ──────────────────────────────────────────
    # The standard compute_loss + clip_grad_norm_ call .item() internally,
    # which forces a CPU-GPU sync and aborts CUDA Graph capture. These
    # helpers duplicate the math without .item() — they return CUDA tensors
    # which the caller writes to static_loss/static_l_cos and reads via
    # .item() AFTER the graph replay (outside the captured region).

    def _graph_safe_compute_loss(s_out, t_out, hp_dict):
        """Graph-friendly duplicate of compute_loss (line 229) — no .item().

        Returns (loss_tensor, l_cos_tensor) as CUDA tensors.
        """
        s = s_out.float()
        t = t_out.detach().float()
        cos_per = F.cosine_similarity(s.flatten(0, 1), t.flatten(0, 1),
                                      dim=-1, eps=1e-4)
        l_cos = (1 - cos_per).mean()
        w = normalize_weights(hp_dict.get("loss_weights",
                                          {"cos": 0.5, "mse": 0.5}))
        loss_type = hp_dict.get("loss_type", "1-cos+norm_mse")
        if loss_type == "1-cos":
            loss = l_cos
        elif loss_type == "1-cos+norm_mse":
            t_var = (t * t).mean().clamp(min=1e-6)
            l_mse = ((s - t) ** 2).mean() / t_var
            loss = w["cos"] * l_cos + w["mse"] * l_mse
        elif loss_type == "norm_mse":
            t_var = (t * t).mean().clamp(min=1e-6)
            loss = ((s - t) ** 2).mean() / t_var
        else:
            loss = l_cos
        return loss, l_cos

    def _graph_safe_clip_grad_norm_(params, max_norm):
        """Graph-friendly clip_grad_norm_ — no .item() sync.

        Equivalent to torch.nn.utils.clip_grad_norm_ but ALWAYS multiplies
        by min(1.0, max_norm / (total_norm + eps)) — no `if clip_coef < 1:`
        branch (which would read clip_coef via .item() and abort capture).

        Returns the total_norm tensor (caller may .item() after replay).
        """
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if not grads:
            return torch.zeros((), dtype=torch.float32, device=DEVICE)
        norms = torch.stack([torch.norm(g, 2) for g in grads])
        total_norm = torch.norm(norms, 2)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
        return total_norm

    def _capture_step_graphs(buf_idx, current_tau, current_hp_sig, current_step):
        """Capture teacher_graph[buf_idx] + student_graph[buf_idx].

        The capture itself EXECUTES the current step (PyTorch CUDA Graphs
        replay during capture), so the captured step is not wasted.

        Pre-conditions (caller MUST ensure):
          - static_batch_ids has been populated with current batch_ids
          - student is in train() mode (eval() mode would freeze dropout
            patterns and break the captured autograd graph)
          - HP / LR / tau have been applied to the model + optimizers
            (the captured graph bakes in the current LR + tau values)

        Captured graphs (Patch 21 + Patch 22):
          teacher_graph[buf_idx] (on stream_t):
            teacher.model.embed_tokens(static_batch_ids) + teacher layers
            → h_out_buf[buf_idx].copy_(h.detach()) → event_t[buf_idx].record()
          student_graph[buf_idx] (on default stream):
            wait_event(event_t[buf_idx])    # Patch 22 sync
            student.model.embed_tokens(static_batch_ids) + student layers
            → _graph_safe_compute_loss(student_out, h_out_buf[buf_idx], hp)
            → static_loss[buf_idx].copy_(loss.detach())
            → static_l_cos[buf_idx].copy_(l_cos.detach())
            → event_s[buf_idx].record()     # Patch 22 sync (invariant 3)
            → loss.backward()
            → _graph_safe_clip_grad_norm_ (per-group, ±1.0 / hp["gradient_clip"])
            → opt_muon/opt_adamw/opt_indices.step()
            → index_logits.clamp_(-5*tau, 5*tau)
            → opt_*.zero_grad(set_to_none=True)

        NOT captured (run eagerly after replay):
          - LR scheduler.step()  (pure Python; doesn't run during replay)
          - NaN check           (forces .item() sync)
          - Eval / save         (rare; would invalidate the captured graph)
          - Logging             (forces .item() syncs on grad_norms etc.)
        """
        # Ensure h_out_buf[buf_idx] is allocated (first capture for this buf).
        # Run a tiny no-grad dummy forward to learn the output shape.
        if h_out_buf[buf_idx] is None:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    _h = teacher.model.embed_tokens(static_batch_ids)
                    _pos_ids = torch.arange(static_batch_ids.shape[1],
                                            device=DEVICE).unsqueeze(0)
                    _pos_emb = None
                    if hasattr(teacher.model, 'rotary_emb') and teacher.model.rotary_emb is not None:
                        _pos_emb = teacher.model.rotary_emb(_h, _pos_ids)
                    for _li in range(sb_end):
                        if _li < len(teacher.model.layers):
                            _layer = teacher.model.layers[_li]
                            if _pos_emb is not None:
                                _out = _layer(_h, position_embeddings=_pos_emb)
                            else:
                                _out = _layer(_h)
                            _h = _out[0] if isinstance(_out, tuple) else _out
            h_out_buf[buf_idx] = torch.empty_like(_h.detach())

        # === Capture teacher_graph[buf_idx] on stream_t (Patch 22) ===
        # Patch 22: explicit stream double-buffer integration with CUDA Graphs.
        #   - teacher_graph runs on stream_t (separate from student's default stream)
        #   - Producer/consumer sync via event_t (teacher→student) + event_s (student→teacher)
        #   - Teacher forward (80ms) is fully hidden behind student compute (980ms)
        #     because the NEXT iter's teacher_graph can replay on stream_t while
        #     the CURRENT iter's student_graph is still doing backward on default.
        teacher_graph[buf_idx] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(teacher_graph[buf_idx], stream=stream_t):
            # Patch 22 invariant 1 (WAIT): teacher must NOT overwrite h_out_buf[buf_idx]
            # while the PREVIOUS iter's student is still reading it. event_s[buf_idx]
            # was recorded by student_graph[buf_idx] after its compute_loss (see
            # invariant 3 below). On first capture for this buf_idx, event_s[buf_idx]
            # is uninitialized → wait_event returns immediately (no-op).
            #
            # This is the captured-graph analogue of the eager Patch 6 line:
            #   if step > 0: stream_t.wait_event(event_s[buf_idx])
            # (train_qwen.py:1667-1668). Baking it into the graph makes the
            # producer/consumer ordering deterministic across replays.
            stream_t.wait_event(event_s[buf_idx])

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    h = teacher.model.embed_tokens(static_batch_ids)
                    position_ids = torch.arange(static_batch_ids.shape[1],
                                                device=DEVICE).unsqueeze(0)
                    if hasattr(teacher.model, 'rotary_emb') and teacher.model.rotary_emb is not None:
                        pos_emb = teacher.model.rotary_emb(h, position_ids)
                    else:
                        pos_emb = None
                    for layer_idx in range(sb_end):
                        if layer_idx < len(teacher.model.layers):
                            layer = teacher.model.layers[layer_idx]
                            if pos_emb is not None:
                                out = layer(h, position_embeddings=pos_emb)
                            else:
                                out = layer(h)
                            h = out[0] if isinstance(out, tuple) else out
                    h_out_buf[buf_idx].copy_(h.detach())
            # Patch 22 invariant 1 (SIGNAL): teacher forward done writing h_out_buf[buf_idx].
            # The student_graph[buf_idx] (captured below) waits on this event before
            # reading h_out_buf[buf_idx] (invariant 2).
            event_t[buf_idx].record(stream_t)

        # === Capture student_graph[buf_idx] on default stream (Patch 21) ===
        student_graph[buf_idx] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(student_graph[buf_idx],
                              stream=torch.cuda.current_stream()):
            # Patch 22 invariant 2: wait for teacher before reading h_out_buf
            torch.cuda.current_stream().wait_event(event_t[buf_idx])

            s_h = student.model.embed_tokens(static_batch_ids)
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                position_ids = torch.arange(static_batch_ids.shape[1],
                                            device=DEVICE).unsqueeze(0)
                if hasattr(student.model, 'rotary_emb') and student.model.rotary_emb is not None:
                    s_pos_emb = student.model.rotary_emb(s_h, position_ids)
                else:
                    s_pos_emb = None
                for layer_idx in range(sb_end + 1):  # +1 for correction layer
                    if layer_idx < len(student.model.layers):
                        layer = student.model.layers[layer_idx]
                        if s_pos_emb is not None:
                            out = layer(s_h, position_embeddings=s_pos_emb)
                        else:
                            out = layer(s_h)
                        s_h = out[0] if isinstance(out, tuple) else out
                student_out = s_h

            # Graph-friendly loss (no .item() inside graph)
            loss, l_cos = _graph_safe_compute_loss(
                student_out, h_out_buf[buf_idx], hp)
            # Save loss tensors to static buffers for post-replay .item() read
            static_loss[buf_idx].copy_(loss.detach())
            static_l_cos[buf_idx].copy_(l_cos.detach())

            # Patch 22 invariant 3: signal student done READING h_out_buf
            # IMMEDIATELY after compute_loss, BEFORE backward(). The student
            # only needs h_out_buf for the loss — recording event_s now lets
            # the NEXT iter's teacher start writing to h_out_buf[1-buf_idx] as
            # soon as compute_loss finishes (overlapping with this iter's bwd).
            event_s[buf_idx].record(torch.cuda.current_stream())

            # Backward (captured — autograd graph is built during capture)
            loss.backward()

            # Two-tier clip (graph-friendly; no .item() inside graph).
            # Mirrors the eager-path split: indices clipped at 1.0, others
            # at hp["gradient_clip"] (default 0.3).
            indices_params_g = [p for _n, p in student.named_parameters()
                                if p.grad is not None and "index_logits" in _n]
            other_params_g   = [p for _n, p in student.named_parameters()
                                if p.grad is not None and "index_logits" not in _n]
            if indices_params_g:
                _graph_safe_clip_grad_norm_(indices_params_g, 1.0)
            if other_params_g:
                _graph_safe_clip_grad_norm_(other_params_g,
                                            hp.get("gradient_clip", 0.3))

            # Optimizer steps. LR is baked in at capture time; the next
            # re-capture picks up LR scheduler updates.
            if opt_muon:    opt_muon.step()
            if opt_adamw:   opt_adamw.step()
            if opt_indices: opt_indices.step()

            # Adaptive clamp on index_logits (±5τ — same as eager path).
            # current_tau is the tau value baked into this capture.
            if opt_indices:
                with torch.no_grad():
                    for _name, par in student.named_parameters():
                        if "index_logits" in _name:
                            par.data.clamp_(-5.0 * current_tau, 5.0 * current_tau)

            # zero_grad INSIDE the graph so grads are clean for next replay.
            if opt_muon:    opt_muon.zero_grad(set_to_none=True)
            if opt_adamw:   opt_adamw.zero_grad(set_to_none=True)
            if opt_indices: opt_indices.zero_grad(set_to_none=True)

        last_capture_step[buf_idx]    = current_step
        last_capture_hp_sig[buf_idx]  = current_hp_sig

    def _replay_step_graphs(buf_idx):
        """Replay teacher_graph[buf_idx] + student_graph[buf_idx] (Patch 22).

        Patch 22 stream double-buffer integration:
          1. teacher_graph[buf_idx].replay() — enqueues the teacher forward
             on stream_t. Inside the captured graph, the FIRST op is
             `stream_t.wait_event(event_s[buf_idx])` (invariant 1 WAIT),
             which blocks until the PREVIOUS iter's student finished reading
             h_out_buf[buf_idx]. On first replay for this buf_idx, the event
             is uninitialized → wait is a no-op. The LAST op is
             `event_t[buf_idx].record(stream_t)` (invariant 1 SIGNAL).
          2. student_graph[buf_idx].replay() — enqueues the student fwd +
             loss + bwd + clip + opt + zero_grad on the DEFAULT stream.
             Inside the captured graph, the FIRST op is
             `current_stream().wait_event(event_t[buf_idx])` (invariant 2),
             which blocks until the teacher finished writing h_out_buf[buf_idx].
             After compute_loss, `event_s[buf_idx].record(current_stream())`
             (invariant 3) signals that the student is done READING h_out_buf —
             the NEXT iter's teacher_graph[buf_idx] can now start writing.

        Stream overlap (the entire point of Patch 22):
          Because teacher_graph runs on stream_t and student_graph runs on
          the default stream, the GPU can execute them concurrently. The
          80ms teacher forward for iter N+1 (enqueued by the NEXT call to
          _replay_step_graphs(1-buf_idx)) runs concurrently with the 980ms
          student backward for iter N — the teacher forward is fully hidden.

        Pre-conditions:
          - static_batch_ids has been populated with current batch_ids
          - Both graphs were previously captured for this buf_idx
        Returns: (loss_val, l_cos_val) as Python floats (post-sync).
        """
        # Patch 22: teacher graph on stream_t — enqueues wait_event(event_s),
        # teacher forward, h_out_buf.copy_(h.detach()), record(event_t).
        # Returns immediately (does not block CPU — runs async on stream_t).
        teacher_graph[buf_idx].replay()
        # Patch 21: student graph on default stream — enqueues wait_event(event_t),
        # student fwd, loss, record(event_s), backward, clip, opt, zero_grad.
        # The wait_event inside the captured graph blocks the default stream
        # until event_t[buf_idx] is signaled by the teacher graph above.
        student_graph[buf_idx].replay()
        # Sync to make static_loss readable via .item() (forces CPU-GPU sync).
        # This is the only sync point in the graph path — everything before
        # this runs async on the GPU.
        torch.cuda.synchronize()
        return static_loss[buf_idx].item(), static_l_cos[buf_idx].item()

    def _cuda_graph_eligible(step_val, hp_val, global_step_val, save_every_val):
        """Decide if the current step is eligible for CUDA Graph replay.

        NOT eligible when:
          - A previous capture/replay failed (graph_capture_failed=True)
          - Still in warmup (step < CUDA_GRAPH_WARMUP_STEPS)
          - Eval step (eval calls student.eval() — invalidates train-mode graph)
          - Save step (save is rare; just run eager to keep code simple)
        """
        if graph_capture_failed:
            return False
        if step_val < CUDA_GRAPH_WARMUP_STEPS:
            return False
        eval_every = hp_val.get("eval_every", 250)
        if global_step_val > 0 and global_step_val % eval_every == 0:
            return False
        if global_step_val > 0 and global_step_val % save_every_val == 0:
            return False
        return True

    def _cuda_graph_needs_recapture(buf_idx, step_val, current_hp_sig):
        """Decide if the captured graph needs to be re-captured for this buf.

        Re-capture when:
          - Never captured (last_capture_step[buf_idx] < 0)
          - Periodic re-capture (step - last_capture_step >= RECAPTURE_INTERVAL)
            picks up LR scheduler changes (LR is baked into the captured kernel)
          - HP signature changed (live JSON update — forces immediate re-capture)
          - Also: previous NaN recovery (last_capture_step reset to -1)
        """
        if last_capture_step[buf_idx] < 0:
            return True
        if (step_val - last_capture_step[buf_idx]) >= CUDA_GRAPH_RECAPTURE_INTERVAL:
            return True
        if last_capture_hp_sig[buf_idx] != current_hp_sig:
            return True
        return False

    # ─── Epilogue / StopIteration handling (Round 1 fix) ────────────────────
    # The for-loop's iteration protocol catches StopIteration automatically
    # when data_stream is exhausted (n_seqs reached at line 918's generator).
    # The `if global_step >= max_steps: break` below handles the case where
    # max_steps < stream length. No explicit `next(data_stream)` call exists
    # in this loop body — Python's `for ... in enumerate(...)` IS the
    # fetch+catch wrapper. If a future refactor introduces an explicit
    # `next(data_stream)` (e.g. for true N+1 prefetch), it MUST be wrapped:
    #     if step + 1 < max_steps:
    #         try:
    #             next_batch = next(data_stream)
    #         except StopIteration:
    #             break
    # to prevent StopIteration from leaking past the for-loop boundary (Python
    # 3.7+ silently propagates StopIteration out of generator-exit contexts).
    #
    # Event-recording order verification (research 06 §3.1 invariants):
    #   * event_s[buf_idx] is recorded IMMEDIATELY after compute_loss (line
    #     ~1169), BEFORE loss.backward(). This is INTENTIONAL — the student
    #     only READS h_out_buf[buf_idx] during compute_loss; backward() does
    #     not touch h_out_buf (it computes grads of student.model.parameters
    #     w.r.t. student_out, which depends on batch_ids, NOT h_out_buf).
    #     Recording early lets the NEXT iter's teacher forward start as soon
    #     as compute_loss completes, overlapping teacher_fwd(N+1) with
    #     student_bwd(N) — the entire point of double-buffering. Recording
    #     event_s AFTER backward would serialize teacher_fwd behind
    #     student_bwd and forfeit the overlap.
    #   * Teacher (next iter) waits on event_s[buf_idx] via
    #     stream_t.wait_event(...) at line ~1107 BEFORE writing
    #     h_out_buf[buf_idx]. Correct producer/consumer ordering.
    #   * Student (current iter) waits on event_t[buf_idx] via
    #     torch.cuda.current_stream().wait_event(...) at line ~1138 BEFORE
    #     reading h_out_buf[buf_idx]. Correct producer/consumer ordering.
    for step, batch_ids in enumerate(data_stream):
        if global_step >= max_steps: break

        # Temperature annealing for Gumbel-Softmax.
        # Schedule (research-indices-training/04_tau_schedule.md §6):
        #   1. Warmup (T_WARMUP=500 steps): hold tau at tau_init (2.0) for high
        #      exploration — gradients are ~1.0× of the τ=2 baseline.
        #   2. Quadratic decay (tau_anneal_steps steps): tau = tau_init *
        #      (1 - progress)^2 with alpha=2. Front-loads the high-τ regime:
        #      ~66% of training is spent at τ≥1.0 where P_loser≥0.09 and the
        #      per-element Gumbel-Softmax gradient is ≥36% of peak.
        #   3. Hold at tau_final (0.5): never below 0.5 — below 0.5 the
        #      gradient magnitude for K=4 falls below 9% of peak (Table 1
        #      in 04_tau_schedule.md) and indices effectively freeze.
        # Net effect: ~62% boost in cumulative gradient signal vs the
        # previous linear 2.0→0.1 schedule, and the indices stay trainable
        # for the entire 8300-step run instead of freezing at step 4000.
        if use_soft_indices:
            # Relative to resume_step — fresh schedule for this run
            rel_step = global_step - resume_step
            T_WARMUP = 500
            T_ANNEAL = tau_anneal_steps  # 6000 by default
            if rel_step < T_WARMUP:
                tau = tau_init  # 2.0 — warmup at high tau
            elif rel_step < T_WARMUP + T_ANNEAL:
                progress = (rel_step - T_WARMUP) / T_ANNEAL
                # alpha=2 quadratic decay; max() floor at tau_final (0.5)
                tau = max(tau_final, tau_init * (1.0 - progress) ** 2)
            else:
                tau = tau_final  # 0.5 — hold (NOT 0.1, which kills gradients)
            # Update tau on all PalettizedLinear modules
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau

        # Check for JSON updates every 10 steps
        if global_step - last_hp_check >= 10:
            new_hp = load_hyperparams()
            ns = json.dumps(new_hp, sort_keys=True)
            if ns != last_hp_sig:
                print(f"  [step {global_step}] Hyperparams updated", flush=True)
                hp = new_hp
                # Re-apply freeze/unfreeze
                apply_groups(student, hp, sb_idx)
                # Update LRs (pass schedulers so base_lrs get updated too)
                update_lrs(opt_muon, opt_adamw, hp, sb_idx,
                           sched_muon=sched_muon, sched_adamw=sched_adamw,
                           opt_indices=opt_indices, sched_indices=sched_indices)
                last_hp_sig = ns
            last_hp_check = global_step

        # ─── [Patch 21 + 22, Wave 4] CUDA Graph step (capture or replay) ───
        # Research: research-kernel-efficiency/08_recommendations.md §11
        #           research-kernel-efficiency/06_stream_overlap.md §6
        #
        # If the current step is eligible for CUDA Graph replay (past warmup,
        # not an eval/save step, no prior capture failure), execute the full
        # training step via the captured graphs and continue. Falls back to
        # the eager path (below) on capture/replay failure or when conditions
        # aren't met.
        #
        # Patch 21: student_graph captures fwd + loss + bwd + clip + opt + zero_grad
        #           on the DEFAULT stream. Eliminates ~500 kernel launch dispatches
        #           × 5µs = 2.5ms of CPU-side overhead per step.
        # Patch 22: teacher_graph captures the teacher forward on stream_t (separate
        #           from student's default stream), with event-based producer/consumer
        #           synchronization between the two graphs (3 invariants from
        #           06_stream_overlap.md §3.1, baked into the captures):
        #             1. WAIT  (teacher, start):  stream_t.wait_event(event_s[buf_idx])
        #                 — wait for previous iter's student to finish reading h_out_buf
        #             2. WAIT  (student, start): current_stream().wait_event(event_t[buf_idx])
        #                 — wait for current iter's teacher to finish writing h_out_buf
        #             3. SIGNAL (student, after compute_loss): event_s[buf_idx].record()
        #                 — let the NEXT iter's teacher start writing to h_out_buf now
        #                   (the student only needed h_out_buf for the loss; backward
        #                   doesn't touch it). This enables true producer/consumer
        #                   overlap: the 80ms teacher forward for iter N+1 runs
        #                   concurrently with the 980ms student backward for iter N.
        #           Teacher forward is fully hidden behind student compute, made
        #           deterministic by graph replay (Patch 6's stream double-buffer
        #           preserved as the eager fallback below).
        #
        # The eager path (below) is the original Patch 6 stream double-buffer +
        # existing training step, preserved verbatim as the fallback.
        save_every = hp.get("save_every", 2000)
        current_hp_sig = json.dumps(hp, sort_keys=True)
        if _cuda_graph_eligible(step, hp, global_step, save_every):
            try:
                # Populate static input buffer (invariant: shapes fixed at
                # capture time — batch_size × seq_len torch.long).
                static_batch_ids.copy_(batch_ids)

                if _cuda_graph_needs_recapture(buf_idx, step, current_hp_sig):
                    # Capture path — executes the step AND captures the graph.
                    # tau + LR + HP are baked into the captured kernels.
                    _capture_step_graphs(buf_idx, tau, current_hp_sig, step)
                    # The capture itself executes the step (PyTorch CUDA Graphs
                    # replay during capture). Sync to make static_loss readable.
                    torch.cuda.synchronize()
                    loss_val = static_loss[buf_idx].item()
                    l_cos_val = static_l_cos[buf_idx].item()
                else:
                    # Replay path — single CPU dispatch replays ~500 captured
                    # kernels back-to-back, no per-kernel launch overhead.
                    loss_val, l_cos_val = _replay_step_graphs(buf_idx)

                # NaN check (post-replay, OUTSIDE the graph — .item() syncs).
                # This is the data-dependent control flow that MUST stay outside
                # the captured region (per 08_recommendations.md §11 req. 3).
                if not math.isfinite(loss_val):
                    n_nan_skip += 1
                    print(f"  [step {global_step}] NaN loss (graph replay) — skipping", flush=True)
                    # Force re-capture on next step (grads were zeroed inside
                    # the captured graph; state is dirty).
                    last_capture_step[buf_idx] = -1
                    global_step += 1  # advance to avoid infinite loop on persistent NaN
                    buf_idx = 1 - buf_idx
                    continue

                # LR scheduler step (eager — NOT captured in graph because
                # sched_*.step() is pure Python that doesn't run during replay).
                # The updated LR takes effect on the NEXT re-capture (when the
                # optimizer kernel is re-baked with the new LR value).
                if sched_muon:    sched_muon.step()
                if sched_adamw:   sched_adamw.step()
                if sched_indices: sched_indices.step()

                global_step += 1

                # Logging (every log_every steps). grad_norms are not available
                # in graph mode (clip happens inside the captured graph) — log
                # "(graph)" instead. Loss/cos come from the static buffers.
                if global_step % hp.get("log_every", 50) == 0:
                    cos_val = 1.0 - l_cos_val
                    elapsed = max(1e-6, time.time() - start_time)
                    tps = (global_step - resume_step) / elapsed
                    # Per-group LRs (unique)
                    group_lrs = {}
                    for opt in (opt_muon, opt_adamw, opt_indices):
                        if opt is None: continue
                        for g in opt.param_groups:
                            grp = g.get("group", "?")
                            if grp not in group_lrs:
                                group_lrs[grp] = g["lr"]
                    lrs_str = " ".join(f"{k}={v:.1e}" for k, v in sorted(group_lrs.items()))
                    tau_str = f"tau={tau:.3f}" if use_soft_indices else ""
                    try:
                        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                        gpu_util = torch.cuda.utilization()
                        gpu_str = f" GPU[{gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f}G util={gpu_util}%]"
                    except Exception:
                        gpu_str = ""
                    print(f"  step={global_step:5d} loss={loss_val:.4f} cos={cos_val:.4f} tps={tps:.1f} {tau_str}{gpu_str} gn=[(graph)] [{lrs_str}]", flush=True)

                # Eval + save (same logic as eager path — evaluate() switches
                # student to eval mode then back to train mode at line ~465).
                # After eval, we force a re-capture on the next eligible step
                # (autograd state may have shifted).
                if global_step % hp.get("eval_every", 250) == 0:
                    eval_result = evaluate(student, teacher, eval_tokens, sb_idx, hp, max_batches=8)
                    eval_cos = eval_result["cos"]
                    print(f"  [EVAL] step={global_step} cos={eval_cos:.6f} loss={eval_result['loss']:.6f} ({eval_result['n_seqs']} held-out seqs)", flush=True)
                    if eval_cos > best_cos and eval_cos > 0:
                        best_cos = eval_cos
                        best_step = global_step
                        best_loss = eval_result['loss']
                        print(f"  ★ NEW BEST (eval): step={best_step} cos={best_cos:.6f}", flush=True)
                        if global_step % save_every == 0 or global_step >= max_steps:
                            out_dir = os.path.join(TRAINED_BASE, f"superblock_{sb_idx}_best")
                            save_state(student, sb_idx, best_step, best_cos, best_loss, out_dir)
                        else:
                            print(f"    (deferred save — next save at step {((global_step // save_every) + 1) * save_every})", flush=True)
                    # Force re-capture on both buffers (eval briefly switched
                    # student to eval mode — even though student.train() was
                    # called inside evaluate(), be safe and re-capture).
                    last_capture_step[0] = -1
                    last_capture_step[1] = -1

                del batch_ids
                buf_idx = 1 - buf_idx  # Patch 6: ping-pong to next buffer
                continue  # Skip the eager path for this step
            except Exception as e:
                print(f"  [step {global_step}] CUDA Graph failed: {e}; falling back to eager", flush=True)
                graph_capture_failed = True
                # Fall through to eager path below

        # === TEACHER FORWARD on stream_t, writes to h_out_buf[buf_idx] ===
        # Patch 6 (Wave 2): persistent stream_t + double-buffered h_out_buf.
        # Invariant 1: teacher never overwrites a buffer the student is reading.
        #   At iter start (step>0), wait for student to release h_out_buf[buf_idx]
        #   (event_s[buf_idx] was recorded 2 iters ago, when the student finished
        #   reading it). On step 0, event_s[buf_idx] is uninitialized → signaled →
        #   wait_event returns immediately (no-op).
        if step > 0:
            stream_t.wait_event(event_s[buf_idx])
        with torch.cuda.stream(stream_t):
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    h = teacher.model.embed_tokens(batch_ids)
                    position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
                    if hasattr(teacher.model, 'rotary_emb') and teacher.model.rotary_emb is not None:
                        pos_emb = teacher.model.rotary_emb(h, position_ids)
                    else:
                        pos_emb = None
                    for layer_idx in range(sb_end):
                        if layer_idx < len(teacher.model.layers):
                            layer = teacher.model.layers[layer_idx]
                            if pos_emb is not None:
                                out = layer(h, position_embeddings=pos_emb)
                            else:
                                out = layer(h)
                            h = out[0] if isinstance(out, tuple) else out
                    # Reuse pre-allocated buffer when shape matches (avoid per-step
                    # alloc); otherwise (first iter or seq_len change) allocate.
                    h_detached = h.detach()
                    if h_out_buf[buf_idx] is None or h_out_buf[buf_idx].shape != h_detached.shape:
                        h_out_buf[buf_idx] = h_detached
                    else:
                        h_out_buf[buf_idx].copy_(h_detached)
        # Signal: teacher forward done writing h_out_buf[buf_idx]
        event_t[buf_idx].record(stream_t)

        # === STUDENT FORWARD + BACKWARD on default stream ===
        # Invariant 2: student never reads a buffer the teacher is writing.
        #   Wait for teacher to finish writing h_out_buf[buf_idx].
        torch.cuda.current_stream().wait_event(event_t[buf_idx])

        # bf16 has same exponent range as fp32 — no overflow in GatedDeltaNet
        s_h = student.model.embed_tokens(batch_ids)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
            if hasattr(student.model, 'rotary_emb') and student.model.rotary_emb is not None:
                s_pos_emb = student.model.rotary_emb(s_h, position_ids)
            else:
                s_pos_emb = None
            for layer_idx in range(sb_end + 1):  # +1 for correction layer
                if layer_idx < len(student.model.layers):
                    layer = student.model.layers[layer_idx]
                    if s_pos_emb is not None:
                        out = layer(s_h, position_embeddings=s_pos_emb)
                    else:
                        out = layer(s_h)
                    s_h = out[0] if isinstance(out, tuple) else out
            student_out = s_h

        # Loss — outside autocast, fp32 math
        # Reverted to Dolphin's combined loss (1-cos+norm_mse) throughout
        loss, comps = compute_loss(student_out, h_out_buf[buf_idx], hp)
        # Alias for downstream references (NaN message + del + post-step logging)
        h_out = h_out_buf[buf_idx]

        # Invariant 3 (research 06 §3.1): record event_s IMMEDIATELY after
        # compute_loss, BEFORE backward(). The student only needs h_out_buf for
        # the loss — by signaling now, the NEXT iter's teacher can start writing
        # to this buffer as soon as compute_loss finishes (much earlier than
        # when loss.backward() completes), enabling true producer/consumer overlap.
        event_s[buf_idx].record(torch.cuda.current_stream())

        if not torch.isfinite(loss):
            # Quick param-NaN diagnostic
            param_nan = {}
            for name, par in student.named_parameters():
                if not par.requires_grad: continue
                if torch.isnan(par).any():
                    g = "indices" if "index_logits" in name else ("palettes" if "palette" in name else ("lora" if ("lora_A" in name or "lora_B" in name) else "layernorms"))
                    param_nan[g] = param_nan.get(g, 0) + 1
            print(f"  [step {global_step}] NaN loss — skipping (param_nan={param_nan}, student_out NaN={torch.isnan(student_out).any().item()})", flush=True)
            n_nan_skip += 1
            if opt_muon: opt_muon.zero_grad(set_to_none=True)
            if opt_adamw: opt_adamw.zero_grad(set_to_none=True)
            if opt_indices: opt_indices.zero_grad(set_to_none=True)
            del batch_ids, h_out, student_out, loss, comps
            global_step += 1  # advance to avoid infinite loop on persistent NaN
            buf_idx = 1 - buf_idx  # Patch 6: ping-pong even on NaN skip (keep pipeline primed)
            continue

        # Backward
        loss.backward()

        # Two-tier clip: indices separate (their tiny Gumbel grads would be zeroed by global norm)
        clip_val = hp.get("gradient_clip", 0.3)
        indices_params = []
        other_params = []
        for name, par in student.named_parameters():
            if par.grad is None: continue
            if "index_logits" in name:
                indices_params.append(par)
            else:
                other_params.append(par)
        grad_norms = {}
        if indices_params:
            gn = torch.nn.utils.clip_grad_norm_(indices_params, 1.0)
            grad_norms["indices"] = gn.item()
        if other_params:
            gn_other = torch.nn.utils.clip_grad_norm_(other_params, clip_val)
            grad_norms["palettes+lora+norms"] = gn_other.item()
        if opt_muon: opt_muon.step()
        if opt_adamw: opt_adamw.step()
        if opt_indices: opt_indices.step()
        # CRITICAL: clamp index_logits after step to prevent both (a) fp16
        # overflow on the next forward and (b) softmax saturation that zeroes
        # the Gumbel-Softmax gradient. The clamp is ADAPTIVE to temperature:
        #   par.data.clamp_(-5.0 * tau, 5.0 * tau)
        # The previous fixed ±20 was a band-aid for (a) only — at the old
        # tau=0.1 it caused (b): softmax(±20/0.1) = softmax(±200) overflows
        # to [1, 0] in fp32 and the gradient is exactly zero (the documented
        # cause of index freeze in research-filter-consolidation/
        # 01_training_recipe.md §3).
        # The new ±5τ clamp fixes both:
        #   - At tau=2.0 (warmup):     clamp ±10   — loose, exploration
        #   - At tau=0.5 (hold/floor): clamp ±2.5  — tight, commitment
        #   - softmax(±5) ≈ [0.993, 0.007] — still essentially one-hot for
        #     the forward pass, but with finite-precision gradient
        #     (BNN-style tight clip from Courbariaux et al. 2016, applied
        #     here in logit-space instead of weight-space).
        # tau is in scope here because opt_indices is non-None only when
        # use_soft_indices=True, which is the same condition that sets tau
        # at the top of this iteration (see build_optimizers line 597 +
        # the tau anneal block ~30 lines above).
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        # Adaptive clamp: ±5τ (was ±20). See comment above.
                        par.data.clamp_(-5.0 * tau, 5.0 * tau)
        if sched_muon: sched_muon.step()
        if sched_adamw: sched_adamw.step()
        if sched_indices: sched_indices.step()
        if opt_muon: opt_muon.zero_grad(set_to_none=True)
        if opt_adamw: opt_adamw.zero_grad(set_to_none=True)
        if opt_indices: opt_indices.zero_grad(set_to_none=True)
        global_step += 1

        # Logging — rich dynamic stats: GPU mem/util, tps, tau, per-group LRs, grad norms
        if global_step % hp.get("log_every", 50) == 0:
            cos_val = 1.0 - comps["cos"]
            elapsed = max(1e-6, time.time() - start_time)
            tps = (global_step - resume_step) / elapsed
            # Per-group LRs (unique)
            group_lrs = {}
            for opt in (opt_muon, opt_adamw, opt_indices):
                if opt is None: continue
                for g in opt.param_groups:
                    grp = g.get("group", "?")
                    if grp not in group_lrs:
                        group_lrs[grp] = g["lr"]
            lrs_str = " ".join(f"{k}={v:.1e}" for k, v in sorted(group_lrs.items()))
            # Current tau (Gumbel-Softmax temperature)
            tau_str = f"tau={tau:.3f}" if use_soft_indices else ""
            # GPU memory + utilization (pynvml via torch.cuda)
            try:
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_util = torch.cuda.utilization()
                gpu_str = f" GPU[{gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f}G util={gpu_util}%]"
            except Exception:
                gpu_str = ""
            # Grad norms per group (from clip_grad_norm_ above)
            gn_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(grad_norms.items()))
            print(f"  step={global_step:5d} loss={comps['loss']:.4f} cos={cos_val:.4f} tps={tps:.1f} {tau_str}{gpu_str} gn=[{gn_str}] [{lrs_str}]", flush=True)

        # Eval + save (track BEST COS on held-out eval set)
        # Only save every save_every steps AND only if eval beats best
        save_every = hp.get("save_every", 2000)
        if global_step % hp.get("eval_every", 250) == 0:
            eval_result = evaluate(student, teacher, eval_tokens, sb_idx, hp, max_batches=8)
            eval_cos = eval_result["cos"]
            print(f"  [EVAL] step={global_step} cos={eval_cos:.6f} loss={eval_result['loss']:.6f} ({eval_result['n_seqs']} held-out seqs)", flush=True)
            if eval_cos > best_cos and eval_cos > 0:
                best_cos = eval_cos
                best_step = global_step
                best_loss = eval_result["loss"]
                print(f"  ★ NEW BEST (eval): step={best_step} cos={best_cos:.6f}", flush=True)
                # Only save to disk every save_every steps (or at final step)
                if global_step % save_every == 0 or global_step >= max_steps:
                    out_dir = os.path.join(TRAINED_BASE, f"superblock_{sb_idx}_best")
                    save_state(student, sb_idx, best_step, best_cos, best_loss, out_dir)
                else:
                    print(f"    (deferred save — next save at step {((global_step // save_every) + 1) * save_every})", flush=True)

        del batch_ids, h_out, student_out, loss, comps
        buf_idx = 1 - buf_idx  # Patch 6: ping-pong to next buffer for next iter

    # Final save — only if training produced a valid cos (best_cos > 0).
    # If everything NaN'd, skip to preserve the previous good checkpoint on disk.
    if best_cos > 0:
        out_dir = os.path.join(TRAINED_BASE, f"superblock_{sb_idx}_final")
        save_state(student, sb_idx, global_step, best_cos, best_loss, out_dir)
    else:
        print(f"  Skipping final save — best_cos={best_cos} (training diverged; keeping previous checkpoint)", flush=True)
    print(f"\n=== Training complete: best cos={best_cos:.6f} @ step {best_step} ===", flush=True)

    # Shutdown hook
    if shutdown_on_done:
        print("\n=== Shutting down server in 10 seconds... ===", flush=True)
        time.sleep(10)
        os.system("shutdown -h now")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_idx", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=1024,
                    help="Tokens per sequence. Default 1024 for Blackwell 96GB VRAM.")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Sequences per step. Default 32 for Blackwell 96GB VRAM.")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Directory to resume from (e.g. trained/superblock_0_best)")
    ap.add_argument("--use_soft_indices", type=int, default=1,
                    help="Enable trainable indices via Gumbel-Softmax (1=on, 0=off).")
    ap.add_argument("--tau_init", type=float, default=2.0,
                    help="Initial Gumbel-Softmax temperature. Default 2.0 (high tau = soft = gradients flow).")
    ap.add_argument("--tau_final", type=float, default=0.5,
                    help="Final Gumbel-Softmax temperature. Default 0.5 (FLOOR: below 0.5, Gumbel-Softmax gradients vanish for K=4 — see research-indices-training/04_tau_schedule.md).")
    ap.add_argument("--tau_anneal_steps", type=int, default=6000,
                    help="Steps over which to anneal temperature from tau_init to tau_final (warmup is 500 steps, then quadratic decay over this many steps, then hold at tau_final).")
    ap.add_argument("--shutdown_on_done", type=int, default=0,
                    help="Shutdown server when training completes (1=yes, 0=no).")
    args = ap.parse_args()
    os.makedirs(TRAINED_BASE, exist_ok=True)
    train_super_block(args.sb_idx, args.max_steps, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                      seq_len=args.seq_len, batch_size=args.batch_size,
                      resume_from=args.resume_from,
                      use_soft_indices=bool(args.use_soft_indices),
                      tau_init=args.tau_init, tau_final=args.tau_final,
                      tau_anneal_steps=args.tau_anneal_steps,
                      shutdown_on_done=bool(args.shutdown_on_done))


if __name__ == "__main__":
    import atexit
    atexit.register(lambda: os._exit(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
