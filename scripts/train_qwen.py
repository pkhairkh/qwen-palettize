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
        new_param_groups = []
        for group in param_groups:
            new_group = dict(group)
            new_params = []
            for p in group["params"]:
                master = p.data.float().clone()
                master.requires_grad_(True)
                self.model_param_map[id(master)] = p
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
        for group in self.opt.param_groups:
            for master in group["params"]:
                p = self.model_param_map[id(master)]
                if p.grad is not None:
                    master.grad = p.grad.float()
                else:
                    master.grad = None
        self.opt.step(closure=closure)
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
    # AdamW with fp32 master for index_logits (Blackwell 96GB can afford 21.4 GB).
    # CRITICAL: fp16 AdamW state + eps=1e-8 → NaN (sqrt(v)+eps underflows to 0 in fp16).
    # FP32 master avoids this. Plain SGD (L4 fallback) was too weak for Gumbel-Softmax grads.
    opt_indices = FP32MasterAdamW(plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) if plain_adamw_groups else None

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
                full_name = f"model.layers.{layer_idx}.{name}.weight"
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
                lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=None)
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora_mod)
                total_lora += 1
    print(f"  Attached LoRA (rank-{lora_rank} + rank-{BIG_LORA_RANK} on 5 worst-cos) to {total_lora} Linears across {sb_end - sb_start} layers", flush=True)

    # Cast to dtype
    model.to(DTYPE)
    # Restore index_logits to fp16 (model.to(bf16) converts them to bf16,
    # but the soft CUDA kernel requires fp16 logits)
    # PartialWrapper is not nn.Module — iterate its layers manually
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
                      tau_init=1.0, tau_final=0.01, tau_anneal_steps=4000,
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
    sched_indices = torch.optim.lr_scheduler.LambdaLR(opt_indices.opt, lr_lambda) if opt_indices else None

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

    for batch_ids in data_stream:
        if global_step >= max_steps: break

        # Temperature annealing for Gumbel-Softmax
        if use_soft_indices:
            tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
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

        # === TEACHER FORWARD on stream_t (overlaps with student backward) ===
        # Producer/consumer: teacher prepares next batch while student trains on current
        stream_t = torch.cuda.Stream()
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
                    h_out = h.detach()
        # Student forward+backward runs on default stream (stream_s)
        # stream_t will be synced when h_out is used in loss computation

        # === STUDENT FORWARD ===
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
        loss, comps = compute_loss(student_out, h_out, hp)

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
        # CRITICAL: clamp index_logits to safe fp16 range after step.
        # Gumbel-Softmax grad at low tau can push fp32 master to ±1e6,
        # which overflows fp16 (max 65504) → inf → NaN on next forward.
        # Clamp to ±20 (softmax(20/0.1) is already numerically one-hot).
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        par.data.clamp_(-20.0, 20.0)
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
