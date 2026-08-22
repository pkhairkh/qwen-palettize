#!/usr/bin/env python3
"""qwen_model.py — Standalone Qwen3.5-4B model utilities for palettization.

NO Dolphin imports. NO FP32 weights in student.

Provides:
  - PalettizedLinear: 2-bit LUT-based Linear replacement
  - QwenLoRA: rank-32 fp16 LoRA adapter (with optional PalettizedLinear base)
  - CorrectionLayer: dense GatedDeltaNet copy (warm-started, zero-init outputs)
  - load_qwen_model: load Qwen3.5-4B from HuggingFace
  - isolate_super_block: freeze everything except active super-block + correction layers
  - insert_correction_layers: add 1 dense GatedDeltaNet copy per super-block + LoRA
  - attach_lora_to_linears: wrap every palettized Linear with QwenLoRA

Memory strategy:
  - Teacher: fp16, full model (shared embeddings with student)
  - Student: fp16, super-block only + shared embeddings
  - NO fp32 master weights (per user instruction)
  - Progressive forward/backward: only run the active super-block + correction layers
"""
import os, sys, json, math, copy, gc, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import (
    load_indices, load_lut, sanitize_name,
    BITWIDTH, GROUP_SIZE, PALETTE_SIZE,
)


# ─── Super-block definition ────────────────────────────────────────────
NUM_LAYERS = 32
SUPER_BLOCKS = [(i*4, i*4+4) for i in range(8)]  # [(0,4), (4,8), ..., (28,32)]

def get_super_block_layers(sb_idx):
    """Return list of layer indices for super-block sb_idx."""
    start, end = SUPER_BLOCKS[sb_idx]
    return list(range(start, end))

def is_full_attn_layer(layer_idx):
    """Layer 3, 7, 11, 15, 19, 23, 27, 31 are full-attn."""
    return (layer_idx + 1) % 4 == 0

def is_gated_delta_layer(layer_idx):
    return not is_full_attn_layer(layer_idx)


# ─── PalettizedLinear (2-bit LUT, NO FP32) ──────────────────────────────
#
# Performance notes:
#   - Microbench (L4, in_dim=2560, out_dim=8192, batch=8, seq=128):
#       eager gather+matmul (native autograd)   :  37 ms/iter (fwd+bwd)
#       custom autograd Function (fused)         : 976 ms/iter  ← 27x slower!
#   - Native autograd already optimizes the gather: forward computes W once,
#     backward uses scatter_add on the cached indices (no re-gather).
#   - The main cost is the 42MB gather itself (memory-bound at ~35ms),
#     not Python dispatch. Caching W across forward calls doesn't help because
#     palette is trainable so the cache must be invalidated every step.
#   - Bottom line: keep native autograd. Speedups come from seq_len/batch_size
#     and from reducing Python overhead elsewhere.

class PalettizedLinear(nn.Module):
    """2-bit palettized Linear replacement with trainable indices + dual-mode forward.

    Training mode (soft): uses Gumbel-Softmax relaxation via the soft CUDA kernel.
    Eval mode (hard): uses frozen int8 indices via the hard CUDA kernel.

    Forward: y = x @ W_reconstructed + bias
    where W_reconstructed[j, o] = palette[ o // group_size, indices[j, o] ]  (hard)
    or      W_reconstructed[j, o] = Σ_k P[j, o, k] * palette[g, k]            (soft)
    """
    def __init__(self, original_linear, indices, n_groups, palette_size, group_size,
                 pre_transposed, initial_palette=None, use_soft_indices=False):
        super().__init__()
        # Indices stored as int8 for CUDA kernel + int64 for fallback
        self.register_buffer("indices", indices.long())  # (in_dim, out_dim) int64 — for fallback
        self.register_buffer("indices_int8", indices.to(torch.int8).contiguous())  # for hard CUDA kernel
        # Palette is trainable (bf16)
        self.palette = nn.Parameter(
            initial_palette.clone().to(torch.bfloat16) if initial_palette is not None
            else torch.zeros(n_groups, palette_size, dtype=torch.bfloat16)
        )
        self.group_size = group_size
        self.n_groups = n_groups
        self.palette_size = palette_size
        self.pre_transposed = pre_transposed
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        # Bias
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone().to(torch.bfloat16))
        else:
            self.register_buffer("bias", None)
        # Pre-build flat_idx_cache for the fallback path
        si, so = indices.shape
        device = indices.device
        group_idx = torch.arange(so, device=device) // group_size
        group_idx_2d = group_idx.unsqueeze(0).expand(si, so)
        self.register_buffer("_flat_idx", (group_idx_2d * palette_size + indices).contiguous())

        # Load CUDA kernels
        self._hard_kernel = None
        self._soft_kernel = None
        self._use_cuda = False
        if not pre_transposed:
            try:
                import fused_lut_linear_cuda
                self._hard_kernel = fused_lut_linear_cuda.fused_lut_linear
                self._soft_kernel = fused_lut_linear_cuda.fused_lut_linear_soft
                self._use_cuda = True
            except Exception:
                pass

        # Trainable indices via Gumbel-Softmax
        self.use_soft_indices = use_soft_indices
        self.tau = 1.0  # temperature, annealed by training loop
        if use_soft_indices and not pre_transposed:
            # Initialize logits as one-hot from current indices
            # logits shape: (4, K, N) — 4 planes for coalesced access
            K_dim, N_dim = indices.shape
            logits = torch.full((4, K_dim, N_dim), -10.0, dtype=torch.float16, device=device)
            for k in range(4):
                mask = (indices == k)
                logits[k][mask] = 10.0
            self.index_logits = nn.Parameter(logits)
        else:
            self.index_logits = None

    def forward(self, x):
        orig_ndim = x.ndim
        if x.ndim == 3:
            B, S, _ = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        if self._use_cuda and x_flat.is_cuda and not self.pre_transposed:
            if self.training and self.use_soft_indices and self.index_logits is not None:
                # Soft forward: Gumbel-Softmax relaxation
                # W = Σ_k P[k] * palette[k], gradients flow to index_logits
                y = self._soft_kernel(
                    x_flat, self.palette, self.index_logits,
                    self.bias, self.group_size, self.tau
                )
            else:
                # Hard forward: frozen indices + gather
                y = self._hard_kernel(
                    x_flat, self.palette, self.indices_int8,
                    self.bias, self.group_size
                )
        else:
            # Fallback: PyTorch gather + matmul
            flat_palette = self.palette.reshape(-1).to(x.dtype)
            gathered = flat_palette[self._flat_idx]
            if not self.pre_transposed:
                y = x_flat @ gathered
            else:
                y = x_flat @ gathered.T
            if self.bias is not None:
                y = y + self.bias.to(x.dtype)

        if orig_ndim == 3:
            y = y.reshape(B, S, -1)
        return y

    def extract_hard_indices(self):
        """After training, extract hard indices from index_logits (argmax).
        Call this when temperature τ→0 to get final frozen indices."""
        if self.index_logits is not None:
            with torch.no_grad():
                # index_logits: (4, K, N) → argmax over dim 0
                self.indices = self.index_logits.argmax(dim=0).long()
                self.indices_int8 = self.indices.to(torch.int8).contiguous()
                # Drop logits (free memory)
                del self.index_logits
                self.index_logits = None
                self.use_soft_indices = False


# ─── QwenLoRA (rank-32, fp16 ONLY) ─────────────────────────────────────
class QwenLoRA(nn.Module):
    """LoRA adapter. ALL fp16 — no fp32, no bf16.

    Two modes:
      1. Wraps a PalettizedLinear (base = PalettizedLinear)
      2. Wraps a regular nn.Linear (base = nn.Linear)

    y = base(x) + (alpha/r) * x @ A @ B^T
    A: (in_dim, r), B: (out_dim, r)
    """
    def __init__(self, base_module, rank=32, alpha=64, init="loftq", original_weight=None):
        super().__init__()
        self.base = base_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Get in/out dims from base
        if hasattr(base_module, 'in_features'):
            in_dim = base_module.in_features
            out_dim = base_module.out_features
        else:
            # Fallback for nn.Linear
            out_dim, in_dim = base_module.weight.shape

        if init == "loftq" and original_weight is not None:
            with torch.no_grad():
                # Get the base weight (palettized or dense)
                if hasattr(base_module, 'palette'):
                    # PalettizedLinear — reconstruct weight
                    W_pal = self._reconstruct_base_weight(base_module).float()
                else:
                    W_pal = base_module.weight.data.float()
                R = original_weight.float().to(W_pal.device) - W_pal
                U, S, Vh = torch.linalg.svd(R.float(), full_matrices=False)
                B_init = (U[:, :rank] * S[:rank].sqrt().unsqueeze(0)).contiguous()
                A_init = (Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0)).contiguous()
                A_init = A_init / math.sqrt(self.scaling)
                B_init = B_init / math.sqrt(self.scaling)
        else:
            A_init = torch.randn(in_dim, rank) * 0.01
            B_init = torch.zeros(out_dim, rank)

        # fp16 ONLY — per user instruction. Move to CUDA explicitly.
        # Handle both nn.Linear (has .weight) and PalettizedLinear (has .palette, .indices)
        if hasattr(base_module, 'weight'):
            device = base_module.weight.device
        elif hasattr(base_module, 'palette'):
            device = base_module.palette.device
        else:
            # Fallback: scan parameters
            device = next(base_module.parameters()).device
        self.lora_A = nn.Parameter(A_init.to(torch.bfloat16).to(device))
        self.lora_B = nn.Parameter(B_init.to(torch.bfloat16).to(device))

    def _reconstruct_base_weight(self, pal_module):
        """Reconstruct the palettized weight for LoftQ init."""
        si, so = pal_module.indices.shape
        gs = pal_module.group_size
        group_idx = torch.arange(so, device=pal_module.indices.device) // gs
        group_idx_2d = group_idx.unsqueeze(0).expand(si, so)
        flat_idx = group_idx_2d * pal_module.palette_size + pal_module.indices
        gathered = pal_module.palette.reshape(-1)[flat_idx]
        if not pal_module.pre_transposed:
            return gathered  # (in_dim, out_dim)
        else:
            return gathered.T  # (out_dim, in_dim)

    def forward(self, x):
        y_base = self.base(x)
        orig_ndim = x.ndim
        if x.ndim == 3:
            B_, S_, _ = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x
        # fp16 matmul
        lora_out = (x_flat @ self.lora_A) @ self.lora_B.T
        lora_out = lora_out * self.scaling
        if orig_ndim == 3:
            lora_out = lora_out.reshape(B_, S_, -1)
        return y_base + lora_out


# ─── Palettized LoRA (stage 2: A and B are 2-bit palettized) ──────────
def palettize_lora_weights(lora_module, tensor_name_prefix, palettized_dir, device="cuda"):
    """Replace a QwenLoRA's lora_A and lora_B with PalettizedLinear versions.

    The LoRA forward changes from:
      lora_out = (x @ lora_A) @ lora_B.T * scaling
    to:
      lora_out = lora_B_pal(lora_A_pal(x)) * scaling
    where each _pal is a PalettizedLinear (2-bit, palettes TRAINABLE).

    Storage orientation (from calib_stage2.py → palettize_tensor_2bit):
      lora_A original shape (in_dim, rank) → palettized as W=(in_dim, rank)
        → stored indices shape (rank, in_dim) [transposed by load_indices]
        → PalettizedLinear needs indices (in_features, out_features) = (in_dim, rank)
        → but stored is (rank, in_dim) → pre_transposed=True (does x @ gathered.T)

      lora_B original shape (out_dim, rank) → palettized as W=(out_dim, rank)
        → stored indices shape (rank, out_dim) [transposed by load_indices]
        → PalettizedLinear needs indices (in_features, out_features) = (rank, out_dim)
        → but stored is (rank, out_dim) → wait, that's already (in_features, out_features)!
        → Actually: in_features=rank, out_features=out_dim, stored (rank, out_dim) = (in, out) ✓
        → pre_transposed=False

    Returns:
        (n_palettized, lora_A_pal, lora_B_pal) or (0, None, None) if files not found
    """
    rank = lora_module.lora_A.shape[1]
    in_dim = lora_module.lora_A.shape[0]
    out_dim = lora_module.lora_B.shape[0]

    a_name = f"{tensor_name_prefix}.lora_A"
    b_name = f"{tensor_name_prefix}.lora_B"

    a_pal = _load_palettized_lora_weight(a_name, palettized_dir, in_dim, rank, device)
    b_pal = _load_palettized_lora_weight(b_name, palettized_dir, rank, out_dim, device)

    if a_pal is None or b_pal is None:
        return 0, None, None

    # Replace the nn.Parameters with PalettizedLinear modules
    lora_module.lora_A_pal = a_pal
    lora_module.lora_B_pal = b_pal
    # Mark as palettized so forward uses the new path
    lora_module._palettized = True
    # Delete old parameters (frees memory + prevents them from being saved)
    del lora_module.lora_A
    del lora_module.lora_B
    # Override the forward method
    lora_module.forward = _palettized_lora_forward.__get__(lora_module, type(lora_module))

    return 2, a_pal, b_pal


def _load_palettized_lora_weight(tensor_name, palettized_dir, in_features, out_features, device="cuda"):
    """Load a 2-bit palettized weight and wrap as PalettizedLinear.

    Handles the transposed storage orientation from palettize_tensor_2bit:
      palettize_tensor_2bit stores W as (out_dim, in_dim) with groups along axis 0.
      load_indices reshapes to (in_dim, out_dim) — transposed relative to W.
      PalettizedLinear expects indices of shape (in_features, out_features).

    For lora_A (in_dim, rank): in_features=in_dim, out_features=rank
      → stored indices shape (rank, in_dim) ≠ (in_dim, rank) → pre_transposed=True

    For lora_B (out_dim, rank): in_features=rank, out_features=out_dim
      → stored indices shape (rank, out_dim) = (in_features, out_features) → pre_transposed=False
    """
    san = sanitize_name(tensor_name)
    idx_path = os.path.join(palettized_dir, f"{san}.idx2")
    lut_path = os.path.join(palettized_dir, f"{san}.lut_scalar")

    if not os.path.exists(idx_path) or not os.path.exists(lut_path):
        return None

    meta_path = os.path.join(palettized_dir, "metadata.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        tensor_meta = meta.get("tensors", {}).get(tensor_name)
        if tensor_meta is None:
            return None
        stored_out, stored_in = tensor_meta["dense_shape"]
        n_groups = tensor_meta["groups"]
    else:
        return None

    indices = load_indices(idx_path, stored_out, stored_in).to(device)
    lut = load_lut(lut_path).reshape(n_groups, PALETTE_SIZE).to(device)

    # indices shape from load_indices: (stored_in, stored_out)
    # PalettizedLinear expects: (in_features, out_features)
    # pre_transposed = (indices shape doesn't match (in_features, out_features))
    indices_shape = indices.shape  # (stored_in, stored_out)
    expected_shape = (in_features, out_features)
    if indices_shape == expected_shape:
        pre_transposed = False
    elif indices_shape == (out_features, in_features):
        pre_transposed = True
    else:
        print(f"  WARNING: {tensor_name} indices shape {indices_shape} doesn't match either "
              f"({in_features}, {out_features}) or ({out_features}, {in_features})", flush=True)
        return None

    # Create a fake nn.Linear shell (only used for in_features/out_features metadata)
    fake_linear = nn.Linear(in_features, out_features, bias=False)
    pal_mod = PalettizedLinear(
        fake_linear, indices, n_groups, PALETTE_SIZE, GROUP_SIZE,
        pre_transposed, initial_palette=lut
    )
    return pal_mod


def _palettized_lora_forward(self, x):
    """Forward for QwenLoRA with palettized A and B."""
    y_base = self.base(x)
    orig_ndim = x.ndim
    if x.ndim == 3:
        B_, S_, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
    else:
        x_flat = x
    # Palettized LoRA: lora_A_pal(x) → (batch, rank), then lora_B_pal → (batch, out_dim)
    intermediate = self.lora_A_pal(x_flat)  # (batch, rank)
    lora_out = self.lora_B_pal(intermediate)  # (batch, out_dim)
    lora_out = lora_out * self.scaling
    if orig_ndim == 3:
        lora_out = lora_out.reshape(B_, S_, -1)
    return y_base + lora_out


# ─── Load Qwen model ────────────────────────────────────────────────────
def load_qwen_model(model_name="Qwen/Qwen3.5-4B", device="cuda", dtype=torch.bfloat16,
                    drop_embed=False):
    """Load Qwen3.5-4B. Returns (model, tokenizer).

    Args:
        model_name: HF model name
        device: "cuda" or "cpu"
        dtype: torch.bfloat16 (NO fp32 — per user instruction)
        drop_embed: if True, drop embed_tokens + lm_head after loading

    Loads full model — use load_qwen_super_block_only() for calibration
    (only loads the super-block's layers + embed_tokens, skipping the rest).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa"
    )

    if drop_embed:
        print(f"  Dropping embed_tokens + lm_head (save ~1.27 GB)", flush=True)
        if hasattr(model, 'embed_tokens'):
            del model.model.embed_tokens
        if hasattr(model, 'lm_head'):
            del model.lm_head
        model.model.embed_tokens = None
        torch.cuda.empty_cache()

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Params: {n_params:,}", flush=True)
    print(f"  Layers: {len(model.model.layers)}", flush=True)
    return model, tokenizer


# ─── PartialModel / PartialWrapper (nn.Module subclasses) ────────────────
# These were previously nested inside load_qwen_super_block_only as plain
# Python classes with hand-rolled to/eval/train/parameters/named_parameters/
# named_modules/get_submodule. Converting them to nn.Module unlocks
# torch.compile, torch.utils.checkpoint, state_dict/load_state_dict, FSDP,
# accelerate, transformers.Trainer, register_forward_hook, requires_grad_,
# apply, and ~50 other nn.Module API methods. See research-architecture-
# review/02_partial_wrapper_problem.md for the full diagnosis.
class PartialModel(nn.Module):
    """Prefix of Qwen3.5: embed_tokens + rotary_emb + first N layers + optional norm.

    All attributes are registered as submodules / parameters / buffers via
    nn.Module.__setattr__, so parameters(), named_modules(), state_dict(),
    to(), eval(), train(), apply(), etc. are inherited and correct.
    """
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        super().__init__()
        self.embed_tokens = embed_tokens
        # rotary_emb is an nn.Module in HF Qwen (Qwen3_5RotaryEmbedding);
        # may be None for architectures that fold RoPE into the layer.
        self.rotary_emb = rotary_emb
        # nn.ModuleList (was: plain Python list) so layers are registered in
        # _modules and participate in state_dict / to / eval / train / apply.
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        # config is plain metadata (not a tensor / module); keep as attribute.
        self.config = config

    def forward(self, input_ids, position_ids=None):
        """Run embed_tokens → rotary_emb → prefix layers → optional norm.

        Mirrors the forward path previously inlined in train_qwen.py's
        training loop (teacher.model.embed_tokens → teacher.model.layers[i]
        → ...). The loop there still calls layers directly for streaming /
        CUDA-stream overlap reasons; this forward() is provided so that
        standard PyTorch subsystems (torch.compile, checkpoint, Trainer,
        hooks) work out of the box.
        """
        h = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
        pos_emb = (
            self.rotary_emb(h, position_ids)
            if self.rotary_emb is not None
            else None
        )
        for layer in self.layers:
            out = (
                layer(h, position_embeddings=pos_emb)
                if pos_emb is not None
                else layer(h)
            )
            h = out[0] if isinstance(out, tuple) else out
        if self.norm is not None:
            h = self.norm(h)
        return h


class PartialWrapper(nn.Module):
    """Thin nn.Module wrapper around a PartialModel.

    Exists so that downstream code can treat the student / teacher prefix
    as a single nn.Module (for torch.compile, FSDP, accelerate, Trainer,
    state_dict, etc.) while still allowing the training loop to reach into
    `.model.embed_tokens` / `.model.layers[i]` for streaming / CUDA-stream
    overlap. All nn.Module API methods (parameters, named_parameters,
    named_modules, state_dict, load_state_dict, to, eval, train, apply,
    requires_grad_, register_forward_hook, ...) are inherited.
    """
    def __init__(self, partial_model, config):
        super().__init__()
        # partial_model is a PartialModel (nn.Module) — registered as a
        # submodule named "model", so wrapper.named_parameters() yields
        # keys like "model.layers.0.linear_attn.out_proj.lora_A".
        self.model = partial_model
        # config is plain metadata (HF config object); kept as attribute.
        self.config = config

    def forward(self, input_ids, position_ids=None):
        """Delegate to the wrapped PartialModel."""
        return self.model(input_ids, position_ids)


# NOTE on key naming: after the nn.Module refactor, wrapper.named_parameters()
# and wrapper.named_modules() yield keys prefixed with "model." (e.g.
# "model.layers.0.linear_attn.out_proj.lora_A"). The pre-refactor hand-rolled
# generators yielded keys WITHOUT the "model." prefix (e.g.
# "layers.0.linear_attn.out_proj.lora_A"). save_state / load_state in
# train_qwen.py derive filenames from these keys via name.replace(".", "_"),
# so legacy trained/superblock_N_best/*.pt files (named
# "layers_0_linear_attn_out_proj_lora_A.pt") will NOT round-trip with the new
# key naming. This is a known consequence of the nn.Module refactor (see
# research-architecture-review/02_partial_wrapper_problem.md §9 Risk #1 and
# §6.3 for the migration shim proposal). Checkpoint migration is out of scope
# for Patch 9 and will be handled separately by the orchestrator. New
# checkpoints saved after this refactor will use the "model." prefix
# consistently and will round-trip correctly.


def load_qwen_super_block_only(sb_idx, model_name="Qwen/Qwen3.5-4B",
                                 device="cuda", dtype=torch.bfloat16):
    """Load embed_tokens + layers 0 to sb_end-1 using standard from_pretrained.
    Loads full model, extracts prefix, deletes rest. All on GPU.

    No monkey-patching — uses standard HF loading.
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    import gc

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    n_layers_loaded = sb_end
    print(f"Loading prefix: layers 0-{sb_end-1} ({n_layers_loaded} layers)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Standard from_pretrained — loads full model on GPU
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    # Extract prefix
    lang_model = model.model  # Qwen3_5TextModel
    embed_tokens = lang_model.embed_tokens
    rotary_emb = lang_model.rotary_emb if hasattr(lang_model, 'rotary_emb') else None
    prefix_layers = [lang_model.layers[i] for i in range(sb_end)]
    final_norm = lang_model.norm if hasattr(lang_model, 'norm') else None
    full_model_config = model.config

    # Detach prefix from model
    lang_model.embed_tokens = None
    lang_model.rotary_emb = None
    lang_model.norm = None
    lang_model.layers = nn.ModuleList()
    if hasattr(model, 'lm_head'):
        del model.lm_head
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Build wrapper — PartialModel and PartialWrapper are now module-level
    # nn.Module subclasses (see above). The hand-rolled to/eval/train/
    # parameters/named_parameters/named_modules/get_submodule methods have
    # been deleted; nn.Module provides all of them correctly.
    partial = PartialModel(embed_tokens, rotary_emb, prefix_layers, final_norm, full_model_config)
    wrapper = PartialWrapper(partial, full_model_config)

    # Already on GPU from from_pretrained — just eval
    wrapper = wrapper.to(device).eval()

    n_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"  Loaded prefix: {n_params:,} params ({n_layers_loaded} layers + embed_tokens)", flush=True)
    return wrapper, tokenizer


def capture_original_weights_from_checkpoint(sb_idx, model_name="Qwen/Qwen3.5-4B",
                                              device="cuda"):
    """Load original fp16/bf16 weights from the HuggingFace checkpoint for the
    given super-block's layers, BEFORE palettization replaces the nn.Linear
    modules with PalettizedLinear. Used for LoftQ SVD initialization of LoRA
    (QwenLoRA init="loftq" branch at qwen_model.py:209-222).

    The QLoRA paper (Dettmers et al. 2023, §3.2 "LoftQ: LoRA-Fine-Tuning-aware
    Quantization") initializes LoRA A and B from the SVD of the quantization
    error: LoRA_A, LoRA_B = SVD(W_orig - W_quantized). This gives LoRA a
    non-zero warm start that directly compensates the leading quantization
    error directions, saving the first ~1000 steps of zero-init climbing.

    Returns:
        dict: {tensor_name: weight_tensor} where tensor_name is the full
        parameter name (e.g. "model.layers.0.linear_attn.in_proj_qkv.weight")
        and weight_tensor is the original weight on `device` (CUDA by default
        so the SVD in QwenLoRA.__init__ runs on-GPU without a host→device copy
        per-tensor).

    Memory:
        Temporarily loads the full HF model (Qwen3.5-4B ~8 GB on disk,
        ~16 GB in bf16 VRAM if device='cuda', ~8 GB RAM if device='cpu').
        The full model is deleted + gc'd before returning, so only the
        captured super-block weights (~1-2 GB for 4 layers) remain.

    Args:
        sb_idx: super-block index (0-7). Captures layers SUPER_BLOCKS[sb_idx].
        model_name: HF model id (default "Qwen/Qwen3.5-4B").
        device: where to place the captured weights ("cuda" by default so
            the SVD in QwenLoRA.__init__ runs on-GPU directly; the call
            site in train_qwen.py also passes device=DEVICE explicitly).
    """
    from transformers import AutoModelForCausalLM
    import gc

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    print(f"Capturing original weights for super-block {sb_idx} "
          f"(layers {sb_start}-{sb_end-1}) from {model_name}...", flush=True)

    # Load full model on CPU with low_mem to avoid OOM during build.
    # dtype=bfloat16 matches the training dtype (DTYPE in train_qwen.py).
    # The SVD in QwenLoRA will upcast to fp32 internally.
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    weights = {}
    for layer_idx in range(sb_start, sb_end):
        layer = full_model.model.layers[layer_idx]
        for name, param in layer.named_parameters():
            if name.endswith(".weight"):
                # Full HF parameter name: model.layers.{idx}.{submodule.path}.weight
                # Matches the full_name format used in build_student_super_block
                # (train_qwen.py:681: full_name = f"model.layers.{layer_idx}.{name}.weight")
                full_name = f"model.layers.{layer_idx}.{name}"
                weights[full_name] = param.data.clone().to(device)

    # Free the full model — we only need the captured super-block weights.
    del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_params = sum(t.numel() for t in weights.values())
    print(f"  Captured {len(weights)} weight tensors ({total_params:,} params) "
          f"on {device}", flush=True)
    return weights


def palettize_linear(linear_module, tensor_name, palettized_dir, device="cuda",
                     use_soft_indices=False):
    """Replace an nn.Linear with a PalettizedLinear using saved 2-bit data.

    Args:
        linear_module: nn.Linear to replace
        tensor_name: name of the tensor (for finding idx/lut files)
        palettized_dir: directory containing .idx2 and .lut_scalar files
        device: cuda

    Returns:
        PalettizedLinear module, or None if files not found
    """
    san = sanitize_name(tensor_name)
    idx_path = os.path.join(palettized_dir, f"{san}.idx2")
    lut_path = os.path.join(palettized_dir, f"{san}.lut_scalar")

    if not os.path.exists(idx_path) or not os.path.exists(lut_path):
        return None

    # Load indices and LUT
    meta_path = os.path.join(palettized_dir, "metadata.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        tensor_meta = meta.get("tensors", {}).get(tensor_name)
        if tensor_meta is None:
            return None
        stored_out, stored_in = tensor_meta["dense_shape"]
        n_groups = tensor_meta["groups"]
    else:
        # Fallback: infer from weight shape
        stored_out, stored_in = linear_module.weight.shape
        n_groups = (stored_out + GROUP_SIZE - 1) // GROUP_SIZE

    indices = load_indices(idx_path, stored_out, stored_in).to(device)
    lut = load_lut(lut_path).reshape(n_groups, PALETTE_SIZE).to(device)

    # Determine pre_transposed
    hf_out, hf_in = linear_module.weight.shape
    pre_transposed = (stored_out != hf_out)

    pal_mod = PalettizedLinear(
        linear_module, indices, n_groups, PALETTE_SIZE, GROUP_SIZE,
        pre_transposed, initial_palette=lut, use_soft_indices=use_soft_indices
    )
    return pal_mod


# ─── Attach LoRA to all palettized Linears in a layer ────────────────────
def attach_lora_to_layer(layer, rank=32, alpha=64, original_weights=None, layer_name=""):
    """Wrap every PalettizedLinear in `layer` with QwenLoRA.
    Uses LoftQ SVD init from original_weights if provided.

    Args:
        layer: a transformer layer module
        rank: LoRA rank (default 32)
        alpha: LoRA alpha (default 64)
        original_weights: dict of {tensor_name: weight_tensor} for LoftQ init
        layer_name: name of the layer (for looking up original_weights)

    Returns:
        count of wrapped Linears
    """
    n_wrapped = 0
    modules_to_wrap = []

    # Build a set of module names that are already inside a QwenLoRA (as .base)
    # These should NOT be wrapped again
    already_wrapped_names = set()
    for name, module in layer.named_modules():
        if isinstance(module, QwenLoRA):
            # The base is at module.base — find its name
            # name is the QwenLoRA's name, base is at name + ".base" (or just the Linear inside)
            already_wrapped_names.add(f"{name}.base")

    for name, module in layer.named_modules():
        # Wrap both PalettizedLinear (palettized base) and nn.Linear (dense correction base)
        # Skip if already wrapped (this module is the .base of a QwenLoRA)
        if name in already_wrapped_names:
            continue
        if isinstance(module, (PalettizedLinear, nn.Linear)) and not isinstance(module, QwenLoRA):
            full_name = f"{layer_name}.{name}" if layer_name else name
            modules_to_wrap.append((name, module, full_name))

    for name, module, full_name in modules_to_wrap:
        orig_w = original_weights.get(full_name + ".weight") if original_weights else None
        # Wrap with QwenLoRA — the base IS the PalettizedLinear
        lora_mod = QwenLoRA(module, rank=rank, alpha=alpha,
                            init="loftq", original_weight=orig_w)
        # Replace in layer
        parent = layer
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora_mod)
        n_wrapped += 1

    return n_wrapped


# ─── Insert correction layers ────────────────────────────────────────────
def insert_correction_layers(model, sb_idx, n_correction=1):
    """Insert n_correction dense GatedDeltaNet layers AFTER the super-block.

    Default: 1 correction layer (per user instruction).
    Warm-started from the last GatedDeltaNet layer in the super-block.
    Zero-init output projections so correction is a no-op at start.
    LoRA will be attached separately by attach_lora_to_layer().

    Args:
        model: the Qwen model
        sb_idx: super-block index (0-7)
        n_correction: number of correction layers (default 2)

    Returns:
        list of correction layer modules
    """
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    layers = model.model.layers

    # Warm-start from layer sb_end-2 (last GatedDeltaNet in super-block)
    warm_start_idx = sb_end - 2
    src_layer = layers[warm_start_idx]

    # IMPORTANT: use copy.deepcopy instead of type(src_layer)(...)+load_state_dict.
    # The state_dict round-trip is slow because it goes through Python
    # per-tensor with extra type / dtype checks. deepcopy does a single
    # C-level memcopy per tensor and is typically 5-10x faster.
    # Both src_layer and the new copy stay on the same GPU + dtype.
    correction_layers = []
    for i in range(n_correction):
        t0 = time.time()
        new_layer = copy.deepcopy(src_layer)
        # Zero-init output projections so the correction layer is a no-op at start
        with torch.no_grad():
            if hasattr(new_layer, 'linear_attn') and hasattr(new_layer.linear_attn, 'out_proj'):
                new_layer.linear_attn.out_proj.weight.zero_()
                if hasattr(new_layer.linear_attn.out_proj, 'bias') and new_layer.linear_attn.out_proj.bias is not None:
                    new_layer.linear_attn.out_proj.bias.zero_()
            if hasattr(new_layer, 'mlp') and hasattr(new_layer.mlp, 'down_proj'):
                new_layer.mlp.down_proj.weight.zero_()
                if hasattr(new_layer.mlp.down_proj, 'bias') and new_layer.mlp.down_proj.bias is not None:
                    new_layer.mlp.down_proj.bias.zero_()

        insert_idx = sb_end + i
        layers.insert(insert_idx, new_layer)
        correction_layers.append(new_layer)
        print(f"  Inserted correction layer {i} for super-block {sb_idx} "
              f"(deepcopy of layer {warm_start_idx}) in {time.time()-t0:.2f}s", flush=True)

    # Update config
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(layers)

    return correction_layers


# ─── Isolate super-block (freeze everything else) ──────────────────────
def isolate_super_block(model, sb_idx, n_correction=1):
    """Freeze everything except the active super-block + its correction layers.

    Super-block sb_idx covers original layers [sb_idx*4, sb_idx*4+4).
    Correction layers are inserted at indices sb_idx*4+4 and sb_idx*4+5.

    Returns:
        (trainable_count, frozen_count)
    """
    sb_start = sb_idx * 4
    sb_end = sb_idx * 4 + 4
    # After insertion of n_correction layers, they're at sb_end, sb_end+1, ...
    correction_indices = list(range(sb_end, sb_end + n_correction))
    active_indices = set(range(sb_start, sb_end)) | set(correction_indices)

    n_trainable = 0
    n_frozen = 0
    for layer_idx, layer in enumerate(model.model.layers):
        is_active = layer_idx in active_indices
        for p in layer.parameters():
            p.requires_grad_(is_active)
            if is_active:
                n_trainable += p.numel()
            else:
                n_frozen += p.numel()

    # Freeze embed_tokens, norm, lm_head (always frozen)
    for name, p in model.named_parameters():
        if not name.startswith("model.layers."):
            p.requires_grad_(False)
            n_frozen += p.numel()

    print(f"  Super-block {sb_idx}: active layers {sorted(active_indices)}", flush=True)
    print(f"  Trainable: {n_trainable:,}  Frozen: {n_frozen:,}", flush=True)
    return n_trainable, n_frozen


# ─── Capture original weights (for LoftQ init) ──────────────────────────
def capture_original_weights(model, sb_idx):
    """Capture original fp16 weights for the super-block's Linears.
    Used for LoftQ SVD init of LoRA.

    Returns:
        dict: {tensor_name: weight_tensor}
    """
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    orig_weights = {}
    for layer_idx in range(sb_start, sb_end):
        layer = model.model.layers[layer_idx]
        layer_prefix = f"model.layers.{layer_idx}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}.weight"
                orig_weights[full_name] = module.weight.data.clone()
    return orig_weights


# ─── Skip patterns (tensors NOT to palettize) ───────────────────────────
SKIP_PATTERNS = [
    "embed_tokens", "lm_head",  # kept fp16 per user instruction
    "conv1d",  # not a Linear
    "A_log", "dt_bias",  # SSM params, tiny, f32
    "norm",  # RMSNorm weights
    "in_proj_a", "in_proj_b",  # too small (32 rows)
]
MIN_PALETTIZE_ROWS = 256


def should_palettize(name, weight):
    """Decide if a Linear weight should be palettized."""
    if weight.ndim != 2:
        return False
    for skip in SKIP_PATTERNS:
        if skip in name:
            return False
    if weight.shape[0] < MIN_PALETTIZE_ROWS or weight.shape[1] < GROUP_SIZE:
        return False
    return True


# ─── Self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("Qwen3.5-4B Model Utilities — Self Test")
    print("=" * 70)
    print(f"\nSuper-blocks: {len(SUPER_BLOCKS)}")
    for i, (start, end) in enumerate(SUPER_BLOCKS):
        layers = list(range(start, end))
        types = ["F" if is_full_attn_layer(l) else "L" for l in layers]
        print(f"  Super-block {i}: layers {layers} = {types}")
    print(f"\nPalettization: {BITWIDTH}-bit, GS={GROUP_SIZE}, palette_size={PALETTE_SIZE}")
    print(f"LoRA: rank-32, fp16 ONLY")
    print(f"Skip patterns: {SKIP_PATTERNS}")
    print(f"Min palettize rows: {MIN_PALETTIZE_ROWS}")
