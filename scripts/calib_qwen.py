#!/usr/bin/env python3
"""calib_qwen.py — Calibration for 2-bit GS=256 palettization of Qwen3.5-4B.

Per super-block:
  1. Load teacher (fp16), capture input activations for every Linear via forward hooks
  2. Stream a BIG calibration set (default 8192 sequences × 2048 tokens = 16.7M tokens)
  3. Run GPTQ + weighted kmeans1d per group → 2-bit indices + fp16 LUT
  4. Write .idx2 + .lut_scalar files + metadata.json

Memory strategy:
  - Teacher: fp16, full model (shared embeddings across super-blocks)
  - Calib set: large — we accumulate H = X^T X per Linear (not full X)
  - No full forward pass per Linear — hooks capture activations during normal forward
  - NO FP32 weights (everything fp16/fp32-accumulator only)

Usage:
  python3 calib_qwen.py --sb_idx 0 --n_seqs 8192 --seq_len 2048
  python3 calib_qwen.py --sb_idx 0 --sb_idx 1  # calibrate multiple super-blocks
  python3 calib_qwen.py --all  # calibrate all 8 super-blocks
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import palettize_tensor_2bit, write_metadata_json, BITWIDTH, GROUP_SIZE
from qwen_model import (
    SUPER_BLOCKS, get_super_block_layers, is_full_attn_layer,
    should_palettize, load_qwen_model,
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
DEVICE = "cuda"
DTYPE = torch.float16
OUT_BASE = "/root/qwen35_palettize/palettized"
DEFAULT_N_SEQS = 8192
DEFAULT_SEQ_LEN = 2048
BATCH_SIZE = 1  # sequences per forward pass (memory-limited — Qwen3.5-4B is big)
MAX_SAMPLES_PER_LINEAR = 32768  # max rows of X kept per Linear for cosine check
MAX_ACT_BYTES = 512 * 1024 * 1024  # 512MB safety cap per Linear accumulator


# ─── Activation accumulator ─────────────────────────────────────────────
class ActivationAccumulator:
    """Per-Linear: accumulate H = X^T X (Hessian), hess_diag = sum(X^2),
    and keep a sample of X rows for cosine check.

    Memory-efficient: we only store H (in_dim × in_dim) + hess_diag + a sample of X.
    We do NOT store the full activation tensor.

    All accumulation in fp32 for numerical stability, but weights stay fp16.
    """
    def __init__(self, name, in_dim, device="cuda"):
        self.name = name
        self.in_dim = in_dim
        # Hessian (in_dim × in_dim) — this is the big one
        # For in_dim=2560, H = 2560² × 4 bytes = 26 MB per Linear
        # For in_dim=9216, H = 9216² × 4 bytes = 339 MB per Linear (large!)
        self.H = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
        self.hess_diag = torch.zeros(in_dim, dtype=torch.float32, device=device)
        self.X_sample = torch.zeros(0, in_dim, dtype=torch.float16, device=device)
        self.n_samples = 0

    def add_batch(self, x):
        """Add a batch of activations.
        x: (B, S, in_dim) or (B, in_dim) or (S, in_dim) — flatten to 2D
        """
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.detach()
        # Cast to fp32 for accumulation (numerical stability)
        x_f = x.float()
        # Accumulate Hessian: H += X^T X
        self.H += x_f.T @ x_f
        # hess_diag += sum(X^2, dim=0)
        self.hess_diag += (x_f * x_f).sum(dim=0)
        # Sample: keep first N rows in fp16
        if self.X_sample.shape[0] < MAX_SAMPLES_PER_LINEAR:
            need = MAX_SAMPLES_PER_LINEAR - self.X_sample.shape[0]
            take = min(need, x.shape[0])
            self.X_sample = torch.cat([self.X_sample, x[:take].to(torch.float16)], dim=0)
        self.n_samples += x.shape[0]

    def cleanup(self):
        """Free GPU memory."""
        del self.H, self.hess_diag, self.X_sample
        self.H = None
        self.hess_diag = None
        self.X_sample = None
        torch.cuda.empty_cache()


# ─── Hook registration ──────────────────────────────────────────────────
def register_hooks_for_super_block(model, sb_idx, device="cuda"):
    """Register forward hooks on every palettizable Linear in the super-block.

    The partial model's named_modules uses "layers.{idx}.{rest}" format (no "model." prefix).
    We store accumulator names with the ORIGINAL tensor name format
    "model.layers.{idx}.{rest}.weight" for metadata consistency.

    Returns:
        (accumulator_map, hook_list)
        accumulator_map: {tensor_name: ActivationAccumulator}
    """
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    accumulators = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # name format from partial model: "layers.0.self_attn.q_proj"
        # Convert to canonical: "model.layers.0.self_attn.q_proj.weight"
        # Parse layer index
        parts = name.split(".")
        if parts[0] != "layers" or not parts[1].isdigit():
            continue
        layer_idx = int(parts[1])
        # Only hook layers in the super-block (sb_start to sb_end-1)
        if layer_idx < sb_start or layer_idx >= sb_end:
            continue
        # Build canonical tensor name
        full_name = f"model.{name}.weight"
        if not should_palettize(full_name, module.weight):
            continue
        in_dim = module.in_features
        acc = ActivationAccumulator(full_name, in_dim, device=device)
        accumulators[full_name] = acc

        def make_hook(a):
            def hook(mod, inp, out):
                a.add_batch(inp[0])
            return hook
        h = module.register_forward_hook(make_hook(acc))
        hooks.append(h)

    print(f"  Registered hooks on {len(accumulators)} Linears in super-block {sb_idx} (layers {sb_start}-{sb_end-1})", flush=True)
    return accumulators, hooks


# ─── Streaming calibration data ─────────────────────────────────────────
def stream_calibration_data(tokenizer, n_seqs, seq_len, device="cuda"):
    """Stream calibration sequences from FineWeb-Edu.

    Yields batches of input_ids: (BATCH_SIZE, seq_len) long tensor on device.
    All sequences are padded/truncated to exactly seq_len.
    """
    from datasets import load_dataset
    print(f"  Loading FineWeb-Edu dataset...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True,
                      name="sample-10BT")  # 10BT sample for diversity

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buf = []
    produced = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 100:
            continue
        # Tokenize with truncation to seq_len
        ids = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=seq_len, return_tensors="pt")["input_ids"].squeeze(0)
        if ids.numel() < 64:
            continue  # skip very short sequences
        # Pad to seq_len (so all sequences are equal size for stacking)
        if ids.numel() < seq_len:
            pad = torch.full((seq_len - ids.numel(),), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:seq_len]
        buf.append(ids)
        while len(buf) >= BATCH_SIZE and produced < n_seqs:
            batch = torch.stack(buf[:BATCH_SIZE]).to(device)
            del buf[:BATCH_SIZE]
            produced += BATCH_SIZE
            yield batch
        if produced >= n_seqs:
            return


# ─── Main calibration for one super-block ──────────────────────────────
def calibrate_super_block(sb_idx, n_seqs, seq_len):
    """Calibrate one super-block to 2-bit GS=256.

    Loads embed_tokens + layers 0 to sb_end-1 (the PREFIX).
    For super-block 0: layers 0-3 only. For super-block 7: layers 0-31 (full).
    Runs forward through the prefix only — hooks capture activations at the super-block's Linears.

    Args:
        sb_idx: super-block index (0-7)
        n_seqs: number of calibration sequences
        seq_len: sequence length
    """
    print(f"\n{'='*70}")
    print(f"=== Calibrating super-block {sb_idx} ===")
    print(f"  seqs: {n_seqs}  seq_len: {seq_len}  total tokens: {n_seqs * seq_len:,}")
    print(f"{'='*70}")

    out_dir = os.path.join(OUT_BASE, f"superblock_{sb_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Load ONLY the prefix (embed_tokens + layers 0 to sb_end-1)
    # Super-block 0: 4 layers. Super-block 7: 32 layers (full model).
    from qwen_model import load_qwen_super_block_only, SUPER_BLOCKS
    model, tokenizer = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]

    # Register hooks on the super-block's Linears ONLY (layers sb_start to sb_end-1)
    # The prefix layers 0 to sb_start-1 are "passthrough" — we don't capture their activations
    print(f"\n=== Registering activation hooks ===", flush=True)
    accumulators, hooks = register_hooks_for_super_block(model, sb_idx, device=DEVICE)

    # Stream calibration data through the model
    print(f"\n=== Streaming {n_seqs} calibration sequences through prefix ({sb_end} layers) ===", flush=True)
    t0 = time.time()
    n_done = 0
    for batch_ids in stream_calibration_data(tokenizer, n_seqs, seq_len, device=DEVICE):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                # Forward: embed_tokens → compute position_embeddings → run prefix layers
                # Hooks capture activations at layers sb_start to sb_end-1 (the super-block)
                h = model.model.embed_tokens(batch_ids)
                # Compute position_ids [0, 1, ..., seq_len-1]
                seq_len_actual = batch_ids.shape[1]
                position_ids = torch.arange(seq_len_actual, device=batch_ids.device).unsqueeze(0)
                # Compute rotary position embeddings (cos, sin tuple)
                if model.model.rotary_emb is not None:
                    position_embeddings = model.model.rotary_emb(h, position_ids)
                else:
                    position_embeddings = None
                # Run all prefix layers (0 to sb_end-1)
                for layer in model.model.layers:
                    if position_embeddings is not None:
                        out = layer(h, position_embeddings=position_embeddings)
                    else:
                        out = layer(h)
                    h = out[0] if isinstance(out, tuple) else out
                # Done — hooks captured activations. No lm_head, no layers after sb_end.
        n_done += batch_ids.shape[0]
        if n_done % 100 == 0 or n_done >= n_seqs:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(f"  {n_done}/{n_seqs}  ({rate:.1f} seq/s, {elapsed:.0f}s elapsed)", flush=True)
        del batch_ids, h
        torch.cuda.empty_cache()

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Drop embed_tokens + prefix layers 0 to sb_start-1 to free memory before palettization
    # We only need the super-block's layers (sb_start to sb_end-1) for palettization
    print(f"\n=== Dropping embed_tokens + prefix layers 0-{sb_start-1} ===", flush=True)
    if hasattr(model.model, 'embed_tokens') and model.model.embed_tokens is not None:
        del model.model.embed_tokens
        model.model.embed_tokens = None
    # Drop prefix layers before the super-block (keep only sb_start to sb_end-1)
    if sb_start > 0:
        model.model.layers = model.model.layers[sb_start:]
    torch.cuda.empty_cache()

    # Palettize each Linear
    print(f"\n=== Palettizing {len(accumulators)} tensors to 2-bit GS=256 ===", flush=True)
    tensor_metas = {}
    cos_values = []
    t_pal_start = time.time()

    for i, (wname, acc) in enumerate(accumulators.items()):
        # Get the Linear module
        # wname format: "model.layers.{layer_idx}.{rest}.weight"
        # The partial model's named_modules uses "layers.{layer_idx}.{rest}" (no "model." prefix)
        mod_name = wname[:-len(".weight")]  # "model.layers.0.self_attn.q_proj"
        mod_name = mod_name.replace("model.", "")  # "layers.0.self_attn.q_proj"
        # The partial model's get_submodule expects "layers.{idx}.{rest}"
        # After dropping prefix layers, the layers list is reindexed to 0-3
        # But the accumulator name has the ORIGINAL layer index
        # We need to subtract sb_start from the layer index
        parts = mod_name.split(".")
        if parts[0] == "layers" and parts[1].isdigit():
            orig_idx = int(parts[1])
            reindexed = orig_idx - sb_start  # 0-3 after dropping prefix
            parts[1] = str(reindexed)
            mod_name = ".".join(parts)

        mod = model.get_submodule(mod_name)
        # Work in fp32 ONLY for the palettization math (weights are fp16, upcast for kmeans)
        W = mod.weight.data.to(torch.float32)
        X = acc.X_sample.to(torch.float32) if acc.X_sample.shape[0] > 0 else None

        if X is None or X.shape[0] < 10:
            print(f"  [{i+1}/{len(accumulators)}] {wname} — SKIP (no activations captured)", flush=True)
            acc.cleanup()
            continue

        print(f"\n[{i+1}/{len(accumulators)}] {wname}", flush=True)
        meta = palettize_tensor_2bit(wname, W, X, out_dir, threshold=0.0, verbose=True)
        # threshold=0.0 means always accept (2-bit only, no fallback)
        if meta is not None:
            tensor_metas[wname] = meta
            cos_values.append(meta.get("_achieved_cos", 0.0))

        # Free accumulator
        acc.cleanup()
        del W, X
        torch.cuda.empty_cache()

    print(f"\n=== Palettization done in {time.time()-t_pal_start:.0f}s ===", flush=True)
    if cos_values:
        print(f"  {len(cos_values)} tensors palettized", flush=True)
        print(f"  cos: min={min(cos_values):.6f}  mean={sum(cos_values)/len(cos_values):.6f}  max={max(cos_values):.6f}", flush=True)

    # Write metadata.json
    meta_path = os.path.join(out_dir, "metadata.json")
    write_metadata_json(meta_path, tensor_metas)
    print(f"\n=== Wrote metadata to {meta_path} ===", flush=True)
    print(f"  {len(tensor_metas)} tensor entries", flush=True)

    # Free model
    del model, tokenizer
    torch.cuda.empty_cache()

    print(f"\n=== Super-block {sb_idx} calibration complete ===", flush=True)
    return tensor_metas


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_idx", type=int, nargs="+", default=[0],
                    help="Super-block index(es) to calibrate (0-7)")
    ap.add_argument("--all", action="store_true", help="Calibrate all 8 super-blocks")
    ap.add_argument("--n_seqs", type=int, default=DEFAULT_N_SEQS)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    args = ap.parse_args()

    if args.all:
        sb_indices = list(range(8))
    else:
        sb_indices = args.sb_idx

    print(f"=== Qwen3.5-4B 2-bit GS=256 Calibration ===", flush=True)
    print(f"  Super-blocks: {sb_indices}", flush=True)
    print(f"  Seqs per super-block: {args.n_seqs}", flush=True)
    print(f"  Seq length: {args.seq_len}", flush=True)
    print(f"  Total tokens per super-block: {args.n_seqs * args.seq_len:,}", flush=True)
    print(f"  Output: {OUT_BASE}", flush=True)

    os.makedirs(OUT_BASE, exist_ok=True)

    for sb_idx in sb_indices:
        calibrate_super_block(sb_idx, args.n_seqs, args.seq_len)

    print(f"\n=== All calibration complete ===", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
