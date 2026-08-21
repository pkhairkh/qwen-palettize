#!/usr/bin/env python3
"""calib_stage2.py — Stage 2 calibration: palettize correction layer's dense + LoRA weights.

Stage 1 (already done): palettize layers 0-3 dense weights to 2-bit.
Stage 2 (this script): palettize the correction layer's:
  - Dense weights (in_proj_qkv, in_proj_z, out_proj, mlp gate/up/down)
  - LoRA weights (lora_A, lora_B for all correction Linears)
Stage 3 (future): palettize the LoRA on the full attention layer.

Workflow:
  1. Build the student as it was at the end of stage 1 (layers 0-3 palettized +
     correction layer dense + correction LoRA).
  2. Load the trained weights from trained/superblock_0_best/ (step 5500, cos=0.910).
  3. Run forward passes on calibration data to capture input activations for
     every Linear in the correction layer (both the base Linear and the LoRA
     inputs).
  4. Palettize each weight to 2-bit GS=256 using GPTQ + weighted kmeans1d.
  5. Save .idx2 + .lut_scalar files to palettized/superblock_0_stage2/.

Usage:
  python3 calib_stage2.py --sb_idx 0 --n_seqs 1024 --seq_len 512
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import palettize_tensor_2bit, write_metadata_json, BITWIDTH, GROUP_SIZE
from qwen_model import (
    SUPER_BLOCKS, is_full_attn_layer, load_qwen_super_block_only,
    insert_correction_layers, attach_lora_to_layer,
    should_palettize, palettize_linear, QwenLoRA, PalettizedLinear,
)
from train_qwen import (
    build_student_super_block, apply_groups, load_state, DEVICE, DTYPE,
    PALETTIZED_BASE, TRAINED_BASE,
)

OUT_BASE = "/root/qwen35_palettize/palettized"
DEFAULT_N_SEQS = 1024
DEFAULT_SEQ_LEN = 512
MAX_SAMPLES_PER_LINEAR = 16384


# ─── Activation accumulator (simplified from calib_qwen.py) ─────────────
class ActAcc:
    """Accumulate H = X^T X + sample of X for one Linear."""
    def __init__(self, name, in_dim, device="cuda"):
        self.name = name
        self.in_dim = in_dim
        self.H = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
        self.hess_diag = torch.zeros(in_dim, dtype=torch.float32, device=device)
        self.X_sample = torch.zeros(0, in_dim, dtype=torch.float16, device=device)
        self.n_samples = 0

    def add_batch(self, x):
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.detach()
        x_f = x.float()
        self.H += x_f.T @ x_f
        self.hess_diag += (x_f * x_f).sum(dim=0)
        if self.X_sample.shape[0] < MAX_SAMPLES_PER_LINEAR:
            need = MAX_SAMPLES_PER_LINEAR - self.X_sample.shape[0]
            take = min(need, x.shape[0])
            self.X_sample = torch.cat([self.X_sample, x[:take].to(torch.float16)], dim=0)
        self.n_samples += x.shape[0]

    def cleanup(self):
        del self.H, self.hess_diag, self.X_sample
        self.H = None; self.hess_diag = None; self.X_sample = None
        torch.cuda.empty_cache()


def register_hooks_on_correction(model, sb_idx, device="cuda"):
    """Register hooks on every Linear + LoRA inside the correction layer.

    The correction layer is at index sb_end (the first layer after the
    super-block's 4 original layers).

    Returns:
        accumulators: {tensor_name: ActAcc}
        hooks: list of hook handles
    """
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    correction_idx = sb_end
    accumulators = {}
    hooks = []

    # Walk the correction layer's modules
    correction_layer = model.model.layers[correction_idx]
    for name, module in correction_layer.named_modules():
        # Hook nn.Linear (the base dense weights of the correction layer)
        if isinstance(module, nn.Linear):
            full_name = f"model.layers.{correction_idx}.{name}.weight"
            in_dim = module.in_features
            acc = ActAcc(full_name, in_dim, device=device)
            accumulators[full_name] = acc
            def make_hook(a):
                def hook(mod, inp, out):
                    a.add_batch(inp[0])
                return hook
            hooks.append(module.register_forward_hook(make_hook(acc)))
        # Hook QwenLoRA's internal matmul — we need the INPUT to lora_A
        # QwenLoRA.forward does: lora_out = (x_flat @ self.lora_A) @ self.lora_B.T
        # The input to lora_A is x_flat (same as input to base).
        # So we hook the QwenLoRA module itself and capture its input.
        if isinstance(module, QwenLoRA):
            # lora_A: (in_dim, rank), lora_B: (out_dim, rank)
            in_dim = module.lora_A.shape[0]
            # Hook for lora_A input (= input to the QwenLoRA module)
            acc_a = ActAcc(f"model.layers.{correction_idx}.{name}.lora_A", in_dim, device=device)
            accumulators[acc_a.name] = acc_a
            def make_hook_a(a):
                def hook(mod, inp, out):
                    a.add_batch(inp[0])
                return hook
            hooks.append(module.register_forward_hook(make_hook_a(acc_a)))
            # For lora_B: the input is (x @ lora_A), shape (batch, rank).
            # We need to capture that intermediate. Hook the output of lora_A.
            # But QwenLoRA computes (x_flat @ lora_A) inline — there's no separate module.
            # Instead, we'll reconstruct the lora_B input during palettization:
            #   X_B = X_A @ lora_A  (we already have X_A sampled)
            # So we don't need a separate hook for lora_B — we compute it from X_A.
            # Just store the lora_B accumulator with in_dim = rank
            rank = module.lora_A.shape[1]
            acc_b = ActAcc(f"model.layers.{correction_idx}.{name}.lora_B", rank, device=device)
            accumulators[acc_b.name] = acc_b
            # Mark it as "derived" — will be filled after forward pass
            acc_b._derived = True
            acc_b._src_acc = acc_a
            acc_b._lora_A_param = module.lora_A

    print(f"  Registered hooks on {len(accumulators)} tensors in correction layer {correction_idx}", flush=True)
    return accumulators, hooks


def fill_derived_accumulators(accumulators):
    """For lora_B accumulators (derived), compute X_B = X_A @ lora_A and fill H + sample."""
    for name, acc in accumulators.items():
        if not hasattr(acc, '_derived'):
            continue
        src = acc._src_acc
        lora_A = acc._lora_A_param
        if src.X_sample.shape[0] == 0:
            continue
        X_A = src.X_sample.float()
        X_B = X_A @ lora_A.float()  # (n_samples, rank)
        acc.H = X_B.T @ X_B
        acc.hess_diag = (X_B * X_B).sum(dim=0)
        acc.X_sample = X_B.to(torch.float16)
        acc.n_samples = src.n_samples


def stream_calibration_data(tokenizer, n_seqs, seq_len, device="cuda"):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    produced = 0
    buf = []
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
        while len(buf) >= 1 and produced < n_seqs:
            batch = torch.stack(buf[:1]).to(device)
            del buf[:1]
            produced += 1
            yield batch
        if produced >= n_seqs: return


def calibrate_stage2(sb_idx, n_seqs, seq_len):
    """Stage 2: palettize correction layer's dense + LoRA weights to 2-bit."""
    print(f"\n{'='*70}")
    print(f"=== Stage 2 Calibration: super-block {sb_idx} ===")
    print(f"  seqs: {n_seqs}  seq_len: {seq_len}")
    print(f"{'='*70}")

    out_dir = os.path.join(OUT_BASE, f"superblock_{sb_idx}_stage2")
    os.makedirs(out_dir, exist_ok=True)

    # Build the stage-1 student (layers 0-3 palettized + correction dense + LoRA)
    # and load the trained weights
    print(f"\n=== Building stage-1 student + loading trained weights ===", flush=True)
    student, tokenizer = build_student_super_block(sb_idx, lora_rank=32, lora_alpha=64, stage=1)
    if student is None:
        print("FAILED to build student")
        return

    # Load trained weights from step 5500 save
    hp = {"groups": {"palettes": True, "lora": True, "correction": True, "layernorms": True}}
    apply_groups(student, hp, sb_idx)
    resume_dir = os.path.join(TRAINED_BASE, f"superblock_{sb_idx}_best")
    load_state(student, resume_dir)

    # Load teacher (prefix only — same as training)
    print(f"\n=== Loading teacher (prefix) ===", flush=True)
    teacher, _ = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
    for layer in teacher.model.layers:
        for p in layer.parameters(): p.requires_grad_(False)
    student.model.embed_tokens = teacher.model.embed_tokens

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    student.eval()  # eval mode for calibration

    # Register hooks on correction layer
    print(f"\n=== Registering hooks on correction layer ===", flush=True)
    accumulators, hooks = register_hooks_on_correction(student, sb_idx, device=DEVICE)

    # Stream calibration data through the full student (teacher prefix + student super-block + correction)
    print(f"\n=== Streaming {n_seqs} calibration sequences ===", flush=True)
    t0 = time.time()
    n_done = 0
    for batch_ids in stream_calibration_data(tokenizer, n_seqs, seq_len, device=DEVICE):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                # Teacher prefix: embed + layers 0 to sb_end-1
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
                h_in = h.detach()

                # Student super-block + correction layer
                s_h = student.model.embed_tokens(batch_ids)
                s_pos_emb = None
                if hasattr(student.model, 'rotary_emb') and student.model.rotary_emb is not None:
                    s_pos_emb = student.model.rotary_emb(s_h, position_ids)
                for layer_idx in range(sb_end + 1):  # +1 for correction layer
                    if layer_idx < len(student.model.layers):
                        layer = student.model.layers[layer_idx]
                        out = layer(s_h, position_embeddings=s_pos_emb) if s_pos_emb is not None else layer(s_h)
                        s_h = out[0] if isinstance(out, tuple) else out
        n_done += batch_ids.shape[0]
        if n_done % 50 == 0 or n_done >= n_seqs:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(f"  {n_done}/{n_seqs}  ({rate:.1f} seq/s)", flush=True)
        del batch_ids, h_in, s_h
        torch.cuda.empty_cache()

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()

    # Fill derived accumulators (lora_B = X_A @ lora_A)
    print(f"\n=== Filling derived lora_B accumulators ===", flush=True)
    fill_derived_accumulators(accumulators)

    # Palettize each weight
    print(f"\n=== Palettizing {len(accumulators)} tensors to 2-bit GS=256 ===", flush=True)
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    correction_idx = sb_end
    correction_layer = student.model.layers[correction_idx]

    tensor_metas = {}
    cos_values = []

    for wname, acc in accumulators.items():
        if acc.X_sample.shape[0] < 10:
            print(f"  SKIP {wname} (no activations)", flush=True)
            acc.cleanup()
            continue

        # Find the weight tensor
        # wname format: "model.layers.{correction_idx}.{rest}.weight" or
        #               "model.layers.{correction_idx}.{rest}.lora_A" or .lora_B
        if wname.endswith(".weight"):
            # Dense weight — find the nn.Linear or PalettizedLinear
            mod_path = wname[:-len(".weight")].replace("model.", "")
            mod = student.get_submodule(mod_path)
            W = mod.weight.data.to(torch.float32)
            X = acc.X_sample.to(torch.float32)
        elif wname.endswith(".lora_A"):
            mod_path = wname.replace("model.", "").replace(".lora_A", "")
            mod = student.get_submodule(mod_path)
            W = mod.lora_A.data.to(torch.float32)  # (in_dim, rank) e.g. (4096, 32)
            # KEY FIX: for lora_A, use the INTERMEDIATE (x @ lora_A, shape (n_samples, rank))
            # as X — NOT the raw input (n_samples, in_dim). This makes hess_diag=(rank,)
            # match W.shape[1]=rank, so palettize_tensor_2bit can group along out_dim=in_dim
            # with GS=256. The intermediate is stored in the corresponding lora_B accumulator.
            # Find the lora_B accumulator for this same QwenLoRA module
            lora_b_name = wname.replace(".lora_A", ".lora_B")
            if lora_b_name in accumulators:
                X = accumulators[lora_b_name].X_sample.to(torch.float32)
            else:
                # Fallback: compute intermediate from acc's X_sample @ lora_A
                X = (acc.X_sample.float() @ W).to(torch.float32)
        elif wname.endswith(".lora_B"):
            mod_path = wname.replace("model.", "").replace(".lora_B", "")
            mod = student.get_submodule(mod_path)
            W = mod.lora_B.data.to(torch.float32)  # (out_dim, rank) e.g. (2560, 32)
            # For lora_B: X = intermediate (already computed by fill_derived_accumulators)
            X = acc.X_sample.to(torch.float32)
        else:
            print(f"  SKIP unknown weight type: {wname}", flush=True)
            acc.cleanup()
            continue

        print(f"\n  [{wname}] W={tuple(W.shape)} X={tuple(X.shape)} n_samples={X.shape[0]}", flush=True)
        meta = palettize_tensor_2bit(wname, W, X, out_dir, threshold=0.0, verbose=True)
        if meta is not None:
            tensor_metas[wname] = meta
            cos_values.append(meta.get("_achieved_cos", 0.0))
        acc.cleanup()
        del W, X
        torch.cuda.empty_cache()

    if cos_values:
        print(f"\n=== Palettization summary ===", flush=True)
        print(f"  {len(cos_values)} tensors palettized", flush=True)
        print(f"  cos: min={min(cos_values):.6f}  mean={sum(cos_values)/len(cos_values):.6f}  max={max(cos_values):.6f}", flush=True)

    meta_path = os.path.join(out_dir, "metadata.json")
    write_metadata_json(meta_path, tensor_metas)
    print(f"\n=== Wrote metadata to {meta_path} ===", flush=True)
    print(f"  {len(tensor_metas)} tensor entries", flush=True)

    del student, teacher
    torch.cuda.empty_cache()
    print(f"\n=== Stage 2 calibration complete ===", flush=True)
    return tensor_metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_idx", type=int, default=0)
    ap.add_argument("--n_seqs", type=int, default=DEFAULT_N_SEQS)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    args = ap.parse_args()

    print(f"=== Stage 2 Calibration: correction layer → 2-bit ===", flush=True)
    print(f"  Super-block: {args.sb_idx}", flush=True)
    print(f"  Seqs: {args.n_seqs}, seq_len: {args.seq_len}", flush=True)
    print(f"  Output: {OUT_BASE}/superblock_{args.sb_idx}_stage2/", flush=True)

    calibrate_stage2(args.sb_idx, args.n_seqs, args.seq_len)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
