#!/usr/bin/env python3
"""sweep_qwen.py — Automated LR sweep for Qwen3.5-4B palettization.

Strategy:
  - Run N short training cycles (e.g., 80 steps each) with different LR combos
  - Pick the combo with best cos after N steps
  - Output: best combo + per-cycle summary table

Design:
  - Each cycle: fresh model build, train N steps, measure cos, discard model
  - To keep cycles fast (~80 steps × 3s = 4 min), use seq_len=128 batch_size=8
  - Grid: 4 groups × 3 values each = up to 81 combos. Too many.
  - Pragmatic grid: ~12 hand-picked combos that explore the relevant space.

Grid rationale (informed by Dolphin v13 defaults + Qwen's first-run behavior):
  - palettes: Dolphin used 1e-5 to 6e-5. Qwen has same palette structure.
    Try: 1e-5, 3e-5, 1e-4, 3e-4
  - lora: Dolphin used 6e-5 to 2e-4. Qwen LoRA is rank-32 (rank-16 in Dolphin).
    Try: 1e-4, 3e-4, 1e-3
  - correction: Dolphin new_block was 3e-4 to 6e-4. Qwen correction is similar size.
    Try: 1e-4, 3e-4, 1e-3
  - layernorms: small, can take 1e-4 to 1e-3.
    Try: 1e-4, 3e-4

To bound the sweep, do it in 2 phases:
  Phase A: Fix lora=3e-4, correction=1e-4, layernorms=1e-4, sweep palettes only.
  Phase B: Fix palettes=winner_A, sweep lora × correction (3×3=9 combos).
  Phase C (optional): Fix lora+correction=winners, sweep layernorms.

Total: 4 + 9 = 13 cycles × ~5 min = ~65 min.
"""
import os, sys, json, time, copy, argparse, subprocess, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from qwen_model import SUPER_BLOCKS, PalettizedLinear, QwenLoRA
from palettize_core import BITWIDTH, GROUP_SIZE, PALETTE_SIZE
from train_qwen import (
    build_student_super_block, apply_groups, build_optimizers, Muon,
    compute_loss, stream_training_data, DEVICE, DTYPE,
    PALETTIZED_BASE, TRAINED_BASE,
)


def run_one_cycle(lrs, n_steps=80, seq_len=128, batch_size=8, sb_idx=0, cached_tokens=None):
    """Build fresh student, train n_steps, return (final_cos, final_loss, history)."""
    print(f"\n{'='*60}", flush=True)
    print(f"=== Cycle: lrs={lrs}, n_steps={n_steps}", flush=True)
    print(f"{'='*60}", flush=True)

    hp = {
        "groups": {"palettes": True, "lora": True, "correction": True, "layernorms": True},
        "lrs": lrs,
        "loss_type": "1-cos+norm_mse",
        "loss_weights": {"cos": 0.5, "mse": 0.5},
        "gradient_clip": 0.3,
        "eval_every": 999,  # disable saves
        "log_every": 20,
    }

    student, tokenizer = build_student_super_block(sb_idx)
    if student is None:
        return -1.0, float('inf'), []

    apply_groups(student, hp, sb_idx)
    opt_muon, opt_adamw = build_optimizers(student, hp, sb_idx)

    # Cosine scheduler over n_steps (so LR decays properly even in short cycle)
    import math
    def lr_lambda(step):
        return 0.5 * (1.0 + math.cos(math.pi * step / max(n_steps, 1)))
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon, lr_lambda) if opt_muon else None
    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw, lr_lambda) if opt_adamw else None

    # Load teacher prefix
    from qwen_model import load_qwen_super_block_only
    teacher, _ = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
    for layer in teacher.model.layers:
        for p in layer.parameters(): p.requires_grad_(False)
    student.model.embed_tokens = teacher.model.embed_tokens

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    student.train()

    # Use cached tokens if available (avoids HF API rate-limiting)
    # Otherwise fall back to streaming
    if cached_tokens is not None:
        n_cached = cached_tokens.shape[0]
        # Yield batches in order, looping if needed
        idx = 0
        def data_gen():
            nonlocal idx
            while True:
                if idx + batch_size > n_cached:
                    idx = 0  # wrap around
                batch = cached_tokens[idx:idx+batch_size].to(DEVICE)
                idx += batch_size
                yield batch
        data_stream = data_gen()
    else:
        data_stream = stream_training_data(tokenizer, n_seqs=10**12, seq_len=seq_len, device=DEVICE, batch_size=batch_size)

    history = []
    global_step = 0
    start_time = time.time()

    for batch_ids in data_stream:
        if global_step >= n_steps: break

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
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

        s_h = student.model.embed_tokens(batch_ids)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
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
        if not torch.isfinite(loss):
            print(f"  [step {global_step}] NaN loss — ABORTING cycle", flush=True)
            return -1.0, float('inf'), history

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.grad is not None],
            hp.get("gradient_clip", 0.3))
        if opt_muon: opt_muon.step()
        if opt_adamw: opt_adamw.step()
        if sched_muon: sched_muon.step()
        if sched_adamw: sched_adamw.step()
        if opt_muon: opt_muon.zero_grad(set_to_none=True)
        if opt_adamw: opt_adamw.zero_grad(set_to_none=True)
        global_step += 1

        cos_val = 1.0 - comps["cos"]
        if global_step % 20 == 0 or global_step == n_steps:
            tps = global_step / max(1e-6, time.time() - start_time)
            print(f"  step={global_step:3d} loss={comps['loss']:.4f} cos={cos_val:.4f} tps={tps:.1f}", flush=True)
            history.append((global_step, comps['loss'], cos_val))

        del batch_ids, h_out, student_out, loss, comps

    final_cos = history[-1][2] if history else -1.0
    final_loss = history[-1][1] if history else float('inf')
    print(f"  → final cos={final_cos:.4f}, loss={final_loss:.4f}", flush=True)

    # Cleanup
    del student, teacher, opt_muon, opt_adamw, sched_muon, sched_adamw
    torch.cuda.empty_cache()
    gc.collect()

    return final_cos, final_loss, history


def main():
    import gc
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=80,
                    help="Steps per cycle (80 ≈ 4 min on L4)")
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--out", type=str, default="/tmp/sweep_results.json")
    ap.add_argument("--cache", type=str, default="/root/qwen35_palettize/cached_tokens.pt",
                    help="Path to pre-tokenized .pt cache. If missing, will create it.")
    args = ap.parse_args()

    # Load (or create) cached tokens once, reuse across all cycles
    cached_tokens = None
    if os.path.exists(args.cache):
        cached_tokens = torch.load(args.cache, weights_only=False)
        print(f"Loaded cached tokens: {cached_tokens.shape} from {args.cache}", flush=True)
    else:
        print(f"Cache not found at {args.cache}.", flush=True)
        print("Run cache_tokens.py first, then re-run sweep.", flush=True)
        return

    # Sweep grid
    # Phase A: sweep palettes only (others fixed at sensible defaults)
    phase_A = [
        {"palettes": 1e-5, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4},
        {"palettes": 3e-5, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4},
        {"palettes": 1e-4, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4},  # baseline
        {"palettes": 3e-4, "lora": 3e-4, "correction": 1e-4, "layernorms": 1e-4},
    ]
    # Phase B: sweep lora × correction (3×3 = 9 combos)
    phase_B = []
    for lora in [1e-4, 3e-4, 1e-3]:
        for correction in [1e-4, 3e-4, 1e-3]:
            phase_B.append({
                "palettes": None,  # filled in from Phase A winner
                "lora": lora,
                "correction": correction,
                "layernorms": 1e-4,
            })
    # Phase C: sweep layernorms only (others fixed at Phase A+B winners)
    phase_C = []
    for ln in [1e-4, 3e-4, 1e-3]:
        phase_C.append({
            "palettes": None,
            "lora": None,
            "correction": None,
            "layernorms": ln,
        })

    results = {"phase_A": [], "phase_B": [], "phase_C": [],
               "best_A": None, "best_B": None, "best_C": None,
               "final_best": None}

    # Phase A
    if args.phase in ["A", "all"]:
        print(f"\n{'#'*60}\n# Phase A: sweep palettes (4 cycles)\n{'#'*60}", flush=True)
        for lrs in phase_A:
            cos, loss, hist = run_one_cycle(lrs, n_steps=args.n_steps, cached_tokens=cached_tokens)
            results["phase_A"].append({"lrs": lrs, "final_cos": cos, "final_loss": loss, "history": hist})
        # Pick winner
        best = max(results["phase_A"], key=lambda r: r["final_cos"])
        results["best_A"] = best
        best_palette = best["lrs"]["palettes"]
        print(f"\n>>> Phase A winner: palettes={best_palette} cos={best['final_cos']:.4f}", flush=True)

        # Fill palettes into Phase B
        for c in phase_B:
            c["palettes"] = best_palette
    else:
        # Need best_A from previous run
        if os.path.exists(args.out):
            prev = json.load(open(args.out))
            if prev.get("best_A"):
                best_palette = prev["best_A"]["lrs"]["palettes"]
                for c in phase_B:
                    c["palettes"] = best_palette
                results["best_A"] = prev["best_A"]
        else:
            print("ERROR: need to run Phase A first")
            return

    # Phase B
    if args.phase in ["B", "all"]:
        print(f"\n{'#'*60}\n# Phase B: sweep lora × correction (9 cycles, palettes={best_palette})\n{'#'*60}", flush=True)
        for lrs in phase_B:
            cos, loss, hist = run_one_cycle(lrs, n_steps=args.n_steps, cached_tokens=cached_tokens)
            results["phase_B"].append({"lrs": lrs, "final_cos": cos, "final_loss": loss, "history": hist})
        best = max(results["phase_B"], key=lambda r: r["final_cos"])
        results["best_B"] = best
        best_lora = best["lrs"]["lora"]
        best_correction = best["lrs"]["correction"]
        print(f"\n>>> Phase B winner: lora={best_lora} correction={best_correction} cos={best['final_cos']:.4f}", flush=True)

        for c in phase_C:
            c["palettes"] = best_palette
            c["lora"] = best_lora
            c["correction"] = best_correction
    else:
        if results.get("best_B"):
            best_lora = results["best_B"]["lrs"]["lora"]
            best_correction = results["best_B"]["lrs"]["correction"]
            for c in phase_C:
                c["palettes"] = best_palette
                c["lora"] = best_lora
                c["correction"] = best_correction

    # Phase C
    if args.phase in ["C", "all"]:
        print(f"\n{'#'*60}\n# Phase C: sweep layernorms (3 cycles)\n{'#'*60}", flush=True)
        for lrs in phase_C:
            cos, loss, hist = run_one_cycle(lrs, n_steps=args.n_steps, cached_tokens=cached_tokens)
            results["phase_C"].append({"lrs": lrs, "final_cos": cos, "final_loss": loss, "history": hist})
        best = max(results["phase_C"], key=lambda r: r["final_cos"])
        results["best_C"] = best
        print(f"\n>>> Phase C winner: layernorms={best['lrs']['layernorms']} cos={best['final_cos']:.4f}", flush=True)

    # Final best
    if results.get("best_C"):
        results["final_best"] = results["best_C"]
    elif results.get("best_B"):
        results["final_best"] = results["best_B"]
    elif results.get("best_A"):
        results["final_best"] = results["best_A"]

    # Save results
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Sweep complete. Results saved to {args.out} ===", flush=True)
    if results.get("final_best"):
        print(f"\nFINAL BEST LRs: {results['final_best']['lrs']}", flush=True)
        print(f"  cos after {args.n_steps} steps: {results['final_best']['final_cos']:.4f}", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
