#!/usr/bin/env python3
"""Profile training step components to find what's getting slower over time.

Measures:
1. Data loading time (streaming from HF vs cached .pt)
2. Teacher forward time
3. Student forward time (including PalettizedLinear gather)
4. Backward time
5. Optimizer step time (Muon + AdamW with fp32 masters)
6. Eval time

Runs for N steps, logs per-component timing every 50 steps to see what grows.
"""
import os, sys, json, time, argparse, math, gc
sys.path.insert(0, "/root/qwen35_palettize/scripts")

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_model import (
    SUPER_BLOCKS, PalettizedLinear, QwenLoRA,
    load_qwen_super_block_only, insert_correction_layers,
    attach_lora_to_layer, should_palettize, palettize_linear,
    palettize_lora_weights,
)
from train_qwen import (
    build_student_super_block, apply_groups, build_optimizers,
    compute_loss, stream_training_data, prepare_eval_set, evaluate,
    load_state, DEVICE, DTYPE, PALETTIZED_BASE, TRAINED_BASE,
    FP32MasterMuon, FP32MasterAdamW, Muon,
)


def profile_training(sb_idx=0, n_steps=500, seq_len=128, batch_size=8):
    """Run training with per-component profiling."""
    print(f"\n{'='*70}")
    print(f"=== PROFILING TRAINING (n_steps={n_steps}) ===")
    print(f"{'='*70}")

    hp = {
        "groups": {"palettes": True, "lora": True, "correction": True, "layernorms": True},
        "lrs": {"palettes": 3e-4, "lora": 3e-4, "correction": 1e-3, "layernorms": 1e-4},
        "loss_type": "1-cos+norm_mse",
        "loss_weights": {"cos": 0.5, "mse": 0.5},
        "gradient_clip": 0.3,
        "eval_every": 999,
        "log_every": 50,
    }

    # Build student
    student, tokenizer = build_student_super_block(sb_idx, lora_rank=32, lora_alpha=64, stage=2)
    if student is None: return

    apply_groups(student, hp, sb_idx)

    # Load trained weights
    resume_dir = os.path.join(TRAINED_BASE, f"superblock_{sb_idx}_best")
    if os.path.isdir(resume_dir):
        load_state(student, resume_dir)

    # Build optimizers
    opt_muon, opt_adamw = build_optimizers(student, hp, sb_idx)

    # Cosine scheduler
    def lr_lambda(step):
        return 0.5 * (1.0 + math.cos(math.pi * step / max(10000, 1)))
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon.opt, lr_lambda) if opt_muon else None
    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw.opt, lr_lambda) if opt_adamw else None

    # Load teacher
    teacher, _ = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
    for layer in teacher.model.layers:
        for p in layer.parameters(): p.requires_grad_(False)
    student.model.embed_tokens = teacher.model.embed_tokens

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    student.train()

    # Prepare eval set (cached)
    eval_tokens = prepare_eval_set(tokenizer, device=DEVICE)

    # Data stream
    data_stream = stream_training_data(tokenizer, n_seqs=10**12, seq_len=seq_len, device=DEVICE, batch_size=batch_size)

    # Profile components
    timings = {
        "data_load": [],
        "teacher_fwd": [],
        "student_fwd": [],
        "loss": [],
        "backward": [],
        "clip": [],
        "muon_step": [],
        "adamw_step": [],
        "sched_step": [],
        "total_step": [],
    }

    global_step = 0
    start_time = time.time()
    last_log = 0

    print(f"\n=== Starting profiled training ===\n", flush=True)

    for batch_ids in data_stream:
        if global_step >= n_steps: break

        t_step_start = time.time()

        # Data load time (already loaded by generator, but measure next batch fetch)
        t_data = time.time()

        # Teacher forward
        t_teacher = time.time()
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
        torch.cuda.synchronize()
        timings["teacher_fwd"].append(time.time() - t_teacher)

        # Student forward
        t_student = time.time()
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
        torch.cuda.synchronize()
        timings["student_fwd"].append(time.time() - t_student)

        # Loss
        t_loss = time.time()
        loss, comps = compute_loss(student_out, h_out, hp)
        torch.cuda.synchronize()
        timings["loss"].append(time.time() - t_loss)

        if not torch.isfinite(loss):
            print(f"  [step {global_step}] NaN — skipping", flush=True)
            if opt_muon: opt_muon.zero_grad(set_to_none=True)
            if opt_adamw: opt_adamw.zero_grad(set_to_none=True)
            del batch_ids, h_out, student_out, loss, comps
            continue

        # Backward
        t_backward = time.time()
        loss.backward()
        torch.cuda.synchronize()
        timings["backward"].append(time.time() - t_backward)

        # Clip
        t_clip = time.time()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.grad is not None],
            hp.get("gradient_clip", 0.3))
        torch.cuda.synchronize()
        timings["clip"].append(time.time() - t_clip)

        # Muon step
        t_muon = time.time()
        if opt_muon: opt_muon.step()
        torch.cuda.synchronize()
        timings["muon_step"].append(time.time() - t_muon)

        # AdamW step
        t_adamw = time.time()
        if opt_adamw: opt_adamw.step()
        torch.cuda.synchronize()
        timings["adamw_step"].append(time.time() - t_adamw)

        # Scheduler step
        t_sched = time.time()
        if sched_muon: sched_muon.step()
        if sched_adamw: sched_adamw.step()
        if opt_muon: opt_muon.zero_grad(set_to_none=True)
        if opt_adamw: opt_adamw.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        timings["sched_step"].append(time.time() - t_sched)

        t_step_end = time.time()
        timings["total_step"].append(t_step_end - t_step_start)

        global_step += 1

        # Log every 50 steps
        if global_step - last_log >= 50:
            last_log = global_step
            # Compute averages for last 50 steps
            n = min(50, len(timings["total_step"]))
            avg = {k: sum(v[-n:]) / n * 1000 for k, v in timings.items() if v}  # ms
            total = avg.get("total_step", 0)
            tps = global_step / max(1e-6, time.time() - start_time)
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  step={global_step:4d} total={total:.0f}ms "
                  f"teacher={avg.get('teacher_fwd',0):.0f}ms "
                  f"student={avg.get('student_fwd',0):.0f}ms "
                  f"backward={avg.get('backward',0):.0f}ms "
                  f"muon={avg.get('muon_step',0):.0f}ms "
                  f"adamw={avg.get('adamw_step',0):.0f}ms "
                  f"tps={tps:.1f} "
                  f"GPU={gpu_mem:.1f}G/{gpu_reserved:.1f}G",
                  flush=True)

        del batch_ids, h_out, student_out, loss, comps

    # Final summary
    print(f"\n=== PROFILE SUMMARY ({n_steps} steps) ===")
    n = len(timings["total_step"])
    for k in ["teacher_fwd", "student_fwd", "loss", "backward", "clip", "muon_step", "adamw_step", "sched_step", "total_step"]:
        vals = timings[k]
        if not vals: continue
        avg_ms = sum(vals) / len(vals) * 1000
        first_50 = sum(vals[:50]) / min(50, len(vals)) * 1000
        last_50 = sum(vals[-50:]) / min(50, len(vals)) * 1000
        print(f"  {k:15s}: avg={avg_ms:.0f}ms  first50={first_50:.0f}ms  last50={last_50:.0f}ms  ratio={last_50/max(first_50,0.001):.1f}x")

    # Memory info
    print(f"\n=== MEMORY ===")
    print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"  GPU peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    profile_training(n_steps=args.n_steps, seq_len=args.seq_len, batch_size=args.batch_size)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
