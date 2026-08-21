#!/usr/bin/env python3
"""Profile WITHOUT per-component sync — measures true step time.
Tests 2 modes:
  A) Streaming dataset (like real training)
  B) Cached tokens (like the sweep)
Runs 200 steps each, logs step time every 50 steps.
"""
import os, sys, json, time, math, gc
sys.path.insert(0, "/root/qwen35_palettize/scripts")

import torch
import torch.nn as nn

from qwen_model import SUPER_BLOCKS, load_qwen_super_block_only
from train_qwen import (
    build_student_super_block, apply_groups, build_optimizers,
    compute_loss, stream_training_data, prepare_eval_set,
    load_state, DEVICE, DTYPE, PALETTIZED_BASE, TRAINED_BASE,
)


def run_profile(mode, n_steps=200, seq_len=128, batch_size=8):
    """Run training for n_steps, return list of step_times (ms)."""
    print(f"\n{'='*60}")
    print(f"=== MODE: {mode} ({n_steps} steps) ===")
    print(f"{'='*60}", flush=True)

    hp = {
        "groups": {"palettes": True, "lora": True, "correction": True, "layernorms": True},
        "lrs": {"palettes": 3e-4, "lora": 3e-4, "correction": 1e-3, "layernorms": 1e-4},
        "loss_type": "1-cos+norm_mse",
        "loss_weights": {"cos": 0.5, "mse": 0.5},
        "gradient_clip": 0.3,
    }

    student, tokenizer = build_student_super_block(0, lora_rank=32, lora_alpha=64, stage=2)
    if student is None: return None

    apply_groups(student, hp, 0)
    resume_dir = os.path.join(TRAINED_BASE, "superblock_0_best")
    if os.path.isdir(resume_dir):
        load_state(student, resume_dir)

    opt_muon, opt_adamw = build_optimizers(student, hp, 0)

    def lr_lambda(step):
        return 0.5 * (1.0 + math.cos(math.pi * step / max(10000, 1)))
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon.opt, lr_lambda) if opt_muon else None
    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw.opt, lr_lambda) if opt_adamw else None

    teacher, _ = load_qwen_super_block_only(0, device=DEVICE, dtype=DTYPE)
    for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
    for layer in teacher.model.layers:
        for p in layer.parameters(): p.requires_grad_(False)
    student.model.embed_tokens = teacher.model.embed_tokens

    sb_start, sb_end = SUPER_BLOCKS[0]
    student.train()

    # Data source
    if mode == "cached":
        # Use cached eval set as training data (deterministic, no streaming)
        cache_path = "/root/qwen35_palettize/eval_tokens.pt"
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False).to(DEVICE)
        else:
            cached = prepare_eval_set(tokenizer, device=DEVICE)
        n_cached = cached.shape[0]
        idx = 0
        def data_gen():
            nonlocal idx
            while True:
                if idx + batch_size > n_cached:
                    idx = 0
                batch = cached[idx:idx+batch_size]
                idx += batch_size
                yield batch
        data_stream = data_gen()
    else:
        data_stream = stream_training_data(tokenizer, n_seqs=10**12, seq_len=seq_len, device=DEVICE, batch_size=batch_size)

    step_times = []
    global_step = 0
    start_time = time.time()

    for batch_ids in data_stream:
        if global_step >= n_steps: break

        t0 = time.time()

        # Teacher forward
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                h = teacher.model.embed_tokens(batch_ids)
                position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
                pos_emb = None
                if hasattr(teacher.model, 'rotary_emb') and teacher.model.rotary_emb is not None:
                    pos_emb = teacher.model.rotary_emb(h, position_ids)
                for layer_idx in range(sb_end):
                    layer = teacher.model.layers[layer_idx]
                    out = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
                    h = out[0] if isinstance(out, tuple) else out
                h_out = h.detach()

        # Student forward
        s_h = student.model.embed_tokens(batch_ids)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
            s_pos_emb = None
            if hasattr(student.model, 'rotary_emb') and student.model.rotary_emb is not None:
                s_pos_emb = student.model.rotary_emb(s_h, position_ids)
            for layer_idx in range(sb_end + 1):
                layer = student.model.layers[layer_idx]
                out = layer(s_h, position_embeddings=s_pos_emb) if s_pos_emb is not None else layer(s_h)
                s_h = out[0] if isinstance(out, tuple) else out
            student_out = s_h

        loss, comps = compute_loss(student_out, h_out, hp)
        if not torch.isfinite(loss):
            del batch_ids, h_out, student_out, loss, comps
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.grad is not None], 0.3)
        if opt_muon: opt_muon.step()
        if opt_adamw: opt_adamw.step()
        if sched_muon: sched_muon.step()
        if sched_adamw: sched_adamw.step()
        if opt_muon: opt_muon.zero_grad(set_to_none=True)
        if opt_adamw: opt_adamw.zero_grad(set_to_none=True)

        # NO torch.cuda.synchronize() — let CUDA pipeline
        t1 = time.time()
        step_ms = (t1 - t0) * 1000
        step_times.append(step_ms)
        global_step += 1

        if global_step % 50 == 0:
            recent = step_times[-50:]
            avg = sum(recent) / len(recent)
            tps = global_step / max(1e-6, time.time() - start_time)
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(f"  step={global_step:4d} avg={avg:.0f}ms tps={tps:.1f} GPU={gpu_mem:.1f}G", flush=True)

        del batch_ids, h_out, student_out, loss, comps

    # Final sync to flush any pending CUDA work
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"\n  Total: {n_steps} steps in {total_time:.0f}s = {n_steps/total_time:.1f} tps")
    print(f"  First 50 avg: {sum(step_times[:50])/50:.0f}ms")
    print(f"  Last 50 avg:  {sum(step_times[-50:])/50:.0f}ms")
    print(f"  Slowdown ratio: {sum(step_times[-50:])/50 / max(sum(step_times[:50])/50, 1):.1f}x")

    # Cleanup
    del student, teacher, opt_muon, opt_adamw
    gc.collect()
    torch.cuda.empty_cache()
    return step_times


def main():
    print("=== INVESTIGATION: What's slow? ===\n")

    # Mode A: streaming (like real training)
    print("\n[1/2] STREAMING DATASET")
    streaming_times = run_profile("streaming", n_steps=200)

    # Clear everything between runs
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    # Mode B: cached tokens
    print("\n[2/2] CACHED TOKENS")
    cached_times = run_profile("cached", n_steps=200)

    # Comparison
    if streaming_times and cached_times:
        print(f"\n{'='*60}")
        print(f"COMPARISON")
        print(f"{'='*60}")
        s_first = sum(streaming_times[:50]) / 50
        s_last = sum(streaming_times[-50:]) / 50
        c_first = sum(cached_times[:50]) / 50
        c_last = sum(cached_times[-50:]) / 50
        print(f"  Streaming: first50={s_first:.0f}ms  last50={s_last:.0f}ms  ratio={s_last/s_first:.1f}x")
        print(f"  Cached:   first50={c_first:.0f}ms  last50={c_last:.0f}ms  ratio={c_last/c_first:.1f}x")
        print(f"  Streaming vs Cached (last50): {s_last/c_last:.1f}x")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
