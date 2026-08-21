#!/usr/bin/env python3
"""Dry-run: build student, print param counts + optimizer group breakdown, exit.
Does NOT start training. Verifies classify_param + apply_groups + build_optimizers.
"""
import os, sys, json
sys.path.insert(0, "/root/qwen35_palettize/scripts")

import torch
from train_qwen import (
    build_student_super_block, apply_groups, build_optimizers,
    load_hyperparams, write_default_hyperparams, classify_param,
    TRAINED_BASE,
)
from qwen_model import SUPER_BLOCKS

sb_idx = 0
print(f"=== Dry run: super-block {sb_idx} ===")
write_default_hyperparams()
hp = load_hyperparams()

student, tokenizer = build_student_super_block(sb_idx)
if student is None:
    print("FAILED to build student")
    sys.exit(1)

counts = apply_groups(student, hp, sb_idx)
print("\n=== Param counts (after fix) ===")
for g, c in counts.items():
    print(f"  {g}: {c:,}")

# Also show per-group param COUNT (not just numel)
group_param_count = {}
for name, p in student.named_parameters():
    grp = classify_param(name, sb_idx)
    if grp not in group_param_count:
        group_param_count[grp] = {"params": 0, "numel": 0, "trainable": 0}
    group_param_count[grp]["params"] += 1
    group_param_count[grp]["numel"] += p.numel()
    if p.requires_grad:
        group_param_count[grp]["trainable"] += 1

print("\n=== Per-group breakdown ===")
for grp in sorted(group_param_count.keys()):
    info = group_param_count[grp]
    print(f"  {grp:15s}: {info['params']:3d} tensors, {info['trainable']:3d} trainable, {info['numel']:>12,} params")

print("\n=== Building optimizers ===")
opt_muon, opt_adamw = build_optimizers(student, hp, sb_idx)

# Show a few example param names per group
print("\n=== Example param names per group ===")
seen = set()
for name, p in student.named_parameters():
    grp = classify_param(name, sb_idx)
    key = grp
    if key not in seen and p.requires_grad:
        seen.add(key)
        print(f"  [{grp:15s}] {name}  shape={list(p.shape)}  req_grad={p.requires_grad}")

print("\n=== Dry run complete (no training started) ===")
