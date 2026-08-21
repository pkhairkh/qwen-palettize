#!/usr/bin/env python3
"""convert_trained_to_packed.py — Convert old .pt saves to .idx2 + .lut_scalar format.

Reads from trained/superblock_{sb_idx}_best/ (old .pt files)
Writes to trained/superblock_{sb_idx}_best_packed/ (new .idx2 + .lut_scalar + metadata.json)

This is a ONE-TIME migration script. After this, all saves use the packed format.
"""
import os, sys, json, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import pack_idx2, pack_indices_transposed_2bit, write_lut_scalar, sanitize_name, GROUP_SIZE, BITWIDTH, PALETTE_SIZE
from qwen_model import PalettizedLinear, QwenLoRA, SUPER_BLOCKS


def convert(sb_idx):
    src_dir = f"/root/qwen35_palettize/trained/superblock_{sb_idx}_best"
    dst_dir = f"/root/qwen35_palettize/trained/superblock_{sb_idx}_best_packed"
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"Source not found: {src_dir}")
        return

    # Load resume.json
    resume_path = os.path.join(src_dir, "_resume.json")
    if os.path.exists(resume_path):
        meta = json.load(open(resume_path))
        print(f"Resume: step={meta.get('step')}, cos={meta.get('cos')}")
    else:
        meta = {}

    tensor_metas = {}

    # Find all .pt files
    pt_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".pt")])
    print(f"Found {len(pt_files)} .pt files to convert")

    for pt_file in pt_files:
        name = pt_file[:-3]  # remove .pt
        # Convert underscores back to dots for the tensor name
        tensor_name = name.replace("_", ".")

        # Load the tensor
        t = torch.load(os.path.join(src_dir, pt_file), map_location="cpu", weights_only=True)

        if "palette" in name:
            # Palette: (n_groups, 4) bf16 → save as .lut_scalar (same as palettized format)
            san = sanitize_name(tensor_name)
            lut_path = os.path.join(dst_dir, f"{san}.lut_scalar")
            write_lut_scalar(lut_path, t)
            print(f"  {san}.lut_scalar — shape={list(t.shape)} dtype={t.dtype}")
            tensor_metas[tensor_name] = {
                "type": "palette",
                "shape": list(t.shape),
                "file": f"{san}.lut_scalar",
            }

        elif "index_logits" in name:
            # index_logits: (4, K, N) → argmax → (K, N) int8 → pack as .idx2
            indices = t.argmax(dim=0).to(torch.uint8)  # (K, N) int8
            san = sanitize_name(tensor_name.replace("index_logits", "weight"))
            idx_path = os.path.join(dst_dir, f"{san}.idx2")
            packed = pack_idx2(indices)
            with open(idx_path, "wb") as f:
                f.write(packed)
            print(f"  {san}.idx2 — indices from argmax, shape={list(indices.shape)} ({len(packed)} bytes)")
            tensor_metas[tensor_name] = {
                "type": "indices",
                "shape": list(indices.shape),
                "file": f"{san}.idx2",
            }

        elif "lora_A" in name or "lora_B" in name:
            # LoRA: (in_dim, rank) or (out_dim, rank) bf16 → save raw .pt for now
            # (LoRA will be merged into weights at merge time, not packed)
            torch.save(t, os.path.join(dst_dir, pt_file))
            print(f"  {pt_file} — LoRA, shape={list(t.shape)} dtype={t.dtype}")
            tensor_metas[tensor_name] = {
                "type": "lora",
                "shape": list(t.shape),
                "file": pt_file,
            }

        else:
            # Everything else (layernorms, SSM, etc.) — small, save as .pt
            torch.save(t, os.path.join(dst_dir, pt_file))
            print(f"  {pt_file} — other, shape={list(t.shape)} dtype={t.dtype}")
            tensor_metas[tensor_name] = {
                "type": "other",
                "shape": list(t.shape),
                "file": pt_file,
            }

    # Write metadata.json
    meta_path = os.path.join(dst_dir, "metadata.json")
    meta["tensors"] = tensor_metas
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote metadata.json with {len(tensor_metas)} entries")

    # Copy resume.json
    if meta:
        with open(os.path.join(dst_dir, "_resume.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Show size comparison
    import subprocess
    src_size = subprocess.check_output(["du", "-sh", src_dir]).decode().split()[0]
    dst_size = subprocess.check_output(["du", "-sh", dst_dir]).decode().split()[0]
    print(f"\nSize: {src_dir} = {src_size} → {dst_dir} = {dst_size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_idx", type=int, default=0)
    args = ap.parse_args()
    convert(args.sb_idx)
