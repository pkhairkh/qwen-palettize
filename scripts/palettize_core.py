#!/usr/bin/env python3
"""palettize_core.py — 2-bit per-group palettization for Qwen3.5-4B.

NO PRE-TRANSPOSE. NO GPTQ.
Just weighted kmeans per group along out_dim (axis 0).

CoreML/ANE-compatible output format (identical to Dolphin's metadata schema).
"""
import os, sys, json, hashlib, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/auto_lut")
from palettize_pytorch import (
    kmeans1d_weighted,
    palettize_groups,
    reconstruct_Wq,
    sanitize_name,
    write_lut_scalar,
    write_metadata_json as _dolphin_write_metadata,
    pack_indices_transposed as _dolphin_pack_indices_transposed,
)

BITWIDTH = 2
GROUP_SIZE = 256
PALETTE_SIZE = 1 << BITWIDTH


# ─── 2-bit packing ─────────────────────────────────────────────────────
def pack_idx2(indices_t):
    flat = indices_t.flatten().to(torch.uint8).cpu().numpy()
    n = flat.size
    out_size = (n + 3) // 4
    out = np.zeros(out_size, dtype=np.uint8)
    for i in range(0, n, 4):
        chunk = flat[i:i+4]
        byte = 0
        for j, v in enumerate(chunk):
            byte |= (int(v) & 0x03) << (2 * j)
        out[i // 4] = byte
    return out.tobytes()


def pack_indices_transposed_2bit(indices):
    indices_t = indices.t().contiguous()
    return pack_idx2(indices_t)


_original_pack = _dolphin_pack_indices_transposed
def pack_indices_transposed_with_2bit(indices, bitwidth):
    if bitwidth == 2:
        return pack_indices_transposed_2bit(indices)
    return _original_pack(indices, bitwidth)

import palettize_pytorch
palettize_pytorch.pack_indices_transposed = pack_indices_transposed_with_2bit


# ─── Palettization ─────────────────────────────────────────────────────
def palettize_tensor_2bit(name, W_orig, X, out_dir, threshold=0.0, verbose=True):
    """Palettize W to 2-bit GS=256. No transpose, no GPTQ.

    Args:
        name: tensor name
        W_orig: (out_dim, in_dim) fp32 on CUDA
        X: (n_samples, in_dim) fp32 on CUDA — input activations
        out_dir: output directory
    Returns: metadata dict or None
    """
    out_dim, in_dim = W_orig.shape
    if out_dim < 1 or in_dim < 1:
        return None
    if out_dim % GROUP_SIZE != 0:
        if verbose:
            print(f"  [{name[:50]:<50s}] SKIP — out_dim {out_dim} not divisible by GS {GROUP_SIZE}", flush=True)
        return None

    san = sanitize_name(name)

    # Hessian for kmeans weighting
    with torch.no_grad():
        X_f = X.float()
        H = X_f.T @ X_f
        hess_diag = torch.diagonal(H).clone()

    if verbose:
        print(f"  [{name[:50]:<50s}] shape=({out_dim},{in_dim}) — 2-bit GS={GROUP_SIZE}…", flush=True)

    # kmeans only (NO GPTQ — tested: GPTQ hurts with kmeans LUT)
    W_comp = W_orig.clone()
    try:
        indices, lut, n_groups = palettize_groups(W_comp, hess_diag, BITWIDTH, GROUP_SIZE)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise

    # Reconstruct + cosine check
    Wq = reconstruct_Wq(indices, lut, GROUP_SIZE)
    with torch.no_grad():
        Y_orig = (X @ W_orig.T).flatten().float()
        Y_quant = (X @ Wq.T).flatten().float()
        cos = F.cosine_similarity(Y_orig.unsqueeze(0), Y_quant.unsqueeze(0), dim=1, eps=1e-8).item()

    if verbose:
        print(f"     2-bit GS={GROUP_SIZE}  cos={cos:.6f}", flush=True)

    # Write files
    idx_path = os.path.join(out_dir, f"{san}.idx2")
    lut_path = os.path.join(out_dir, f"{san}.lut_scalar")
    packed = pack_indices_transposed_2bit(indices)
    with open(idx_path, "wb") as f:
        f.write(packed)
    write_lut_scalar(lut_path, lut)

    sha_idx = hashlib.sha256(packed).hexdigest()
    with open(lut_path, "rb") as f:
        sha_lut = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "var": name,
        "dense_shape": [out_dim, in_dim],
        "indices_shape": [out_dim, in_dim],
        "bitwidth": BITWIDTH,
        "groups": n_groups,
        "group_axis": 1,
        "group_size": GROUP_SIZE,
        "nibble_order": "LSB_FIRST",
        "indices_layout": "D0D1",
        "consumer_transpose_y": False,
        "index_file": f"{san}.idx2",
        "lut_file": f"{san}.lut_scalar",
        "sha256_idx": sha_idx,
        "sha256_lut": sha_lut,
        "packed_len_bytes": len(packed),
        "idx_payload_offset_used": 0,
        "lut_payload_offset_used": 0,
        "_achieved_cos": cos,
    }
    if verbose:
        print(f"     → wrote {san}.idx2 ({len(packed)}B) + {san}.lut_scalar ({lut.numel()*2}B)  cos={cos:.6f}", flush=True)
    return meta


def write_metadata_json(path, tensor_metas):
    if isinstance(tensor_metas, dict):
        tensor_metas = list(tensor_metas.items())
    _dolphin_write_metadata(path, tensor_metas, default_group_size=GROUP_SIZE)
    with open(path, "r") as f:
        meta = json.load(f)
    meta["teacher_id"] = "qwen35-4b"
    meta["bitwidth"] = BITWIDTH
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def unpack_idx2(data, n):
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = (arr[i // 4] >> ((i % 4) * 2)) & 0x03
    return out


def load_indices(idx_path, out_dim, in_dim):
    with open(idx_path, "rb") as f:
        data = f.read()
    n = out_dim * in_dim
    flat = unpack_idx2(data, n)
    return torch.from_numpy(flat.reshape(in_dim, out_dim))


def load_lut(lut_path):
    with open(lut_path, "rb") as f:
        data = f.read()
    return torch.from_numpy(np.frombuffer(data, dtype=np.float16).astype(np.float32))
