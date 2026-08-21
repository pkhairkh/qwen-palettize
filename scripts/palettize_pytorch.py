#!/usr/bin/env python3
"""palettize_pytorch.py — Standalone implementation of palettization functions.

Replaces the Dolphin-era /root/auto_lut/palettize_pytorch.py module.
Provides the functions needed by palettize_core.py:
  - kmeans1d_weighted: 1D weighted k-means clustering
  - palettize_groups: per-group palettization using kmeans
  - reconstruct_Wq: reconstruct quantized weights from indices + LUT
  - sanitize_name: clean tensor names for filenames
  - write_lut_scalar: write LUT to binary file
  - write_metadata_json: write metadata.json
  - pack_indices_transposed: pack indices with given bitwidth
"""
import os, json, math
import numpy as np
import torch
import torch.nn.functional as F


def sanitize_name(name):
    """Convert a tensor name to a safe filename."""
    return name.replace(".", "_").replace("/", "_").replace("[", "").replace("]", "")


def kmeans1d_weighted(values, weights, k, max_iters=100):
    """1D weighted k-means clustering.

    Args:
        values: (N,) tensor — values to cluster
        weights: (N,) tensor — per-value weights (importance)
        k: int — number of clusters (2^bitwidth)
        max_iters: int — max iterations

    Returns:
        centers: (k,) tensor — cluster centers (sorted ascending)
        assignments: (N,) int64 tensor — cluster index per value
    """
    values = values.float().flatten()
    weights = weights.float().flatten()
    n = values.shape[0]

    if n == 0:
        return torch.zeros(k), torch.zeros(0, dtype=torch.long)

    # Initialize centers as quantiles
    sorted_vals, sort_idx = torch.sort(values)
    sorted_w = weights[sort_idx]
    cumw = torch.cumsum(sorted_w, dim=0)
    total_w = cumw[-1]
    quantiles = torch.linspace(0, 1, k + 2)[1:-1]  # k quantile points
    target_cumw = quantiles * total_w
    init_centers = torch.zeros(k)
    for i, tc in enumerate(target_cumw):
        idx = torch.searchsorted(cumw, tc)
        idx = min(idx, n - 1)
        init_centers[i] = sorted_vals[idx]

    centers = init_centers.clone()

    for _ in range(max_iters):
        # Assign each value to nearest center
        dists = torch.cdist(values.unsqueeze(1), centers.unsqueeze(1)).squeeze(1)  # (N,)
        assignments = torch.argmin(dists, dim=0)

        # Update centers as weighted mean
        new_centers = centers.clone()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                w = weights[mask]
                v = values[mask]
                new_centers[c] = (v * w).sum() / w.sum().clamp(min=1e-12)

        # Check convergence
        if torch.allclose(new_centers, centers, atol=1e-7):
            break
        centers = new_centers

    # Sort centers ascending
    sorted_centers, sort_order = torch.sort(centers)
    # Remap assignments to sorted order
    remap = torch.zeros(k, dtype=torch.long)
    for i, s in enumerate(sort_order):
        remap[s] = i
    assignments = remap[assignments]

    return sorted_centers, assignments


def palettize_groups(W, hess_diag, bitwidth, group_size):
    """Palettize W per group along axis 0 (out_dim).

    Args:
        W: (out_dim, in_dim) tensor
        hess_diag: (in_dim,) tensor — Hessian diagonal for weighting
        bitwidth: int (2 for 2-bit)
        group_size: int (256)

    Returns:
        indices: (out_dim, in_dim) int64 tensor — index per weight
        lut: (n_groups, 2^bitwidth) tensor — LUT entries per group
        n_groups: int
    """
    out_dim, in_dim = W.shape
    palette_size = 1 << bitwidth  # 4 for 2-bit
    n_groups = (out_dim + group_size - 1) // group_size

    indices = torch.zeros(out_dim, in_dim, dtype=torch.long, device=W.device)
    lut = torch.zeros(n_groups, palette_size, dtype=torch.float32, device=W.device)

    for g in range(n_groups):
        s = g * group_size
        e = min(s + group_size, out_dim)
        w_group = W[s:e, :]  # (gs, in_dim)
        gs = e - s

        # Flatten and cluster
        flat = w_group.reshape(-1).float()  # (gs * in_dim,)
        # Weight each value by its corresponding hess_diag (repeated per row)
        w_flat = hess_diag.unsqueeze(0).expand(gs, in_dim).reshape(-1).float()

        centers, assign = kmeans1d_weighted(flat, w_flat, palette_size)
        lut[g] = centers
        indices[s:e, :] = assign.reshape(gs, in_dim)

    return indices, lut, n_groups


def reconstruct_Wq(indices, lut, group_size):
    """Reconstruct quantized weights from indices + LUT.

    Args:
        indices: (out_dim, in_dim) int64
        lut: (n_groups, palette_size)
        group_size: int

    Returns:
        Wq: (out_dim, in_dim) — reconstructed weights
    """
    out_dim, in_dim = indices.shape
    n_groups = lut.shape[0]
    Wq = torch.zeros(out_dim, in_dim, dtype=lut.dtype, device=indices.device)
    for g in range(n_groups):
        s = g * group_size
        e = min(s + group_size, out_dim)
        Wq[s:e, :] = lut[g][indices[s:e, :]]
    return Wq


def write_lut_scalar(path, lut):
    """Write LUT to binary file as fp16 scalars.

    Args:
        path: output file path
        lut: (n_groups, palette_size) tensor
    """
    # Flatten and convert to float16
    flat = lut.reshape(-1).to(torch.float16).cpu().numpy()
    flat.tofile(path)


def write_metadata_json(path, tensor_metas, default_group_size=256):
    """Write metadata.json.

    Args:
        path: output file path
        tensor_metas: list of (name, meta_dict) or dict of name -> meta_dict
        default_group_size: int
    """
    if isinstance(tensor_metas, dict):
        tensor_metas = list(tensor_metas.items())

    meta = {
        "tensors": {},
        "default_group_size": default_group_size,
    }
    for name, tm in tensor_metas:
        meta["tensors"][name] = tm

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def pack_indices_transposed(indices, bitwidth):
    """Pack indices with given bitwidth, transposed.

    Args:
        indices: (out_dim, in_dim) tensor
        bitwidth: int (2, 3, 4, etc.)

    Returns:
        packed: bytes — packed indices
    """
    indices_t = indices.t().contiguous()
    flat = indices_t.flatten().to(torch.uint8).cpu().numpy()
    n = flat.size
    elements_per_byte = 8 // bitwidth
    out_size = (n + elements_per_byte - 1) // elements_per_byte
    out = np.zeros(out_size, dtype=np.uint8)
    mask = (1 << bitwidth) - 1

    for i in range(n):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * bitwidth
        out[byte_idx] |= (int(flat[i]) & mask) << bit_offset

    return out.tobytes()
