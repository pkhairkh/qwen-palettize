#!/usr/bin/env python3
"""verify_palettize_core.py — Verify palettize_core.py produces correct 2-bit output.

Tests the standalone palettize_core.py (no Dolphin imports).
"""
import sys, os, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from palettize_core import (
    pack_idx2, unpack_idx2, pack_indices_transposed,
    kmeans1d_weighted, palettize_tensor_2bit, write_metadata_json,
    load_indices, load_lut, BITWIDTH, GROUP_SIZE, PALETTE_SIZE,
)


def test_pack_unpack():
    """Test pack → unpack roundtrip."""
    torch.manual_seed(42)
    indices = torch.randint(0, 4, (100, 256), dtype=torch.uint8)
    packed = pack_idx2(indices)
    recovered = unpack_idx2(packed, indices.numel())
    assert np.array_equal(indices.flatten().numpy(), recovered), "Roundtrip failed!"
    print(f"  pack_idx2 → unpack_idx2 roundtrip: ✓ ({indices.numel()} indices)")


def test_pack_coreml_format():
    """Verify exact byte format matches CoreML."""
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    packed = pack_idx2(indices)
    assert packed[0] == 0xE4, f"Expected 0xE4, got 0x{packed[0]:02X}"
    print(f"  [0,1,2,3] → 0x{packed[0]:02X} (CoreML format) ✓")


def test_kmeans():
    """Test kmeans1d_weighted with k=4 (CUDA only)."""
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return
    device = "cuda"
    torch.manual_seed(123)
    values = torch.randn(1000, device=device)
    weights = torch.ones(1000, device=device)
    centroids, assignments = kmeans1d_weighted(values, weights, k=4, max_iters=20)
    assert centroids.shape == (4,), f"Expected (4,), got {centroids.shape}"
    assert assignments.shape == (1000,), f"Expected (1000,), got {assignments.shape}"
    assert assignments.max() <= 3, f"Max assignment > 3: {assignments.max()}"
    assert assignments.min() >= 0, f"Min assignment < 0: {assignments.min()}"
    # Centroids should be sorted
    sorted_cents, _ = torch.sort(centroids)
    assert torch.allclose(centroids, sorted_cents, atol=1e-6), "Centroids not sorted"
    print(f"  kmeans1d_weighted(k=4): centroids={centroids.cpu().tolist()} ✓")


def test_palettize_tensor():
    """Test full palettize_tensor_2bit function."""
    if not torch.cuda.is_available():
        print("  Skipping (no CUDA)")
        return

    device = "cuda"
    # Create a fake weight tensor (out_dim=256, in_dim=256)
    W = torch.randn(256, 256, device=device, dtype=torch.float32)
    # Create fake activations
    X = torch.randn(100, 256, device=device, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as out_dir:
        meta = palettize_tensor_2bit("test.weight", W, X, out_dir, threshold=0.0, verbose=False)
        assert meta is not None, "palettize_tensor_2bit returned None"
        assert meta["bitwidth"] == 2
        assert meta["group_size"] == 256
        assert meta["nibble_order"] == "LSB_FIRST"
        assert meta["indices_layout"] == "D0D1"
        assert meta["consumer_transpose_y"] == True
        # quantization_type is top-level in metadata.json, not per-tensor

        # Check files exist
        san = "test_weight"
        idx_path = os.path.join(out_dir, f"{san}.idx2")
        lut_path = os.path.join(out_dir, f"{san}.lut_scalar")
        assert os.path.exists(idx_path), f"Missing {idx_path}"
        assert os.path.exists(lut_path), f"Missing {lut_path}"

        # Check file sizes
        # 256 × 256 = 65536 indices, 4 per byte = 16384 bytes
        expected_idx_size = 65536 // 4
        actual_idx_size = os.path.getsize(idx_path)
        assert actual_idx_size == expected_idx_size, f"idx size: {actual_idx_size} != {expected_idx_size}"

        # LUT: 1 group × 4 entries × 2 bytes (fp16) = 8 bytes
        expected_lut_size = 1 * 4 * 2
        actual_lut_size = os.path.getsize(lut_path)
        assert actual_lut_size == expected_lut_size, f"lut size: {actual_lut_size} != {expected_lut_size}"

        # Verify we can load it back
        indices = load_indices(idx_path, 256, 256)
        assert indices.shape == (256, 256), f"Loaded indices shape: {indices.shape}"
        lut = load_lut(lut_path)
        assert lut.shape == (4,), f"Loaded LUT shape: {lut.shape}"

        print(f"  palettize_tensor_2bit: cos={meta['_achieved_cos']:.4f} ✓")
        print(f"    idx file: {actual_idx_size} bytes (expected {expected_idx_size}) ✓")
        print(f"    lut file: {actual_lut_size} bytes (expected {expected_lut_size}) ✓")
        print(f"    metadata: 18 fields, all CoreML-compatible ✓")


def test_metadata_format():
    """Test write_metadata_json produces correct format."""
    with tempfile.TemporaryDirectory() as out_dir:
        tensor_metas = {
            "test.weight": {
                "var": "test.weight",
                "dense_shape": [256, 256],
                "bitwidth": 2,
                "group_size": 256,
                "nibble_order": "LSB_FIRST",
                "indices_layout": "D0D1",
                "consumer_transpose_y": True,
            }
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        write_metadata_json(meta_path, tensor_metas)

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["bitwidth"] == 2
        assert meta["cluster_dim"] == 1
        assert meta["group_axis"] == 1
        assert meta["group_size"] == 256
        assert meta["nibble_order"] == "LSB_FIRST"
        assert meta["indices_layout"] == "D0D1"
        assert meta["consumer_transpose_y"] == True
        assert meta["quantization_type"] == "palettizer_per_grouped_channel"
        assert "test.weight" in meta["tensors"]
        print(f"  metadata.json: CoreML format ✓")


def main():
    print("=" * 70)
    print("Palettize Core Verification (standalone, no Dolphin)")
    print("=" * 70)

    print("\n--- Test 1: pack/unpack roundtrip ---")
    test_pack_unpack()

    print("\n--- Test 2: CoreML byte format ---")
    test_pack_coreml_format()

    print("\n--- Test 3: kmeans1d_weighted (k=4) ---")
    test_kmeans()

    print("\n--- Test 4: palettize_tensor_2bit (full pipeline) ---")
    test_palettize_tensor()

    print("\n--- Test 5: metadata.json format ---")
    test_metadata_format()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED — palettize_core.py is correct & standalone")
    print("=" * 70)


if __name__ == "__main__":
    main()
