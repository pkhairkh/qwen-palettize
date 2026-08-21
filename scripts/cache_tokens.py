#!/usr/bin/env python3
"""Pre-tokenize a small set of FineWeb-Edu samples and cache as .pt file.

This avoids the HF API rate-limiting that blocks `datasets.load_dataset(streaming=True)`
on every sweep cycle. Instead, each cycle just torch.load()s the cached .pt file.
"""
import os, sys, json, torch
sys.path.insert(0, "/root/qwen35_palettize/scripts")

from transformers import AutoTokenizer
from datasets import load_dataset

CACHE_PATH = "/root/qwen35_palettize/cached_tokens.pt"
N_SEQS = 512         # 512 sequences × 128 tokens = 65K tokens (plenty for sweep)
SEQ_LEN = 128

def main():
    if os.path.exists(CACHE_PATH):
        print(f"Cache already exists: {CACHE_PATH}")
        return

    print(f"Tokenizing {N_SEQS} sequences of length {SEQ_LEN} from FineWeb-Edu...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
    seqs = []
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 100: continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=SEQ_LEN, return_tensors="pt")["input_ids"].squeeze(0)
        if ids.numel() < 64: continue
        if ids.numel() < SEQ_LEN:
            pad = torch.full((SEQ_LEN - ids.numel(),), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:SEQ_LEN]
        seqs.append(ids)
        if len(seqs) % 50 == 0:
            print(f"  {len(seqs)}/{N_SEQS}", flush=True)
        if len(seqs) >= N_SEQS: break

    batch = torch.stack(seqs)  # (N_SEQS, SEQ_LEN)
    torch.save(batch, CACHE_PATH)
    print(f"Saved {batch.shape} → {CACHE_PATH} ({os.path.getsize(CACHE_PATH)} bytes)")

if __name__ == "__main__":
    main()
