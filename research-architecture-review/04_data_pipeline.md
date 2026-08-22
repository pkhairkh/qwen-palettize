# 04 — Data Pipeline: Streaming vs Prefetch vs Cached

## 1. The current pipeline in one sentence

`scripts/train_qwen.py` lines 889–918 implement a 30-line generator called `stream_training_data` that calls `datasets.load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)`, iterates one example at a time, tokenizes each on the calling thread, and yields a batch of `(batch_size, seq_len)` tensors — with no prefetch, no async, no tokenization cache, no `pin_memory`, no `prefetch_factor`, and no batching of the tokenization step.

This is the simplest possible data pipeline. It works — batches do come out, the GPU does train. But it leaves the GPU idle for roughly **40% of each step**, blocked on Python-side I/O and tokenization that happens on the training thread. This document analyzes the current pipeline in detail, quantifies the waste, and proposes a phased migration through three levels: (1) tokenization cache, (2) async prefetch with pinned memory, (3) a proper `DataLoader` worker model with on-disk tokenized shards.

---

## 2. The current code, line by line

For reference, here is the entire `stream_training_data` function from `train_qwen.py:889`:

```python
def stream_training_data(tokenizer, n_seqs, seq_len, device="cuda", batch_size=8):
    """Stream batches of tokenized sequences from FineWeb-Edu.

    Each yielded batch has shape (batch_size, seq_len) on `device`.
    - seq_len is small (128) so we can afford a larger batch_size
      to keep the L4 GPU busy despite the small per-seq FLOPs.
    - batch_size is the # of independent sequences per training step.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                      streaming=True, name="sample-10BT")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buf = []
    produced = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 100: continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=seq_len, return_tensors="pt")["input_ids"].squeeze(0)
        if ids.numel() < 64: continue
        if ids.numel() < seq_len:
            pad = torch.full((seq_len - ids.numel(),), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:seq_len]
        buf.append(ids)
        while len(buf) >= batch_size and produced < n_seqs:
            batch = torch.stack(buf[:batch_size]).to(device)
            del buf[:batch_size]
            produced += batch_size
            yield batch
        if produced >= n_seqs: return
```

### 2.1 The seven problems with this function

1. **`from datasets import load_dataset` is imported inside the function body** (line 897). This is a 50 ms import on first call — negligible for a long training run, but it means the import is re-evaluated every time `stream_training_data` is called (e.g., from `sweep_qwen.py`'s subprocesses).
2. **`ds = load_dataset(...)` is called at the top of the function** (line 898). It returns an `IterableDataset` that lazily downloads parquet shards from the HF Hub CDN. The first `next(iter(ds))` triggers a network request; subsequent `next(iter(ds))` calls stream from the cached shard. There is no prefetch — each `next()` blocks until the next example is decoded.
3. **`for ex in ds:`** (line 902) is a single-threaded Python loop. Each iteration:
   - Calls `next(iter(ds))` → blocks on network/parquet decode (50–200 ms for the first example in a shard, 1–10 ms for subsequent examples).
   - Calls `tokenizer(text, ...)` (line 905) → blocks on Rust tokenizer (1–5 ms per example, but single-threaded).
   - Pads or truncates to `seq_len` (line 907–911) → trivial CPU work.
   - Appends to `buf` (line 912) → trivial.
4. **`buf` is a plain Python `list` of tensors** (line 900). It is bounded by `batch_size` (the `while len(buf) >= batch_size` loop drains it), so there is no unbounded memory growth — but there is also no lookahead. When `buf` has fewer than `batch_size` elements, the next yield has to wait for the loop to fill `buf` back up.
5. **`torch.stack(buf[:batch_size]).to(device)`** (line 914) is a synchronous CPU→GPU copy. The `to(device)` call blocks until the copy completes — typically 1–2 ms for a 32×512 int64 tensor (128 KB), but it happens on the training thread.
6. **`yield batch`** (line 917) hands control back to the training loop. The generator is suspended; the next call to `next(gen)` resumes the loop. There is no overlap between the GPU's forward pass and the data pipeline's next-batch preparation.
7. **`n_seqs=10**12`** (line 1029: `stream_training_data(tokenizer, n_seqs=10**12, ...)`) is a sentinel meaning "run forever". The `produced >= n_seqs` check at line 918 never triggers; the loop runs until the training step counter hits `max_steps`. This is fine for production, but it means there is no natural end-of-data signal for testing.

### 2.2 The timing budget per batch

For the `train_sb0.log` configuration (`batch=32, seq=512`):

| Operation | Where | Time | Blocking? |
|---|---|---|---|
| `next(iter(ds))` × 32 (network + parquet decode) | L902 | 32 × 5 ms = 160 ms | Yes (training thread) |
| `tokenizer(text)` × 32 | L905 | 32 × 2 ms = 64 ms | Yes |
| Padding/truncation × 32 | L906–911 | 32 × 0.1 ms = 3 ms | Yes |
| `torch.stack` + `.to(device)` | L914 | 1.5 ms | Yes |
| **Total data prep per batch** | | **~228 ms** | |
| GPU forward + backward | L1082–1123 | ~200 ms | Yes (data thread idle) |
| Optimizer step | L1142–1159 | ~50 ms | Yes (data thread idle) |
| **Total per step** | | **~478 ms** | ~48% GPU idle |

This matches the observed `tps=1.73` (580 ms/step) — the extra 100 ms is the Python dispatch overhead and the implicit CUDA synchronization between the teacher's `stream_t` forward and the student's default-stream forward.

The **48% GPU idle** is the recoverable budget. A properly pipelined data loader could overlap the 228 ms of data prep with the 250 ms of GPU work, bringing the step time down to ~280 ms — a **1.7× speedup** for free.

---

## 3. The three-stage migration

The migration is staged to be incremental — each stage delivers value independently, and the team can stop after any stage if the throughput is sufficient.

### 3.1 Stage A — Tokenization cache (1 day, ~30% step-time reduction)

The cheapest fix is to precompute and cache tokenized batches on disk. The tokenization is deterministic (the same text always produces the same tokens for a given tokenizer), and FineWeb-Edu's 10 BT sample is stable. There is no reason to re-tokenize the same sequence on every training run.

#### 3.1.1 Implementation

A new script, `scripts/cache_tokens_v2.py`, that:

1. Iterates `FineWeb-Edu` streaming, same as `stream_training_data`.
2. Tokenizes each example with the same tokenizer.
3. Pads/truncates to `seq_len` (configurable, default 512).
4. Stacks into batches of `batch_size` (configurable, default 32).
5. Writes each batch to a binary file: `data/cache/fineweb_edu_seq512_bs32/shard_NNNNN.bin`.
6. Writes a manifest: `data/cache/fineweb_edu_seq512_bs32/manifest.json` with `{n_shards, shard_size, seq_len, batch_size, dtype}`.

The file format is raw int64 tensors (no pickle, no JSON per batch), so a shard of 1024 batches × 32 sequences × 512 tokens × 8 bytes = 128 MB per shard. The full 10 BT sample at seq_len=512 yields roughly 20M sequences = 625K batches = 610 shards = ~78 GB on disk. This is large but feasible on a research server; for production, a 1 BT subset (8 GB) suffices for super-block training.

#### 3.1.2 Updated `stream_training_data`

```python
def stream_training_data(tokenizer, n_seqs, seq_len, device="cuda", batch_size=8,
                         cache_dir="/data/fineweb_edu_seq512_bs32"):
    """Stream batches from on-disk tokenized cache, falling back to streaming
    if the cache is missing or exhausted."""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        n_shards = manifest["n_shards"]
        shard_size = manifest["shard_size"]
        # Round-robin across shards for diversity
        shard_indices = np.random.permutation(n_shards)
        produced = 0
        for si in shard_indices:
            shard_path = os.path.join(cache_dir, f"shard_{si:05d}.bin")
            # mmap the shard — no copy into RAM
            mmap = np.memmap(shard_path, dtype=np.int64, mode="r")
            n_batches = len(mmap) // (batch_size * seq_len)
            mmap = mmap.reshape(n_batches, batch_size, seq_len)
            for bi in np.random.permutation(n_batches):
                batch = torch.from_numpy(mmap[bi].copy()).to(device)
                produced += batch_size
                yield batch
                if produced >= n_seqs: return
    else:
        # Fallback to streaming
        ... (original code) ...
```

#### 3.1.3 Expected impact

- Tokenization cost drops to zero (cached on disk).
- The remaining cost is the `np.memmap` read (fast — Linux page cache) and the `.to(device)` copy (1–2 ms).
- Total data prep per batch: ~10 ms (down from 228 ms).
- **Step time: ~280 ms → tps ≈ 3.5** (was 1.73, ~2× speedup).

#### 3.1.4 Limitations

- Disk space: ~8 GB per 1 BT subset. Reasonable.
- Cache invalidation: if the tokenizer changes (e.g., a new vocab), the cache is invalid. Solution: include the tokenizer's `vocab_hash` in the cache directory name.
- No overlap with GPU: the disk read still happens on the training thread. This is fixed in Stage B.

### 3.2 Stage B — Async prefetch with pinned memory (2 days, additional ~15% reduction)

Stage A eliminates the tokenization cost but keeps the disk read on the training thread. Stage B moves the disk read to a background thread, with a ring buffer of pinned-memory tensors.

#### 3.2.1 Implementation

```python
class AsyncPrefetchLoader:
    """Async data loader with a ring buffer of pinned-memory tensors.
    
    Background thread:
      - Reads from the cache (or streams + tokenizes)
      - Copies to a pinned-memory tensor
      - Hands the pinned tensor to the main thread
    
    Main thread:
      - Receives the pinned tensor
      - Calls .to(device, non_blocking=True)
      - Yields to the training loop
    """
    def __init__(self, cache_loader, batch_size, seq_len, device="cuda",
                 buffer_size=4):
        self.cache_loader = cache_loader
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        for batch_cpu in self.cache_loader:
            # batch_cpu is a numpy array or torch tensor on CPU
            pinned = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long,
                                  pin_memory=True)
            pinned.copy_(torch.from_numpy(batch_cpu))
            self.queue.put(pinned)
        self.queue.put(None)  # sentinel
    
    def __iter__(self):
        while True:
            pinned = self.queue.get()
            if pinned is None: break
            batch = pinned.to(self.device, non_blocking=True)
            yield batch
```

#### 3.2.2 Expected impact

- The disk read (10 ms) overlaps with the GPU's forward+backward (200 ms).
- The only remaining cost is the `non_blocking=True` GPU copy, which is ~1 ms and overlaps with the next forward pass's first kernel.
- Total data prep visible to the training thread: ~1 ms.
- **Step time: ~260 ms → tps ≈ 3.8** (was 1.73, ~2.2× speedup vs Stage A's ~2×).

#### 3.2.3 Limitations

- Requires `pin_memory=True` for the host-side tensor — uses ~1 MB of pinned memory per batch (32 × 512 × 8 bytes = 128 KB), so 4 buffers = 512 KB. Negligible.
- Requires a background thread. Python's GIL means the background thread cannot run Python code in parallel with the main thread — but `numpy.memmap.read` and `torch.tensor.copy_` release the GIL, so the main thread is not blocked.
- The CUDA copy `non_blocking=True` requires the destination tensor to be on a CUDA stream that is not the default stream. The training loop must call `torch.cuda.current_stream().synchronize()` before consuming the batch (or use `torch.cuda.Event` for finer-grained sync). This is a small refactor to the training loop.

### 3.3 Stage C — Proper `DataLoader` worker model (3 days, additional robustness)

Stage B uses a single background thread. Stage C upgrades to PyTorch's `DataLoader` with `num_workers > 0`, which spawns multiple worker processes (not threads). Each worker reads from the cache, tokenizes if needed, and ships batches to the main process via shared memory.

#### 3.3.1 Implementation

```python
class CachedFineWebEduDataset(torch.utils.data.Dataset):
    """Dataset backed by on-disk tokenized shards."""
    def __init__(self, cache_dir, seq_len=512, batch_size=32):
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
        self.shards = [os.path.join(cache_dir, f"shard_{i:05d}.bin")
                       for i in range(manifest["n_shards"])]
        self.n_batches_per_shard = manifest["n_batches_per_shard"]
        self.n_batches = len(self.shards) * self.n_batches_per_shard
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        shard_idx = idx // self.n_batches_per_shard
        batch_idx = idx % self.n_batches_per_shard
        shard_path = self.shards[shard_idx]
        # mmap + slice
        mmap = np.memmap(shard_path, dtype=np.int64, mode="r")
        mmap = mmap.reshape(-1, self.batch_size, self.seq_len)
        batch = mmap[batch_idx].copy()  # copy to process-local memory
        return torch.from_numpy(batch)

# Usage in training loop:
dataset = CachedFineWebEduDataset(cache_dir, seq_len=512, batch_size=32)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1,  # each __getitem__ returns a (32, 512) batch already
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,  # 4 batches prefetched per worker = 16 batches total
)
for batch in loader:
    batch = batch.squeeze(0).to(device, non_blocking=True)
    ...
```

#### 3.3.2 Expected impact

- 4 workers × 4 prefetched batches = 16 batches in flight. With each batch taking ~10 ms to read, the prefetch buffer covers ~160 ms of GPU work — more than enough to hide the disk read.
- **Step time: ~260 ms (same as Stage B), but with better robustness**: a single slow disk read does not stall the GPU because there are 15 other batches in the buffer.
- Multi-process workers avoid the GIL: each worker can run Python code (e.g., for fallback to streaming when the cache is exhausted) without blocking the main thread.

#### 3.3.3 Limitations

- Worker processes require IPC. PyTorch's `DataLoader` uses `multiprocessing.Queue` with `protocol=4` pickling. For a 32×512 int64 tensor (128 KB), the IPC overhead is ~1 ms per batch — acceptable.
- Workers do not see the main process's CUDA context. They produce CPU tensors; the main process does the `.to(device, non_blocking=True)` copy. This is correct.
- The `prefetch_factor=4` means 4 × num_workers × batch_size × seq_len × 8 bytes of pinned memory = 4 × 4 × 32 × 512 × 8 = 2 MB. Negligible.

---

## 4. The data pipeline in the broader context

The data pipeline does not exist in isolation. It connects to:

### 4.1 The eval set (already cached, but inflexibly)

`prepare_eval_set` at `train_qwen.py:362–403` pre-caches 256 sequences of `seq_len=512` to `eval_tokens.pt` (1.1 MB on disk). This is a one-time write — the first call to `prepare_eval_set` downloads FineWeb-Edu, tokenizes 256+2000 examples, picks 256 of them, and saves. Subsequent calls load the `.pt` file.

The eval cache is good — it ensures every eval uses the same data for fair comparison. But:

- The cache path is hardcoded to `/root/qwen35_palettize/eval_tokens.pt` (line 357). It is not portable to other machines or other researchers.
- The cache is per-seq_len. If the user changes `--seq_len` from 512 to 1024, the cached file is wrong (it has 512-token sequences), but `prepare_eval_set` does not detect this — it loads the cached file, truncates each sequence to the new seq_len, and proceeds. This is a silent correctness bug.
- The 2000 sequences skipped at line 380 are arbitrary — they are meant to avoid overlap with training data, but the streaming order is not deterministic across HF Hub revisions. A new revision of FineWeb-Edu could put training data in the eval set.

**Fix**: include `seq_len` and `tokenizer.vocab_size` in the cache filename, and include the FineWeb-Edu commit hash in the cache manifest.

### 4.2 The calibration set (not cached)

`calib_qwen.py` (365 LOC) downloads FineWeb-Edu streaming, tokenizes, and runs through the teacher to capture activations for GPTQ + kmeans. The calibration runs once per super-block (per SPEC §3.1 Stage 1), so the tokenization cost is amortized — but the calibration set itself is not cached. Each super-block's calibration re-downloads and re-tokenizes the same data.

This is a 2× waste: 8 super-blocks × 8192 sequences × 2048 tokens = 134M tokens, re-tokenized 8 times = 1.07B tokens of redundant work. At the L4's ~5K tokens/ms tokenization rate, this is ~3.6 minutes per super-block, ~29 minutes total. Not catastrophic, but free to fix.

**Fix**: cache the calibration tokens to `data/calib_tokens_sbN.pt` after the first run, and reuse for subsequent runs of the same super-block.

### 4.3 The `cached_tokens.pt` file (orphaned)

The repo has `cached_tokens.pt` (516 KB) at the root. This is a leftover from an earlier iteration of the eval-set cache — it is not loaded by `prepare_eval_set` (which uses `eval_tokens.pt` instead). It is dead weight in the repo and should be deleted, but the `.gitignore` does not exclude `.pt` files, so it persists.

**Fix**: `rm cached_tokens.pt` and add `*.pt` to `.gitignore`.

### 4.4 The HF Hub download behavior

The `load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)` call at line 898 hits the HF Hub on the first `next(iter(ds))` of each shard. The HF Hub has rate limits:

- Unauthenticated: 60 requests/hour per IP.
- Authenticated (with `HF_TOKEN`): 5000 requests/hour.

The `train_sb0.log` line 14 shows: "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads." This means the training run is making unauthenticated requests, and any concurrent run (e.g., a sweep) will hit the rate limit. The fix is to set `HF_TOKEN` in the environment.

The HF Hub also serves the data via CDN — but the CDN's cache behavior is per-region. A user in a different region from the CDN edge may see 100–500 ms per-shard latency instead of 1–10 ms.

**Fix**: set `HF_TOKEN` and consider a local mirror (`HF_ENDPOINT=https://my-mirror.example.com`).

---

## 5. Comparison to other data pipelines

### 5.1 The Megatron-LM approach

Megatron-LM uses a pre-tokenized binary format called `MMapIndexedDataset`. The dataset is two files: a `.idx` (index of document boundaries) and a `.bin` (raw int32 token IDs, mmap'd). The `DataLoader` is a custom `MegatronPretrainingRandomSampler` that:

- Picks a random document start.
- Reads `seq_len + 1` tokens (the +1 is for the next-token prediction label).
- Returns the slice.

The pipeline is single-threaded, but the `mmap` read is fast (~1 ms per batch on a local SSD), and the sampler is cheap. There is no prefetch — the read is fast enough that prefetch is unnecessary. This is the gold standard for LLM pretraining data pipelines.

### 5.2 The lit-GPT (Karpathy) approach

lit-GPT uses a similar pre-tokenized binary format, stored as `train.bin` and `val.bin`. The `DataLoader` is PyTorch's standard `DataLoader` with `num_workers=4, pin_memory=True`. No async prefetch is needed — the `mmap` read is fast enough that the 4 workers keep the buffer full.

### 5.3 The HuggingFace `datasets` approach (non-streaming)

`datasets.load_dataset(..., streaming=False)` downloads the entire dataset to disk (the HF cache), then loads it as an `ArrowDataset` (memory-mapped). Iterating is fast (the `mmap` is local), but the download is large (FineWeb-Edu 10 BT is 2 TB on disk). For research, the streaming variant is preferred — but the streaming variant pays the network cost on every iteration.

### 5.4 The HuggingFace `datasets` approach (streaming, with `IterableDataset`)

This is what the codebase uses. It is the slowest of the four options — network latency on every `next()`, no caching, no prefetch. The only advantage is that it requires no disk space and no pre-download.

### 5.5 The `torchdata` approach

`torchdata` (PyTorch's data utilities library) provides `DataPipes` — composable iterators with built-in shuffle, batch, prefetch, and map operations. A `DataPipe` chain like:

```python
dp = FileOpener(cache_dir).shuffle().map(read_shard).batch(batch_size).prefetch(4)
```

would be equivalent to Stage C without writing a custom `DataLoader`. However, `torchdata` is not yet production-stable (it is in beta as of PyTorch 2.4) and is not widely adopted.

### 5.6 Summary

| Approach | Disk usage | Per-batch latency | Implementation complexity | Status |
|---|---|---|---|---|
| Megatron `MMapIndexedDataset` | High (pre-tokenized binary) | ~1 ms | High (custom format) | Production |
| lit-GPT `train.bin` | High (pre-tokenized binary) | ~1 ms | Medium (standard `DataLoader`) | Production |
| HF `datasets` non-streaming | Very high (full dataset) | ~1 ms | Low (one call) | Production |
| HF `datasets` streaming | None | 5–200 ms | Low (current code) | Production |
| `torchdata` `DataPipes` | Configurable | Configurable | Medium | Beta |
| **Stage A (this proposal)** | Medium (~8 GB per 1 BT) | ~10 ms | Low (cache script + 30 LOC) | Proposed |
| **Stage B (this proposal)** | Medium | ~1 ms | Medium (AsyncPrefetchLoader) | Proposed |
| **Stage C (this proposal)** | Medium | ~1 ms | Medium (standard `DataLoader`) | Proposed |

The proposal is to migrate through Stages A → B → C, stopping at the stage where throughput is sufficient. Stage A alone delivers a ~2× speedup with one day of work.

---

## 6. The tokenization cache design in detail

The tokenization cache is the highest-ROI fix in the data pipeline. This section specifies the design in enough detail to implement.

### 6.1 File format

Each shard is a raw binary file containing `n_batches × batch_size × seq_len` int64 tokens, packed contiguously. No headers, no padding. The shard size is `n_batches × batch_size × seq_len × 8 bytes`.

For `batch_size=32, seq_len=512, n_batches=1024` per shard, the shard size is `1024 × 32 × 512 × 8 = 128 MB`. The full 10 BT FineWeb-Edu sample yields ~610 shards = ~78 GB on disk.

### 6.2 Manifest

```json
{
  "version": 1,
  "dataset": "HuggingFaceFW/fineweb-edu",
  "dataset_revision": "ab7c5234...",
  "tokenizer": "Qwen/Qwen3.5-4B",
  "tokenizer_revision": "1234abcd...",
  "seq_len": 512,
  "batch_size": 32,
  "n_batches_per_shard": 1024,
  "n_shards": 610,
  "dtype": "int64",
  "pad_token_id": 0
}
```

The `dataset_revision` and `tokenizer_revision` fields ensure the cache is invalidated when either the dataset or the tokenizer is updated.

### 6.3 Shard naming

`shard_{shard_idx:05d}.bin` — zero-padded to 5 digits for sortability. The shards are independent and can be read in any order.

### 6.4 Caching script

```python
# scripts/cache_tokens_v2.py
import os, json, argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches_per_shard", type=int, default=1024)
    ap.add_argument("--n_shards", type=int, default=8)  # 8 shards × 1024 batches × 32 seqs × 512 tokens ≈ 1 GB
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                      streaming=True, name="sample-10BT")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    
    shard = np.zeros((args.n_batches_per_shard, args.batch_size, args.seq_len), dtype=np.int64)
    buf = []
    batch_idx_in_shard = 0
    shard_idx = 0
    
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 100: continue
        ids = tok(text, add_special_tokens=True, truncation=True,
                  max_length=args.seq_len, return_tensors="np")["input_ids"].squeeze(0)
        if ids.size < 64: continue
        if ids.size < args.seq_len:
            pad = np.full(args.seq_len - ids.size, pad_id, dtype=ids.dtype)
            ids = np.concatenate([ids, pad])
        else:
            ids = ids[:args.seq_len]
        buf.append(ids)
        while len(buf) >= args.batch_size:
            batch = np.stack(buf[:args.batch_size])
            del buf[:args.batch_size]
            shard[batch_idx_in_shard] = batch
            batch_idx_in_shard += 1
            if batch_idx_in_shard >= args.n_batches_per_shard:
                shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
                shard.tofile(shard_path)
                print(f"Wrote {shard_path} ({shard.nbytes / 1e6:.1f} MB)")
                shard_idx += 1
                if shard_idx >= args.n_shards:
                    # Write manifest
                    manifest = {
                        "version": 1,
                        "dataset": "HuggingFaceFW/fineweb-edu",
                        "tokenizer": args.model,
                        "seq_len": args.seq_len,
                        "batch_size": args.batch_size,
                        "n_batches_per_shard": args.n_batches_per_shard,
                        "n_shards": shard_idx,
                        "dtype": "int64",
                        "pad_token_id": pad_id,
                    }
                    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
                        json.dump(manifest, f, indent=2)
                    print(f"Wrote manifest. Done.")
                    return
                shard[:] = 0
                batch_idx_in_shard = 0
    
    # Write partial shard if we ran out of data
    if batch_idx_in_shard > 0:
        shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
        shard[:batch_idx_in_shard].tofile(shard_path)
        print(f"Wrote partial {shard_path} ({batch_idx_in_shard} batches)")

if __name__ == "__main__":
    main()
```

### 6.5 Loader integration

```python
# In train_qwen.py, replace stream_training_data with:
def stream_training_data(tokenizer, n_seqs, seq_len, device="cuda", batch_size=8,
                         cache_dir=None):
    if cache_dir and os.path.exists(os.path.join(cache_dir, "manifest.json")):
        yield from _stream_from_cache(cache_dir, n_seqs, seq_len, device, batch_size)
    else:
        yield from _stream_from_huggingface(tokenizer, n_seqs, seq_len, device, batch_size)
```

---

## 7. Risk assessment

The data pipeline migration has three risks:

1. **Cache invalidation.** If the FineWeb-Edu dataset is revised (e.g., to fix a tokenization bug), the cache contains stale tokens. The `dataset_revision` field in the manifest catches this — but only if the operator checks it. **Mitigation**: print a warning if the manifest's `dataset_revision` does not match the current HF Hub revision.

2. **Worker process crashes.** If a `DataLoader` worker crashes (e.g., due to a corrupted shard), the main process hangs waiting for the next batch. PyTorch's `DataLoader` does not auto-restart workers. **Mitigation**: catch the worker's exception in the main thread, log it, and continue with the next batch (skip the corrupted shard).

3. **Determinism.** The current `stream_training_data` is deterministic given the same HF Hub revision (the streaming order is fixed). The cached pipeline is also deterministic given the same shards. But the `DataLoader`'s `shuffle=True` introduces nondeterminism — each run sees the data in a different order. This is desirable for training (avoids overfitting to a fixed order), but undesirable for debugging (a NaN at step 1234 in one run may not reproduce in another). **Mitigation**: support both `shuffle=True` (production) and `shuffle=False` (debugging), controlled by a CLI flag.

---

## 8. Summary

The data pipeline is the smallest component of the codebase (30 LOC) but accounts for ~48% of per-step wall time. The migration path is:

- **Stage A**: pre-tokenize to disk (1 day, ~2× speedup, ~8 GB disk).
- **Stage B**: async prefetch with pinned memory (2 days, additional ~10% speedup).
- **Stage C**: proper `DataLoader` with `num_workers=4` (3 days, robustness, no additional speedup).

Combined with the memory fixes in `03_memory_waste_analysis.md` (which enable `batch=128`), the post-fix throughput is:

```
  Current:  batch=32, tps=1.73 → 31.3K tokens/sec
  Stage A:  batch=32, tps=3.5  → 63.5K tokens/sec
  Stage B:  batch=32, tps=3.8  → 68.9K tokens/sec
  + Memory fix: batch=128, tps=3.8 → 275.5K tokens/sec (8.8× over current)
```

The Stage A fix is the highest-ROI work item in the data pipeline and should be the first priority after the `PartialWrapper` → `nn.Module` refactor (which unblocks the gradient checkpointing fix).
