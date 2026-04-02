"""
Compute ground truth KNN for MSMARCO using GPU FAISS.
Strategy: add base vectors to GPU index in batches to control memory.
GPU 1 has ~44GB free. 8.8M x 768 x 4 bytes = 27GB.
We split into 2 halves, search each, then merge top-k.
"""
import os
import numpy as np
import faiss
import time

OUTDIR = 'data/msmarco'
base_file = os.path.join(OUTDIR, 'base.fbin')
query_file = os.path.join(OUTDIR, 'queries.fbin')
gt_file = os.path.join(OUTDIR, 'gt.bin')

with open(base_file, 'rb') as f:
    nb, d = np.fromfile(f, dtype='uint32', count=2)
    base = np.fromfile(f, dtype='float32').reshape(nb, d)
with open(query_file, 'rb') as f:
    nq, d2 = np.fromfile(f, dtype='uint32', count=2)
    queries = np.fromfile(f, dtype='float32').reshape(nq, d2)

print(f"Base: {base.shape} ({base.nbytes/1e9:.1f}GB), Queries: {queries.shape}")

k = 100
n_splits = 3  # split base into 3 chunks (~9GB each, fits in 44GB GPU)

res = faiss.StandardGpuResources()
res.setTempMemory(2 * 1024 * 1024 * 1024)

chunk_size = (nb + n_splits - 1) // n_splits
all_D = np.full((nq, k), -np.inf, dtype='float32')
all_I = np.full((nq, k), -1, dtype='int64')

t_total = time.time()

for chunk_idx in range(n_splits):
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, nb)
    chunk = base[start:end]
    print(f"\nChunk {chunk_idx+1}/{n_splits}: vectors [{start}, {end}) = {len(chunk)} vectors ({chunk.nbytes/1e9:.1f}GB)")

    # Build GPU index for this chunk
    index_cpu = faiss.IndexFlatIP(int(d))
    index_cpu.add(chunk)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    t0 = time.time()
    D_chunk, I_chunk = gpu_index.search(queries, k)
    elapsed = time.time() - t0
    print(f"  Searched in {elapsed:.1f}s")

    # Offset indices by chunk start
    I_chunk += start

    # Merge: for each query, keep top-k from combined results
    for q in range(nq):
        combined_D = np.concatenate([all_D[q], D_chunk[q]])
        combined_I = np.concatenate([all_I[q], I_chunk[q]])
        top_k_idx = np.argsort(-combined_D)[:k]  # descending for IP
        all_D[q] = combined_D[top_k_idx]
        all_I[q] = combined_I[top_k_idx]

    # Free GPU memory
    del gpu_index, index_cpu
    print(f"  Merged. GPU memory freed.")

elapsed_total = time.time() - t_total
print(f"\nTotal time: {elapsed_total:.1f}s")

with open(gt_file, 'wb') as f:
    np.array([nq, k], dtype='uint32').tofile(f)
    all_I.astype('uint32').tofile(f)
    all_D.astype('float32').tofile(f)

print(f"Saved {gt_file} ({os.path.getsize(gt_file)} bytes)")
print("Done!")
