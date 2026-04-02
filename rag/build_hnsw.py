#!/usr/bin/env python3
"""Build HNSW index using hnswlib (much faster than FAISS HNSW)."""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import hnswlib
import faiss
import numpy as np
import time

FLAT_INDEX = "/disk/u/zikai/robustness/ANN-Robustness-Revision/data/search-r1/e5_Flat.index"
OUT_PATH = "/disk/u/zikai/robustness/ANN-Robustness-Revision/data/search-r1/hnsw/e5_HNSW64.hnswlib"

M = 64
efC = 300
NUM_THREADS = 16

# Load vectors from flat index
t0 = time.time()
print(f"Loading flat index...")
flat = faiss.read_index(FLAT_INDEX)
n, d = flat.ntotal, flat.d
print(f"  {n} vectors, dim={d}, {time.time()-t0:.1f}s")

t0 = time.time()
print("Reconstructing vectors...")
data = np.zeros((n, d), dtype=np.float32)
flat.reconstruct_n(0, n, data)
del flat
print(f"  Done in {time.time()-t0:.1f}s")

# Build hnswlib index
t0 = time.time()
print(f"Building hnswlib HNSW: M={M}, efC={efC}, {NUM_THREADS} threads...")
index = hnswlib.Index(space="ip", dim=d)
index.init_index(max_elements=n, ef_construction=efC, M=M)
index.set_num_threads(NUM_THREADS)
index.add_items(data, np.arange(n))
print(f"  Built in {time.time()-t0:.1f}s")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
index.save_index(OUT_PATH)
print(f"  Saved: {OUT_PATH}")
print(f"  Size: {os.path.getsize(OUT_PATH)/1e9:.1f}GB")
