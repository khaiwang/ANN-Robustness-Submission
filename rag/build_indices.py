#!/usr/bin/env python3
"""Build HNSW and IVF indices directly from flat FAISS index (no intermediate embedding file)."""
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import faiss
import numpy as np
import time
import sys
sys.path.insert(0, os.path.dirname(__file__))
from agentic_rag_pipeline import FLAT_INDEX_PATH, HNSW_DIR, IVF_DIR, HNSW_CONFIGS, IVF_CONFIGS

faiss.omp_set_num_threads(16)

# Read flat index once, extract vectors in memory
t0 = time.time()
print(f"Loading flat index from {FLAT_INDEX_PATH}...")
flat = faiss.read_index(str(FLAT_INDEX_PATH))
n, d = flat.ntotal, flat.d
print(f"  {n} vectors, dim={d}, loaded in {time.time()-t0:.1f}s")

t0 = time.time()
print("Reconstructing vectors from flat index...")
embeddings = np.zeros((n, d), dtype=np.float32)
flat.reconstruct_n(0, n, embeddings)
print(f"  Reconstructed in {time.time()-t0:.1f}s")
del flat

# IVF (GPU training on cuda:2)
ivf_path = IVF_DIR / f"e5_IVF{IVF_CONFIGS['n_list']}.index"
if not ivf_path.exists():
    ivf_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Building IVF (n_list={IVF_CONFIGS['n_list']}, GPU cuda:2)...")
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, IVF_CONFIGS["n_list"], faiss.METRIC_INNER_PRODUCT)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 2, index)
    gpu_index.train(embeddings[:500000])
    index = faiss.index_gpu_to_cpu(gpu_index)
    del gpu_index, res
    print(f"  Trained in {time.time()-t0:.1f}s. Adding vectors...")
    t1 = time.time()
    index.add(embeddings)
    faiss.write_index(index, str(ivf_path))
    print(f"  IVF done in {time.time()-t1:.1f}s total. Saved: {ivf_path}")
    del index
else:
    print(f"IVF exists: {ivf_path}")

# HNSW (CPU only, 16 threads)
hnsw_path = HNSW_DIR / f"e5_HNSW{HNSW_CONFIGS['M']}.index"
if not hnsw_path.exists():
    hnsw_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Building HNSW (M={HNSW_CONFIGS['M']}, efC={HNSW_CONFIGS['efConstruction']}, 16 threads)...")
    index = faiss.IndexHNSWFlat(d, HNSW_CONFIGS["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_CONFIGS["efConstruction"]
    index.train(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(hnsw_path))
    print(f"  HNSW done in {time.time()-t0:.1f}s. Saved: {hnsw_path}")
else:
    print(f"HNSW exists: {hnsw_path}")

print("Done.")
