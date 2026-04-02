"""
Encode MSMARCO dataset with LLM-Embedder and prepare for ANN benchmarking.
Produces: data/msmarco/query_embeddings.npy, data/msmarco/corpus_embeddings_*.npy
Then runs prepare() to create base.fbin and queries.fbin.
Then computes ground truth KNN using GPU FAISS.
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import numpy as np
from datasets import load_dataset
from FlagEmbedding import FlagModel

OUTDIR = 'data/msmarco'
os.makedirs(OUTDIR, exist_ok=True)

# Check if embeddings already exist
query_file = os.path.join(OUTDIR, 'query_embeddings.npy')
if os.path.exists(query_file):
    print(f"Query embeddings already exist at {query_file}, skipping encoding.")
else:
    print("Loading MSMARCO dataset...")
    data = load_dataset("namespace-Pt/msmarco", split="dev")
    queries = np.array(data["query"])
    print(f"Queries: {len(queries)}")

    corpus_ds = load_dataset("namespace-PT/msmarco-corpus", split="train")
    print(f"Corpus: {len(corpus_ds)}")

    INSTRUCTIONS = {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    }

    print("Loading LLM-Embedder model...")
    model = FlagModel('BAAI/llm-embedder',
                      use_fp16=True,
                      query_instruction_for_retrieval=INSTRUCTIONS['query'])

    # Encode queries
    print("Encoding queries...")
    query_embeddings = model.encode_queries(queries)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    np.save(os.path.join(OUTDIR, 'query_embeddings.npy'), query_embeddings)

    # Encode corpus in batches
    batch_size = 1000000
    corpus_content = corpus_ds['content']
    for i in range(0, len(corpus_content), batch_size):
        outfile = os.path.join(OUTDIR, f'corpus_embeddings_{i}.npy')
        if os.path.exists(outfile):
            print(f"Batch {i} already exists, skipping.")
            continue
        end = min(i + batch_size, len(corpus_content))
        print(f"Encoding corpus batch {i}-{end}...")
        batch = corpus_content[i:end]
        # Use encode() with passage instruction prefix for corpus
        prefixed = [INSTRUCTIONS['key'] + text for text in batch]
        embeddings = model.encode(prefixed)
        print(f"  Shape: {embeddings.shape}")
        np.save(outfile, embeddings)

    del model
    print("Encoding complete.")

# Prepare binary files for the benchmark framework
print("\nPreparing binary files (base.fbin, queries.fbin)...")
from benchmark.datasets import DATASETS
ds = DATASETS['msmarco-10M']()
ds.prepare()
print("Binary files ready.")

# Compute ground truth using GPU FAISS
gt_file = os.path.join(OUTDIR, 'gt.bin')
if os.path.exists(gt_file):
    print(f"Ground truth already exists at {gt_file}, skipping.")
else:
    print("\nComputing ground truth KNN using GPU FAISS...")
    import faiss

    # Load the binary files we just created
    base_file = os.path.join(OUTDIR, 'base.fbin')
    query_file_bin = os.path.join(OUTDIR, 'queries.fbin')

    # Read binary format: [nb, d] as uint32 header, then float32 data
    with open(base_file, 'rb') as f:
        nb, d = np.fromfile(f, dtype='uint32', count=2)
        base = np.fromfile(f, dtype='float32').reshape(nb, d)
    with open(query_file_bin, 'rb') as f:
        nq, d2 = np.fromfile(f, dtype='uint32', count=2)
        queries = np.fromfile(f, dtype='float32').reshape(nq, d2)

    print(f"Base: {base.shape}, Queries: {queries.shape}")
    assert d == d2

    # MSMARCO uses inner product
    k = 100  # compute top-100 for ground truth

    # Use GPU
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(int(d))
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    print("Adding vectors to GPU index...")
    gpu_index.add(base)

    print(f"Searching {nq} queries for top-{k}...")
    D, I = gpu_index.search(queries, k)

    # Sort by distance descending (IP: higher is better)
    # FAISS already returns sorted descending for IP
    print(f"Results: D shape={D.shape}, I shape={I.shape}")

    # Write ground truth in the expected binary format
    with open(gt_file, 'wb') as f:
        np.array([nq, k], dtype='uint32').tofile(f)
        I.astype('uint32').tofile(f)
        D.astype('float32').tofile(f)

    print(f"Ground truth saved to {gt_file}")
    print(f"File size: {os.path.getsize(gt_file)} bytes")

print("\nAll done! MSMARCO dataset is ready for benchmarking.")
