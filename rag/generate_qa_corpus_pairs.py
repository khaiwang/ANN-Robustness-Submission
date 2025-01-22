from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import os

corpus = load_dataset("namespace-PT/msmarco-corpus", split="train")['content']

# load the queries and answers from fname
queries = []
with open("data/gt.json") as f:
    for line in f:   
        queries.append(json.loads(line))

fname = "data/gt.bin"

# load gt binary
n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
print(n, d)
assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
f = open(fname, "rb")
f.seek(4+4)
I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
knn_results = {}
# load the queries and corresponding knn corpus
for i in range(len(queries)):
    closest_docs = I[int(queries[i]["id"])][:10]
    doc_text = []
    for doc_id in closest_docs:
        doc_text.append(corpus[doc_id])
    knn_results[i] = {"id": queries[i]["id"], "query": queries[i]["query"], "answer": queries[i]["answer"], "doc": doc_text}

# store the results in a json file line by line
with open("data/knn.json", "w+") as f:
    for key, value in knn_results.items():
        f.write(json.dumps(value) + "\n")