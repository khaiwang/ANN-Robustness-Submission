from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}

print("Start loading MSMARCO dataset")
data = load_dataset("namespace-Pt/msmarco", split="dev")
queries = np.array(data["query"])
print("shape of the queries: ", queries.shape)

corpus = load_dataset("namespace-PT/msmarco-corpus", split="train")
print("shape of the corpus: ", corpus.shape)
task = "qa"
# get the BGE embedding model
model = FlagModel('BAAI/llm-embedder', 
                  use_fp16=False,
                  query_instruction_for_retrieval=INSTRUCTIONS[task]['query'],
                  passage_instruction_for_retrieval=INSTRUCTIONS[task]['key'],
                  devices=['cuda:0'])

# encode the queries and corpus in batches
query_batch = 1000000
corpus_batch = 1000000
# write the query embeddings to a json
# write the corpus embeddings to a file
for i in tqdm(range(0, queries.shape[0], query_batch)):
    query_embeddings = model.encode_queries(queries[i:i+query_batch])
    print("shape of the query embeddings:", query_embeddings.shape)
    print("data type of the embeddings: ", query_embeddings.dtype)
    np.save(f"query_embeddings_{i}.npy", query_embeddings)

for i in tqdm(range(0, corpus.shape[0], corpus_batch)):
    print(f"processing {i} to {i+corpus_batch}")
    print("shape of the corpus: ", len(corpus['content'][i:i+corpus_batch]))
    corpus_embeddings = model.encode_corpus(corpus['content'][i:i+corpus_batch])
    print("shape of the corpus embeddings:", corpus_embeddings.shape)
    print("data type of the embeddings: ", corpus_embeddings.dtype)
    np.save(f"corpus_embeddings_{i}.npy", corpus_embeddings)