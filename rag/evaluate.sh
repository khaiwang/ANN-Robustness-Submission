export GPT_API=your_api_key
# Build the environment
conda env create -f environment.yml
# Load the MSMARCO dataset and encode them
python encode_dataset.py
# Calculate the encoded dataset's knn
python calculate_gt.py
# Generate the query-corpus results for knn
python generate_qa_corpus_pairs.py
# Calculate the RAG results for knn
# We input empty docs in non_gt to verify that GPT cannot answer the questions without retrieved docs
python evaluate_rag.py --dataset knn
python evaluate_rag.py --dataset non_gt
# Before this step, first need to run MSMARCO vector search for diskann and scann
# Load vector search results
python generate_rag_q_corpus_pairs.py
# Calculate the RAG results for scann and diskann
python evaluate_rag.py --dataset scann_95
python evaluate_rag.py --dataset scann_90
python evaluate_rag.py --dataset diskann_95
python evaluate_rag.py --dataset diskann_90
