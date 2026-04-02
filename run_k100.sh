#!/bin/bash
# Run K=100 evaluation on text2image-10M for all algorithms
cd /disk/u/zikai/robustness/ANN-Robustness-Revision
LOG_DIR="logs_16cpu"

run_one() {
    local algo=$1
    echo "$(date): === Running $algo on text2image-10M K=100 ===" | tee -a "$LOG_DIR/run.log"
    python3 run.py --neurips23track ood --algorithm $algo --dataset text2image-10M --count 100 --runs 5 \
        2>&1 | tee "$LOG_DIR/${algo}_text2image-10M_k100.log"
    echo "$(date): $algo K=100 finished with exit code $?" | tee -a "$LOG_DIR/run.log"
}

for algo in faiss_hnsw diskann scann zilliz puck faiss-ivf; do
    run_one "$algo"
done

echo "$(date): All K=100 runs complete" | tee -a "$LOG_DIR/run.log"
