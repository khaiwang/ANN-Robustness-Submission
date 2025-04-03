# ANN Robustness
This is the codebase of our VLDB'26 submission: Towards Robustness: A Critique of Current Vector Database
Assessments.
We extend the Big-ANN-Benchmarks:
## Robustness Metric
#### Robustness@$\delta$ metric support
By assigning the `x-axis` or `y-axis` to the `robustness@$\delta$`($\delta$ is a customized threshold to show the number of queries with recall $\ge\ \delta$) metric, we can plot the robustness metric with a fixed average recall.
#### Benchmark dockers according to the paper setup.
We adopted the neurips23track ood setup to run our evaluation, as it is the latest setup and contains up-to-date datasets and algorithms.
* Install the requirements in the requirements_py3.10.txt
* Indices support: ScaNN, Zilliz, Puck, DiskANN, Faiss and Faiss-hnsw
* Dataset: Text2Image10M, MSSPACEV10M, DEEP10M, and our customized MSMARCO for RAG
    * We apply the setup of ScaNN Zilliz Puck and DiskANN in the original ood track as they have been tuned for this out-of-distribution setup.
    * On other datasets, we use default setting and tune the parameters to limit them to a similar average recall rate.
* The evaluation script can be found in the robustness_evaluation.py
#### Plotting: Filter with a third condition beyond -x and -y, Robustness@$\delta$ figure with a fixed average recall.
1. For the figure with filter, use `--plot-type filter` to plot the figure with a third condition.
For this figure, use `--fix-metric` to denote the metric used for filtering, `--min` and `--max` to denote the range of the third condition
2. For the figure with robustness@$\delta$, use `--plot-type cdf` to plot the figure with a fixed average recall.
Use `--fix-recall` to denote the fixed average recall for the robustness plot.
## RAG
1. MSMARCO (Emebdding with LLM-Embedder) for RAG evaluation.
2. Embedding the dataset with LLM with LLM-Embedder.
3. Filter the question set, keep the queries that LLM can answer with the embedded top-10 KNN ground truth.
4. RAG with the embedded corpus and ANN results. 
Note that the current workflow is not automated, should manually run the vector search and use the results to run RAG.
5. RAG Evaluation

---
Below is the original Big-ANN-Benchmarks README contents, the corresponding environment is in the `requirements_py3.10.txt` file. 
---


# Big ANN Benchmarks

<http://big-ann-benchmarks.com/>

## Datasets

See <http://big-ann-benchmarks.com/> for details on the different datasets.

## NeurIPS 2023 competition: Practical Vector Search

Please see [this readme](./neurips23/README.md) for a guide to the NeurIPS 23 competition.

## NeurIPS 2021 competition: Billion-Scale ANN 

Please see [this readme](./neurips21/README.md) for a guide of running billion-scale benchmarks and a summary of the results from the NeurIPS 21 competition.

# Credits

This project is a version of [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) by [Erik Bernhardsson](https://erikbern.com/) and contributors targeting evaluation of algorithms and hardware for newer billion-scale datasets and practical variants of nearest neighbor search.
