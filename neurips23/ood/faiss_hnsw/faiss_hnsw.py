import faiss
import numpy as np
from benchmark.datasets import DATASETS

from ..faiss.faiss import Faiss


class FaissHNSW(Faiss):
    def __init__(self, metric, index_params):
        self._metric = metric
        self.method_param = index_params

    def fit(self, dataset):
        faiss.omp_set_num_threads(64)
        ds = DATASETS[dataset]()
        d = ds.d
        X = ds.get_dataset().astype(np.float32)
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == "angular" else faiss.METRIC_L2
        factory_string = f"HNSW{self.method_param.get('M', 32)},Flat"
        self.index = faiss.index_factory(d, factory_string, faiss_metric)
        self.index.verbose = True

        # if self._metric == "angular":
        #     X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        # if X.dtype != np.float32:
        #     X = X.astype(np.float32)
        self.index.train(X)
        self.index.add(X)
        

    def set_query_arguments(self, query_arguments):
        faiss.omp_set_num_threads(64)
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = query_arguments.get("ef", 32)

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p
    
    def load_index(self, dataset):
        pass
