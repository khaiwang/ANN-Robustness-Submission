import sys

sys.path.append("install/lib-faiss")  # noqa
import faiss
import numpy as  np
import sklearn.preprocessing
import os
from benchmark.dataset_io import download
from benchmark.dataset_io import read_fbin
from benchmark.datasets import DATASETS
from neurips23.ood.base import BaseOODANN


class Faiss(BaseOODANN):
    def query(self, X, k):
        X_NORM = X
        if self._metric == "angular":
            X_NORM = X/np.linalg.norm(X)
        self.res = self.index.search(X_NORM.astype(np.float32), k)
        self.res = np.array(self.get_batch_results())

    def batch_query(self, X, n):
        X_NORM = X
        if self._metric == "angular":
            X_NORM = X/np.linalg.norm(X)
        
        self.res = self.index.search(X_NORM.astype(np.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res


class FaissLSH(Faiss):
    def __init__(self, metric, n_bits):
        self._n_bits = n_bits
        self.index = None
        self._metric = metric
        self.name = "FaissLSH(n_bits={})".format(self._n_bits)

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        f = X.shape[1]
        self.index = faiss.IndexLSH(f, self._n_bits)
        self.index.train(X)
        self.index.add(X)


class FaissIVF(Faiss):
    def __init__(self, metric, index_params):
        self._n_list = index_params.get("n_list", 100)
        self._metric = metric
        print("metric", metric)

    def fit(self, dataset):
        faiss.omp_set_num_threads(128)
        ds = DATASETS[dataset]()
        d = ds.d
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == "angular" else faiss.METRIC_L2
        factory_string = f"IVF{self._n_list},Flat"
        index = faiss.index_factory(d, factory_string, faiss_metric)
        X = ds.get_dataset().astype(np.float32)

        index.train(X)
        index.add(X)
        self.index = index
        faiss.write_index(index, self.index_name(dataset))
        
    
    def load_index(self, dataset):
        if os.path.exists(self.index_name(dataset)):
            self.index = faiss.read_index(self.index_name(dataset))
            return True
        return False

    def set_query_arguments(self, query_arguments):
        faiss.omp_set_num_threads(128)
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = query_arguments.get("n_probe", 32)
        self.index.nprobe = self._n_probe
    
    def index_name(self, name):
        return f"data/{name}.{self._n_list}.faissivfindex"
    
    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._n_list}  # noqa

    def __str__(self):
        return "FaissIVF(n_list=%d, n_probe=%d)" % (self._n_list, self._n_probe)


class FaissIVFPQfs(Faiss):
    def __init__(self, metric, index_params):
        self._n_list = index_params.get("n_list", 100)
        self._metric = metric
        print("metric", metric)

    def fit(self, dataset):
        faiss.omp_set_num_threads(64)
        ds = DATASETS[dataset]()
        d = ds.d
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == "angular" else faiss.METRIC_L2
        factory_string = f"IVF{self._n_list},PQ{d//2}x4fs"
        index = faiss.index_factory(d, factory_string, faiss_metric)
        X = ds.get_dataset().astype(np.float32)
        index.train(X)
        index.add(X)
        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.base_index = index
        self.refine_index = index_refine
        faiss.write_index(index, self.index_name(dataset))
    
    def load_index(self, dataset):
        pass

    def index_name(self, name):
        return f"data/{name}.{self._n_list}.faissivfqpfsindex"
    
    def set_query_arguments(self, query_arguments):
        faiss.cvar.indexIVF_stats.reset()
        faiss.omp_set_num_threads(64)
        self._n_probe = query_arguments.get("n_probe", 32)
        self._k_reorder = query_arguments.get("k_reorder", 100)
        self.base_index.nprobe = self._n_probe
        self.refine_index.k_factor = self._k_reorder
        
        if self._k_reorder == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._n_list}  # noqa

    def __str__(self):
        return "FaissIVFPQfs(n_list=%d, n_probe=%d, k_reorder=%d)" % (self._n_list, self._n_probe, self._k_reorder)
