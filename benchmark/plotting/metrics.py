from __future__ import absolute_import
import numpy as np
import itertools
import operator
import random
import sys
import copy
import fnmatch

from collections import defaultdict

from benchmark.plotting.eval_range_search import compute_AP
from benchmark.sensors.power_capture import power_capture

def compute_recall_without_distance_ties(true_ids, run_ids, count):
    return len(set(true_ids) & set(run_ids))

def compute_recall_with_distance_ties(true_ids, true_dists, run_ids, count):
    # This function assumes "true_dists" is monotonic either increasing or decreasing
    found_tie = False
    gt_size = np.shape(true_dists)[0]

    if gt_size==count:
        # nothing fancy to do in this case
        recall =  len(set(true_ids[:count]) & set(run_ids))

    else:
        dist_tie_check = true_dists[count-1] # tie check anchored at count-1 in GT dists
     
        set_end = gt_size

        for i in range(count, gt_size):
          is_close = abs(dist_tie_check - true_dists[i] ) < 1e-6 
          if not is_close:
            set_end = i
            break

        found_tie = set_end > count

        recall =  len(set(true_ids[:set_end]) & set(run_ids))
 
    return recall, found_tie

def get_recall_values(true_nn, run_nn, count, count_ties=True):
    true_ids, true_dists = true_nn
    if not count_ties:
        true_ids = true_ids[:, :count]
        assert true_ids.shape == run_nn.shape
    recalls = np.zeros(len(run_nn))
    queries_with_ties = 0
    # TODO probably not very efficient
    for i in range(len(run_nn)):
        if count_ties:
            # if i == 55:
            #     print("run_nn", run_nn[i])
            #     print("true_ids", true_ids[i])
            #     print("true_dists", true_dists[i])
            recalls[i], found_tie = compute_recall_with_distance_ties(true_ids[i], true_dists[i], run_nn[i], count)
            if found_tie: queries_with_ties += 1 
        else:
            recalls[i] = compute_recall_without_distance_ties(true_ids[i], run_nn[i], count)
    return (np.mean(recalls) / float(count),
            np.std(recalls) / float(count),
            recalls,
            queries_with_ties)

def compute_mrr(true_ids, run_ids, count):
    for i, id in enumerate(true_ids):
        if i >= len(run_ids):
            break
        if id in run_ids:
            return count / float(i + 1)
    return 0.0

def compute_map(true_ids, run_ids, count):
    num_correct = 0
    sum_precision = 0.0
    for i, id in enumerate(true_ids):
        if i >= len(run_ids):
            break
        if id in run_ids:
            num_correct += 1
            sum_precision += count * num_correct / float(i + 1)
    
    return sum_precision

def compute_ndcg(true_ids, run_ids, count):
    dcg = 0.0
    idcg = 0.0
    for i, id in enumerate(true_ids):
        if i >= len(run_ids):
            break
        if id in run_ids:
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(count, len(true_ids))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg * count

def get_IR_values(true_nn, run_nn, count, name, count_ties=True):
    true_ids, _ = true_nn
    # order the run_nn by distance

    if not count_ties:
        true_ids = true_ids[:, :count]
        assert true_ids.shape == run_nn.shape
    metric_values = np.zeros(len(run_nn))
    # TODO probably not very efficient
    for i in range(len(run_nn)):
        if name == "mrr":
            metric_values[i] = compute_mrr(true_ids[i], run_nn[i], count)
        elif name == "map":
            metric_values[i] = compute_map(true_ids[i], run_nn[i], count)
        elif name == "ndcg":
            metric_values[i] = compute_ndcg(true_ids[i], run_nn[i], count)
    return (np.mean(metric_values),
            np.std(metric_values),
            metric_values)
def tail_knn(true_nn, run_nn, count, metrics, tail_percentile):
    s = "tail" + str(tail_percentile)
    # tail percentile is a number at the right of the decimal point, can be 99, 999, 9999, etc
    # get the digit at the right of the decimal point by dividing by 10 until the number is less than 1
    tail = tail_percentile
    while tail >= 100.0:
        tail /= 10.0


    print("tail: ", tail)
    if "knn" not in metrics:
        print("Computing knn metrics")
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls, queries_with_ties = get_recall_values(true_nn, run_nn, count)
        if queries_with_ties>0:
            print("Warning: %d/%d queries contained ties accounted for in recall" % (queries_with_ties, len(run_nn)))
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached recall result")
    if s not in metrics["knn"]:        
        print("Computing tail metrics", s)
        metrics["knn"].attrs[s] = np.percentile(np.array(metrics["knn"]["recalls"]), 100 - tail)
    else:
        print("Found cached tail result", s)
    return metrics["knn"].attrs[s]/count

def knn(true_nn, run_nn, count, metrics):
    if 'knn' not in metrics:
        print('Computing knn metrics')
        knn_metrics = metrics.create_group('knn')
        mean, std, recalls, queries_with_ties = get_recall_values(true_nn, run_nn, count)
        if queries_with_ties>0:
            print("Warning: %d/%d queries contained ties accounted for in recall" % (queries_with_ties, len(run_nn)))
        knn_metrics.attrs['mean'] = mean
        knn_metrics.attrs['std'] = std
        knn_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics['knn']

def ap(true_nn, run_nn, metrics):
    if'ap' not in metrics:
        print('Computing ap metrics')
        gt_nres, gt_I, gt_D = true_nn
        nq = gt_nres.shape[0]
        gt_lims = np.zeros(nq + 1, dtype=int)
        gt_lims[1:] = np.cumsum(gt_nres)
        ap = compute_AP((gt_lims, gt_I, gt_D), run_nn)
        ap_metric = metrics.create_group('ap')
        ap_metric.attrs['mean'] = ap
    else:
        print("Found cached result")
    return metrics['ap'].attrs['mean']

def queries_per_second(nq, attrs):
    return nq / attrs["best_search_time"]


def index_size(attrs):
    return attrs.get("index_size", 0)


def build_time(attrs):
    return attrs.get("build_time", -1)


def dist_computations(nq, attrs):
    return attrs.get("dist_comps", 0) / (attrs['run_count'] * nq)

def watt_seconds_per_query(queries, attrs):
    return power_capture.compute_watt_seconds_per_query(queries, attrs )

def mean_ssd_ios(attrs):
    return attrs.get("mean_ssd_ios", 0)

def mean_latency(attrs):
    return attrs.get("mean_latency", 0)

def cdf(dataset_distances, run_distances, count, metrics):
    cdf = np.zeros(11)
    if "knn" not in metrics:
        print("Computing knn metrics")
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls, queries_with_ties = get_recall_values(dataset_distances, run_distances, count)
        if queries_with_ties>0:
            print("Warning: %d/%d queries contained ties accounted for in recall" % (queries_with_ties, len(run_distances)))
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached knn result")
    if "cdf" not in metrics["knn"]:
        print("Computing cdf metrics")
        knn_metrics = metrics["knn"]
        # recall_values = round_to_nearest_tenth(np.array(knn_metrics["recalls"]))
         # num_bins = 11
        bins = np.arange(-0.05, 1.15, 0.1)
        print(knn_metrics["recalls"])
        counts, bins = np.histogram(np.array(knn_metrics["recalls"])/count, bins=bins)
        cdf = np.cumsum(counts)
        print(cdf)
        cdf = cdf / cdf[-1]
        # we want to calculate >=, so we add 0 to the start of the cdf and remove the last one
        cdf = np.insert(cdf, 0, 0)
        cdf = cdf[:-1]
        knn_metrics["cdf"] = cdf
        print(knn_metrics["cdf"])
    else:
        knn_metrics = metrics["knn"]
        cdf = np.array(knn_metrics["cdf"])
        print(cdf)
        print("Found cached cdf result")
    return cdf

def robustness(dataset_distances, run_distances, count, metrics, robust_threshold):
    # name robustness: robustness + threshold * 100 round to integer  
    s = "robustness-" + str(robust_threshold) + "@" + str(count)
    delta = robust_threshold
    print("delta", delta)
    if "knn" not in metrics:
        print("Computing knn metrics")
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls, queries_with_ties = get_recall_values(dataset_distances, run_distances, count)
        if queries_with_ties>0:
            print("Warning: %d/%d queries contained ties accounted for in recall" % (queries_with_ties, len(run_distances)))
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached recall result")
    if s not in metrics["knn"]:        
        print("Computing robustness metrics", s)
        metrics["knn"].attrs[s] = np.sum(np.array(metrics["knn"]["recalls"]) >= delta * count) / len(metrics["knn"]["recalls"])
    else:
        print("Found cached robustness result", s)
    return metrics["knn"].attrs[s]

def robustness_metric(key):
    if key.startswith("robustness"):
        # keey two decimal places
        threshold = float(key.split("robustness@")[1])
        threshold = round(threshold, 2)
        # threshold = float(key.split("robustness@")[1])
        # print("threshold", threshold)
        return {
            "description": f"Frequency of Queries with Recall >= {threshold} (Robusness@{threshold})",
            "function": lambda true_distances, run_distances, metrics, run_attrs: robustness(
                true_distances, run_distances, run_attrs["count"], metrics, threshold
            ),
            # noqa
            "worst": float("-inf"),
            "best": float("inf"),
            "lim": [0.0, 1.00],
        }
    raise ValueError(f"Unknown metric {key}")

class KeyAwareDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            return robustness_metric(key)


all_metrics = KeyAwareDefaultDict(robustness_metric, {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_nn, run_nn, metrics, run_attrs: knn(true_nn, run_nn, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "robustness*": {
        # keep this empty for keys() to find it
        # avoid using this key directly
    },
    "cdf": {
        "description": "CDF",
        "function": lambda true_distances, run_distances, metrics, run_attrs: cdf(
            true_distances, run_distances, run_attrs["count"], metrics
        ),
        "worst": float("-inf"),
        "best": float("inf"),
        "lim": [0.0, 1.00],
    },
    "ap": {
        "description": "Average Precision",
        "function": lambda true_nn, run_nn, metrics, run_attrs: ap(true_nn, run_nn, metrics),  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
        "search_type" : "range",
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: queries_per_second(len(true_nn[0]), run_attrs),  # noqa
        "worst": float("-inf")
    },
    "distcomps": {
        "description": "Distance computations",
        "function": lambda true_nn, run_nn,  metrics, run_attrs: dist_computations(len(true_nn[0]), run_attrs), # noqa
        "worst": float("inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: build_time(run_attrs), # noqa
        "worst": float("inf")
    },
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: index_size(run_attrs),  # noqa
        "worst": float("inf")
    },
    # "queriessize": {
    #     "description": "Index size (kB)/Queries per second (s)",
    #     "function": lambda true_nn, run_nn, metrics, run_attrs: index_size(run_attrs) / queries_per_second(len(true_nn[0]), run_attrs), # noqa
    #     "worst": float("inf")
    # },
    "wspq": {
        "description": "Watt seconds per query (watt*s/query)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: watt_seconds_per_query(true_nn, run_attrs),  
        "worst": float("-inf")
    },
    "mean_ssd_ios": {
        "description": "Average SSD I/Os per query",
        "function": lambda true_nn, run_nn, metrics, run_attrs: mean_ssd_ios(run_attrs),  
        "worst": float("inf")
    },
    "mean_latency": {
        "description": "Mean latency across queries",
        "function": lambda true_nn, run_nn, metrics, run_attrs: mean_latency(run_attrs),  
        "worst": float("inf")
    },
    "search_times": {
        "description": "List of consecutive search times for the same run parameter",
        "function": lambda true_nn, run_nn, metrics, run_attrs: run_attrs.get("search_times",[]), 
        "worst": float("inf")
    },
    "mrr": {
        "description": "Mean Reciprocal Rank",
        "function": lambda true_nn, run_nn, metrics, run_attrs: IR(true_nn, run_nn, run_attrs["count"], metrics, 'mrr'),
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },

    "map": {
        "description": "Mean Average Precision",
        "function": lambda true_nn, run_nn, metrics, run_attrs: IR(true_nn, run_nn, run_attrs["count"], metrics, 'map'),
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },

    "ndcg": {
        "description": "Mean Normalized Discounted Cumulative Gain",
        "function": lambda true_nn, run_nn, metrics, run_attrs: IR(true_nn, run_nn, run_attrs["count"], metrics, 'ndcg'),
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "tail99": {
        "description": "Recall at 99th percentile",
        "function": lambda true_nn, run_nn, metrics, run_attrs: tail_knn(true_nn, run_nn, run_attrs["count"], metrics, 99),
        "worst": float("-inf"),
        "lim": [-0.03, 1.03],
    },

    "tail999": {
        "description": "Recall at 99.9th percentile",
        "function": lambda true_nn, run_nn, metrics, run_attrs: tail_knn(true_nn, run_nn, run_attrs["count"], metrics, 999),
        "worst": float("-inf"),
        "lim": [-0.03, 1.03],
    },

    "tail95": {
        "description": "Recall at 95th percentile",
        "function": lambda true_nn, run_nn, metrics, run_attrs: tail_knn(true_nn, run_nn, run_attrs["count"], metrics, 95),
        "worst": float("-inf"),
        "lim": [-0.03, 1.03],
    },

})

