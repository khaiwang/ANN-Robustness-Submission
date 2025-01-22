import h5py
from datasets import load_dataset
import numpy as np
import os
import json

corpus = load_dataset("namespace-PT/msmarco-corpus", split="train")['content']
questions = load_dataset("namespace-PT/msmarco", split="dev")
pos_count = {}

diskann_doc_90 = h5py.File('msmarco/ip_L_300_R_16_buildthreads_64_Ls_40_T_32.hdf5', 'r+', libver='latest')
diskann_doc_95 = h5py.File('msmarco/ip_L_500_R_32_buildthreads_64_Ls_30_T_32.hdf5', 'r+', libver='latest')

scann_doc_95 = h5py.File('msmarco/ip_download_false_tree_size_40000_leaves_to_search_75_reorder_150.hdf5', 'r+', libver='latest')
scann_doc_90 = h5py.File('msmarco/ip_download_false_tree_size_40000_leaves_to_search_20_reorder_150.hdf5', 'r+', libver='latest')

diskann_neighbors_90 = diskann_doc_90['neighbors']
diskann_neighbors_95 = diskann_doc_95['neighbors']

scann_neighbors_90 = scann_doc_90['neighbors']
scann_neighbors_95 = scann_doc_95['neighbors']

recalls_diskann_90 = diskann_doc_90['metrics']['knn']['recalls']
recalls_diskann_95 = diskann_doc_95['metrics']['knn']['recalls']
recalls_scann_90 = scann_doc_90['metrics']['knn']['recalls']
recalls_scann_95 = scann_doc_95['metrics']['knn']['recalls']

gt = []
with open("result-knn/prediction_knn_chatgpt_temp0.2_noise0.0_passage10_correct0.0.json") as f:
    for line in f:   
        gt.append(json.loads(line))

disk_90_results = {}
disk_95_results = {}
scann_90_results = {}
scann_95_results = {}
non_gt_results = {}
for q in range(len(gt)):
    if gt[q]['label'] != 1:
        # wrong in knn, skip
        continue
    q_id = int(gt[q]["id"])
    closet_docs_90_disk = diskann_neighbors_90[q_id]
    closet_docs_95_disk = diskann_neighbors_95[q_id]
    closet_docs_90_scann = scann_neighbors_90[q_id]
    closet_docs_95_scann = scann_neighbors_95[q_id]
    doc_text_90_disk = []
    doc_text_95_disk = []
    for doc_id in closet_docs_90_disk:
        doc_text_90_disk.append(corpus[doc_id])
    for doc_id in closet_docs_95_disk:
        doc_text_95_disk.append(corpus[doc_id])
    doc_text_90_scann = []
    doc_text_95_scann = []
    for doc_id in closet_docs_90_scann:
        doc_text_90_scann.append(corpus[doc_id])
    for doc_id in closet_docs_95_scann:
        doc_text_95_scann.append(corpus[doc_id])
    doc_text_non_gt = ["This is a placeholder for the document text. Please replace this with the actual document text."] * 10
    disk_90_results[q] = {"id": q_id, "query": gt[q]["query"], "recall": recalls_diskann_90[q_id], "answer": gt[q]["ans"], "doc": doc_text_90_disk}
    disk_95_results[q] = {"id": q_id, "query": gt[q]["query"], "recall": recalls_diskann_95[q_id], "answer": gt[q]["ans"], "doc": doc_text_95_disk}
    scann_90_results[q] = {"id": q_id, "query": gt[q]["query"], "recall": recalls_scann_90[q_id], "answer": gt[q]["ans"], "doc": doc_text_90_scann}
    scann_95_results[q] = {"id": q_id, "query": gt[q]["query"], "recall": recalls_scann_95[q_id], "answer": gt[q]["ans"], "doc": doc_text_95_scann}
    non_gt_results[q] = {"id": q_id, "query": gt[q]["query"], "recall": 1, "answer": gt[q]["ans"], "doc": doc_text_non_gt}
    

with open("data/diskann_90.json", "w+") as f:
    for key, value in disk_90_results.items():
        f.write(json.dumps(value) + "\n")

with open("data/diskann_95.json", "w+") as f:
    for key, value in disk_95_results.items():
        f.write(json.dumps(value) + "\n")

with open("data/scann_90.json", "w+") as f:
    for key, value in scann_90_results.items():
        f.write(json.dumps(value) + "\n")

with open("data/scann_95.json", "w+") as f:
    for key, value in scann_95_results.items():
        f.write(json.dumps(value) + "\n")

with open("data/non_gt.json", "w+") as f:
    for key, value in non_gt_results.items():
        f.write(json.dumps(value) + "\n")