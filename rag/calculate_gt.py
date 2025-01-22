import numpy as np
import faiss
import os
batch = 1000000
k = 100
# initialize the base np array
# base = np.zeros((nb, d), dtype=np.float32)
separateD = []
separateI = []
xb = []
nb = 0
d = 0
for i in range(9):
    xb_batch = np.load(f"corpus_embeddings_{i * batch}.npy")
    xb.append(xb_batch)
    nb += xb_batch.shape[0]
    d = xb_batch.shape[1]

xb = np.concatenate(xb, axis=0)
print("shape of the xb: ", xb.shape) 

xq = np.load(f"query_embeddings.npy")
nq = xq.shape[0]
print(nq)
for i in range(0, nb, batch):
    x = xb[i:i+batch]
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    index_flat = faiss.IndexFlatIP(d)
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index.add(x)
    D, I = index.search(xq, k)

    separateD.append(D)
    # add the offset
    separateI.append(I)
    index.reset()
D = np.concatenate(separateD, axis=1)

I = np.concatenate(separateI, axis=1)

# merge the results, and remain the top 100

# sort the results by distance 
# argsort return the order in acsending order, so we need to reverse it
D_sorted = np.argsort(D, axis=1)[:, ::-1]
I_sorted = np.zeros_like(I)
for i_q in range(nq):
    for k_q in range(k):
        I_sorted[i_q][k_q] = I[i_q][D_sorted[i_q][k_q]]
D = np.take_along_axis(D, D_sorted, axis=1)[:, :k]
I = I_sorted[:, :k]

D = np.array(D)
I = np.array(I)

# save the results
with open("gt_msmarco.bin", "wb") as f:
    np.array([nq, d], dtype='uint32').tofile(f)
    print("write I shape to file: ", I.shape)
    I.astype('uint32').tofile(f)
    D.astype('float32').tofile(f)