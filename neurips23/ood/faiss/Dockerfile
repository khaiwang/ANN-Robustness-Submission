FROM neurips23

# Download and install Anaconda
RUN apt update && apt install -y wget && \
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && \
    bash Anaconda3-2020.11-Linux-x86_64.sh -b && \
    rm Anaconda3-2020.11-Linux-x86_64.sh

ENV PATH=/root/anaconda3/bin:$PATH

# Create new conda environment with Python 3.10
RUN conda create -y -n py310 python=3.10 && \
    echo "source activate py310" > ~/.bashrc

# Use the Python 3.10 environment as default
ENV PATH=/root/anaconda3/envs/py310/bin:$PATH

# Install pip packages
RUN pip install ansicolors==1.1.8 docker==5.0.2

# Install conda packages (inside py310)
RUN conda install -n py310 -y -c pytorch faiss-cpu h5py numpy

RUN pip install -r requirements_py3.10.txt

# Fix MKL-related issues
ENV LD_PRELOAD=/root/anaconda3/envs/py310/lib/libmkl_core.so:/root/anaconda3/envs/py310/lib/libmkl_sequential.so

# Final check: confirm Python 3.10 and faiss
RUN python3 -c "import sys; print(sys.version); import faiss; print(faiss.IndexFlatL2)"


