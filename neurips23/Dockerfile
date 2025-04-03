FROM ubuntu:jammy

RUN apt-get update && apt-get install -y python3-numpy python3-scipy python3-pip build-essential git axel wget
RUN wget https://aka.ms/downloadazcopy-v10-linux && mv downloadazcopy-v10-linux azcopy.tgz && tar xzf azcopy.tgz --transform 's!^[^/]\+\($\|/\)!azcopy_folder\1!' 
RUN cp azcopy_folder/azcopy /usr/bin

# change python3 to python3.10
RUN apt-get install -y python3.10 python3.10-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3

RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements_py3.10.txt run_algorithm.py ./
RUN pip3 install -r requirements_py3.10.txt

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
