FROM neurips23

RUN apt update
RUN apt install -y software-properties-common
RUN pip install --no-cache-dir scann

RUN pip install -r requirements_py3.10.txt

WORKDIR /home/app
