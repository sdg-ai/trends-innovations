FROM python:3.11.4-bookworm

RUN mkdir /trends-innovations

COPY requirements.txt /trends-innovations/requirements.txt
COPY train.py /trends-innovations/train.py
COPY app.py /trends-innovations/app.py
ADD utils /trends-innovations/utils
ADD models /trends-innovations/models
ADD data /trends-innovations/data

RUN pip3 install -r /trends-innovations/requirements.txt

CMD python3 /trends-innovations/app.py

# build
# docker build . -t trends-innovations
# run
# docker run --rm -it -p 8000:8000/tcp trends-innovations