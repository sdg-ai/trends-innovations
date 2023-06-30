FROM python:3.11.4-bookworm

RUN pip install torchmetrics
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install tqdm
RUN pip install wandb
RUN pip install numpy
RUN pip install transformers
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pydantic
RUN pip3 install torch torchvision torchaudio

COPY train.py /trends-innovations/train.py
COPY app.py /trends-innovations/app.py
ADD utils /trends-innovations/utils
ADD models /trends-innovations/models
ADD data /trends-innovations/data

CMD python3 /app/app.py

# build
# docker build . -t trends-innovations
# run
# docker run --rm -it -p 8000:8000/tcp trends-innovations