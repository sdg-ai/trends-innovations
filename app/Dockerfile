FROM python:3.11.4-bookworm

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY train.py /app/train.py
COPY app.py /app/main.py
ADD utils /app/utils
ADD models /app/models
ADD data /app/data
ADD checkpoints /app/checkpoints
ADD templates /app/templates

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]

# build
#  docker build . -t tandic:latest
# run
# docker run -d -p 80:80 tandic:latest