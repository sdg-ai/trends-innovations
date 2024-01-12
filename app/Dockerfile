FROM python:3.11.4-bookworm

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY run.py /app/run.py
COPY app.py /app/app.py
ADD src /app/src
ADD checkpoints /app/checkpoints

EXPOSE 80
# start run.py
CMD ["python3", "run.py"]

# build
#  docker build . -t tandic:latest
# run
#  docker run -p 80:80 tandic:latest