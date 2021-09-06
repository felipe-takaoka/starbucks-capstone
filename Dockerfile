FROM python:3.8-slim-buster

# Copy only requirements.txt first so that "pip install" can be run
# before copying the whole app/ folder
COPY app/requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app