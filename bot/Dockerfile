FROM python:3.12-slim-bullseye

RUN apt-get update

ENV BOT_HOME=/home/bot/

RUN mkdir -p ${BOT_HOME}

WORKDIR ${BOT_HOME}

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY bot.py .
