FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y ffmpeg python3-pip python3-dev
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV APP_HOME=/home/app/

RUN mkdir -p ${APP_HOME}

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', download_root='/models', device='cpu')"

COPY app.py tools.py ./