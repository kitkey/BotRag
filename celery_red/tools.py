import os
from typing import List, Dict

import yt_dlp
import logging

from faster_whisper import WhisperModel
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_snapshot_dir() -> str:
    return os.path.join(
        'models',
        'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo',
        'snapshots'
    )


def download_audio(urls: List[str], output_dir: str = "downloads") -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.mp3'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    logger.info(f"downloaded {os.listdir(output_dir)}")
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp3")]


def set_models() -> Dict:
    models = {"embedding_model": OllamaEmbeddings(
        model=os.getenv("OLLAMA_RETRIEVER", "jeffh/intfloat-multilingual-e5-large-instruct:f16"),
        base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
        num_ctx=512,
    )}
    snapshot_dir = get_snapshot_dir()
    if os.path.isdir(snapshot_dir) and os.listdir(snapshot_dir):
        snapshot_dir = os.path.join(snapshot_dir, os.listdir('./models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots')[0])
        models['stt_model'] = WhisperModel(
            model_size_or_path=snapshot_dir,
            local_files_only=True,
            device="cuda",
            num_workers=8,
        )
    else:
        models['stt_model'] = WhisperModel(
            model_size_or_path='large-v3-turbo',
            download_root='/models',
            device='cuda',
        )

    return models