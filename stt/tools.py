import os
from typing import List, Dict
import yt_dlp
import logging

from faster_whisper import WhisperModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_snapshot_dir() -> str:
    return os.path.join(
        'models',
        'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo',
        'snapshots',
        os.listdir('./models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots')[0]
    )

def set_models(models: Dict) -> None:
    models["stt_model"] = WhisperModel(
        model_size_or_path=f'./{get_snapshot_dir()}',
        local_files_only=True,
        device="cuda",
        num_workers=8,
    )
    models["embedding_model"] = OllamaEmbeddings(
        model=os.getenv("embedding_model", "jeffh/intfloat-multilingual-e5-large-instruct:f16"),
        base_url=os.getenv("ollama_url", "http://ollama:11434"),
        num_ctx=512,
    )
    models["llm"] = OllamaLLM(
        model=os.getenv("llm", "owl/t-lite:q4_0-instruct"),
        base_url=os.getenv("ollama_url", "http://ollama:11434"),
        temperature=0.03
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
    logger.info("in_func")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    logger.info(f"downloaded {os.listdir(output_dir)}")
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp3")]




