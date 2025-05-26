import os
import shutil
from typing import List, Dict, Any

from celery import Celery
from celery.signals import worker_process_init

from tools import download_audio, set_models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

text_splitter: RecursiveCharacterTextSplitter
models: Dict[str, Any]
qdrant_client: QdrantClient
vector_store: QdrantVectorStore

app = Celery(main="celery_text_processing", broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))


@worker_process_init.connect()
def setup(**kwargs):
    global text_splitter, models, qdrant_client, vector_store

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    models = set_models()

    qdrant_client = QdrantClient(url='http://qdrant', port=6333)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="videos",
        embedding=models["embedding_model"],
    )

@app.task
def get_text_from_videos(urls: List[str]) -> None:
    output_dir = "downloads"
    paths = download_audio(urls, output_dir=output_dir)

    for path in paths:
        segments, _ = models["stt_model"].transcribe(path, vad_filter=True)
        text = " ".join(segment.text for segment in segments)
        logger.info("audio segmented")
        chunks = text_splitter.split_text(text)
        logger.info("audio chunked")
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"video": path[:-4]} for _ in chunks],
            batch_size=2,
        )
        logger.info("added to vdb")
    shutil.rmtree(output_dir)