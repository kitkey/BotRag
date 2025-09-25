from typing import Dict

from dataclasses import dataclass

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


@dataclass
class Services:
    qdrant: QdrantClient
    models: Dict[str, any]
    vector_store: QdrantVectorStore
    text_splitter: RecursiveCharacterTextSplitter
