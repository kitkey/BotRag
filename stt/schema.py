from typing import Dict

import pydantic
from dataclasses import dataclass

from celery import Celery
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


@dataclass
class Services:
    qdrant: QdrantClient
    celery: Celery
    models: Dict[str, any]
    vector_store: QdrantVectorStore
    mmr_retriever: any
    generator_prompt: ChatPromptTemplate
    retriever_prompt: str
