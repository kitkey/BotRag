import os
from contextlib import asynccontextmanager

from typing import List, Dict
import uvicorn
from celery import Celery
from fastapi import FastAPI, Body

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.prompts import ChatPromptTemplate
from tools import set_models
from schema import Services
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_services: Services

def build_services() -> Services:
    qdrant_client = QdrantClient(url='http://qdrant', port=6333)

    celery = Celery(main="celery_text_processing", broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))

    models = {}
    set_models(models)

    try:
        qdrant_client.create_collection(
            collection_name="videos",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except Exception:
        logger.info('exists')

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="videos",
        embedding=models["embedding_model"],
    )

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'fetch_k': 20, 'k': 5, 'lambda_mult': 0.25}
    )

    with open('prompts/generator_prompt.txt', 'r') as f:
        generator_text_prompt = f.read()

    with open('prompts/retriever_prompt.txt', 'r') as f:
        retriever_prompt = f.read()

    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", generator_text_prompt),
        ("user", "Контекст:\n{context}\n\nВопрос: {question}")
    ])

    return Services(
        qdrant=qdrant_client,
        celery=celery,
        models=models,
        vector_store=vector_store,
        mmr_retriever=mmr_retriever,
        generator_prompt=generator_prompt,
        retriever_prompt=retriever_prompt
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_services
    rag_services = build_services()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/get_text")
async def get_text_from_videos(urls: List[str] = Body()) -> Dict:
    rag_services.celery.send_task("redis_tasks.get_text_from_videos", args=[urls])
    logger.info("sent task")
    return {"status": "queued"}


@app.post("/retrieve_information")
async def get_answer_for_query(query: str = Body()) -> str:
    # emb_query = await models["embedding_model"].aembed_query(prompt_retrieve)
    docs = await rag_services.mmr_retriever.ainvoke(input=rag_services.retriever_prompt.format(query=query))
    context = "\n\n".join(d.page_content for d in docs)
    logger.info("prompt invoking")
    chain = rag_services.generator_prompt | rag_services.models["llm"]
    ans = await chain.ainvoke({"context": context, "question": query})
    return ans

if __name__ == "__main__":
    uvicorn.run(app, port=8000)