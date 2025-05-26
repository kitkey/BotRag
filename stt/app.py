import os

from typing import List, Dict
import uvicorn
from celery import Celery
from fastapi import FastAPI, Body

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from tools import set_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


models = {}

prompt = ChatPromptTemplate.from_messages([
        ("system", """
    Ты — помощник по поиску информации в документах.
    Отвечай только по контексту. Если ответ не найден, говори: "Информации в контексте нет."
    
    Правила:
    - Пиши кратко и по делу, без воды.
    - Не выдумывай. Не используй внешние знания.
    - Пиши только на русском.
    - Не вставляй формулы, LaTeX и спецсимволы.
    - Переводы строки должны быть настоящими (LF, U+000A), а не \\n.
    - Источники, если есть, — в виде нумерованного списка в конце (без ссылок, только номер документа/источника из контекста).
    
    Будь сдержан, точен и немногословен.
    """),
        ("user", """
    Контекст:
    {context}
    ---
    Теперь ответь на следующий вопрос.

    Вопрос: {question}
    """)
    ])


qdrant_client = QdrantClient(url='http://qdrant', port=6333)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
celery = Celery(main="celery_text_processing", broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))

has_warmed_up = False

set_models(models)
models["embedding_model"].embed_query("initial")
models["llm"].invoke("initial")

if not qdrant_client.collection_exists(collection_name="videos"):
    qdrant_client.create_collection(
        collection_name="videos",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="videos",
    embedding=models["embedding_model"],
)

app = FastAPI()

@app.post("/get_text")
async def get_text_from_videos(urls: List[str] = Body()) -> Dict:
    celery.send_task("redis_tasks.get_text_from_videos", args=[urls])
    logger.info("sent task")
    return {"status": "queued"}


@app.post("/retrieve_information")
async def get_answer_for_query(query: str = Body()) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    prompt_retrieve  = f'Instruct: {task_description}\nQuery: {query}'
    emb_query = await models["embedding_model"].aembed_query(prompt_retrieve)

    retrieve_ = qdrant_client.query_points(collection_name='videos', query=emb_query, limit=10)
    logger.info("prompt invoking")
    chain = prompt | models["llm"]
    ans = await chain.ainvoke({"context": retrieve_, "question": query})

    return ans

if __name__ == "__main__":
    uvicorn.run(app, port=8000)