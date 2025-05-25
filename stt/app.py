import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from typing import List
import uvicorn
from fastapi import FastAPI, Body


from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tools import download_audio, set_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


models = {}

prompt = ChatPromptTemplate.from_messages([
        ("system", """
    Ты чат-ассистент, помогающий находить информацию в документах.
    Используй только представленный контекст для ответа на вопрос.
    Отвечай кратко и по делу, только на заданный вопрос.
    Если ответ нельзя получить из контекста — не выдумывай и скажи, что информации нет.
    Когда отвечаешь:
    - Пиши обычным текстом на русском.
    - Используй Markdown-заголовки и **жирный** текст.
    - Ты должен ставить **настоящие** символы перевода строки (LF, U+000A), а не печатать последовательность «\\n».
    - Не вставляй LaTeX формулы!! Это очень важно. Формулы пиши обычными символами, никаких \\
    - Отвечай кратко, по делу
    - Укажи источники релевантной информации, если такие будут, после своего ответа в виде нумерованного списка
    Поменьше вспомогательных символов, текст должен быть хорошо читаем и понятен.
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_models(models)
    await models["embedding_model"].aembed_query("initial")
    await models["llm"].ainvoke("initial")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/get_text")
async def get_text_from_videos(urls: List[str] = Body()) -> None:
    output_dir = "downloads"
    paths = download_audio(urls, output_dir=output_dir)


    logger.info("Url: %s", urls)
    for path in paths:
        segments, _ = models["stt_model"].transcribe(path, vad_filter=True)
        text = " ".join(segment.text for segment in segments)

        chunks = text_splitter.split_text(text)

        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"video": path[:-4]} for _ in chunks],
            batch_size = 2,
        )

    shutil.rmtree(output_dir)


@app.post("/retrieve_information")
async def get_answer_for_query(query: str = Body()) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    prompt_retrieve  = f'Instruct: {task_description}\nQuery: {query}'
    emb_query = await models["embedding_model"].aembed_query(prompt_retrieve)

    retrieve_ = qdrant_client.query_points(collection_name='videos', query=emb_query, limit=10)

    chain = prompt | models["llm"]
    ans = await chain.ainvoke({"context": retrieve_, "question": query})

    return ans

if __name__ == "__main__":
    uvicorn.run(app, port=8000)