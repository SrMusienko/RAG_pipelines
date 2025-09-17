import os
import json
import hashlib
from datetime import datetime

from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer



def build_rag_index(
    chunks: list[dict],
    collection_name: str = "dzencode_articles",
    host: str = "localhost",
    port: int = 6333,
) -> QdrantClient:
    
    client = QdrantClient(host=host, port=port)
    vector_size = len(chunks[0]["embedding"])
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    points = [
        PointStruct(
            id=chunk["id"],
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                "topic": chunk.get("topic"),
                "project": chunk.get("project"),
                "lang": chunk.get("lang"),
            },
        )
        for chunk in chunks
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print(f"В Qdrant загружено {len(points)} чанков в коллекцию '{collection_name}'")
    return client


def answer_query(
    client: QdrantClient,
    query: str,
    collection_name: str = "dzencode_articles",
    top_k: int = 3
) -> str:
    
    query_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = query_model.encode([query])[0]
    collection_name = "dzencode_articles"

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k
    )

    answer = " ".join([hit.payload["text"] for hit in result.points])
    return answer


def llm_answer_query(query: str, data: str) -> str:

    load_dotenv()
    API_KEY = os.getenv('API_KEY')

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "Ты помощник, который отвечает на вопросы на русском."},
        {"role": "system", "content": f"Информация из векторной БД: {data}"},
        {"role": "user", "content": query}
    ]
    data = {
        "model": "llama-3.1-8b-instant",  
        "messages": messages,
        "max_tokens": 300
    }

    response = requests.post(ENDPOINT, headers=headers, json=data)
    answer = response.json()

    return answer["choices"][0]["message"]["content"]


def log_query(query: str, answer: str, log_file: str = "logs/query_log.json"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            query_log = json.load(f)
    else:
        query_log = {"queries": []}
    
    query_entry = {
        "timestamp": str(datetime.now()),
        "query": query,
        "answer": answer
    }
    query_log["queries"].append(query_entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(query_log, f, ensure_ascii=False, indent=2)
    
    print(f"Запрос записан в лог: {log_file}")


def log_rag_version(rag_file_path: str, version_log_path: str = "version_log.json"):

    if not os.path.exists(rag_file_path):
        print(f"RAG-файл не найден: {rag_file_path}")
        return
    
    dir_name = os.path.dirname(version_log_path)
    if dir_name:  # создаем только если это не пустая строка
        os.makedirs(dir_name, exist_ok=True)

    if os.path.exists(version_log_path):
        with open(version_log_path, "r", encoding="utf-8") as f:
            version_log = json.load(f)
    else:
        version_log = {"versions": []}

    with open(rag_file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if version_log["versions"]:
        last_version = version_log["versions"][-1]
        if last_version["file_hash"] == file_hash:
            print(f"⏭️  RAG-файл не изменился, пропускаем запись: {rag_file_path}")
            return

    version_entry = {
        "timestamp": str(datetime.now()),
        "rag_file": rag_file_path,
        "file_size": os.path.getsize(rag_file_path),
        "file_hash": file_hash,
        "version_id": f"v{len(version_log['versions']) + 1}"
    }
    version_log["versions"].append(version_entry)

    with open(version_log_path, "w", encoding="utf-8") as f:
        json.dump(version_log, f, ensure_ascii=False, indent=2)

    print(f"✅ Версия RAG-файла записана: {version_log_path}")