import os
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "app"))
from app.pipelines.data_preprocessing import chunk_text, load_text
from app.pipelines.embeddings import compute_embeddings
from app.pipelines.rag_pipeline import build_rag_index, answer_query


@pytest.fixture(scope="session")
def article_text():
    """Загружаем исходный текст один раз на всю сессию тестов"""
    url = "https://blog.dzencode.com/ru/illyuziya-kachestva-vash-sayt-idealen-pozdravlyaem-vy-tolko-chto-sozhgli-byudzhet/"
    return load_text(url)


def test_cleaning_and_chunking(article_text):
    """Проверяем, что текст загружается и делится на чанки"""
    chunks = chunk_text(article_text, chunk_size=300, semantic=False)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, (str, dict)) for c in chunks)
    assert all(isinstance(c["text"], str) for c in chunks)


def test_embeddings(article_text):
    """Проверяем, что эмбеддинги считаются"""
    chunks = chunk_text(article_text, chunk_size=300, semantic=False)
    model_name = "all-MiniLM-L6-v2"
    chunks_with_embeddings = compute_embeddings(chunks, model_name)
    assert len(chunks_with_embeddings) == len(chunks)
    # проверим, что эмбеддинги не пустые
    assert "embedding" in chunks_with_embeddings[0]
    assert len(chunks_with_embeddings[0]["embedding"]) > 100


def test_rag_index_and_query(article_text):
    """Проверяем поиск ответа"""
    chunks = chunk_text(article_text, chunk_size=300, semantic=False)
    model_name = "all-MiniLM-L6-v2"
    chunks_with_embeddings = compute_embeddings(chunks, model_name)
    index = build_rag_index(chunks_with_embeddings)

    query = "Что значит 'Иллюзия качества'?"
    answer = answer_query(index, query)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "качест" in answer.lower()  # проверяем по корню слова
