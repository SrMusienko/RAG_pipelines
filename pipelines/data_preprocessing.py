from datetime import datetime
import json
import os
import requests
from bs4 import BeautifulSoup

def load_text(
    url: str,
    save_html_path: str = "artifacts/article.html",
    save_text_path: str = "artifacts/cleaned_article.txt",
    save_article_path: str = "artifacts/article.txt",  # артефакт с метаданными
    timeout: int = 10,
) -> str:
    """
    Загружает страницу по url, сохраняет raw HTML и текст статьи.
    Дополнительно сохраняет:
      - cleaned_article.txt (основной текст без мусора)
      - uncertain_data.txt (короткие строки/мусор)
      - article.txt (текст + метаданные)
    Возвращает article_text (str).
    """
    os.makedirs(os.path.dirname(save_html_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(save_text_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(save_article_path) or ".", exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-pipeline/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    # Save raw HTML
    with open(save_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    article_tag = soup.find("article")
    lines = []

    if article_tag:
        for tag in article_tag.find_all(["h1","h2","h3","h4","h5","h6","p","li"]):
            text = tag.get_text(" ", strip=True)
            if text:
                lines.append(text)
        article_text = "\n".join(lines)
    else:
        article_text = soup.get_text(" ", strip=True)

    # Save cleaned article
    with open(save_text_path, "w", encoding="utf-8") as f:
        f.write(article_text)

    # Create metadata
    metadata = {
        "url": url,
        "language": "ru",  # можно позже автоматизировать определение
        "date": datetime.now().isoformat(),
        "topic": lines[0] if lines else "",  # можно взять первый заголовок
    }

    # Save article with metadata
    article_with_meta = {
        "metadata": metadata,
        "text": article_text
    }
    with open(save_article_path, "w", encoding="utf-8") as f:
        json.dump(article_with_meta, f, ensure_ascii=False, indent=2)

    return article_text

