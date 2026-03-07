from datetime import date

from pydantic import BaseModel


class NewsArticle(BaseModel):
    title: str
    text: str
    date: date
    source: str | None = None
    language: str = "en"


class QAPair(BaseModel):
    question: str
    expected_answer: str


class Fact(BaseModel):
    fact_id: str
    description: str
    source_article_title: str
    questions: list[QAPair]
