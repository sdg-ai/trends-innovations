from pydantic import BaseModel
from typing import List


class Article(BaseModel):
    id: str
    text: str

class ArticleChunk(BaseModel):
    id: str
    text: str
    article_id: str
    start_pos: int
    end_pos: int


class Prediction(BaseModel):
    class_label: str
    probability: float


class ChunkResult(BaseModel):
    chunk: ArticleChunk
    class_probabilities: List[Prediction]


class ArticleResult(Prediction):
    article_id: str
    chunk_predictions: List[ChunkResult]


class InferenceResponse(BaseModel):
    predictions: List[ArticleResult]

class InferenceRequest(BaseModel):
    articles: List[Article]
