from fastapi import APIRouter, status
from .models import InferenceRequest, InferenceResponse
from .services import compute_sentence_chunks
from .. import MODEL

router = APIRouter()

@router.post("/", status_code=status.HTTP_200_OK)
def predict(request: InferenceRequest) -> InferenceResponse:
    """
    receives a list of articles and returns a list of predictions, one prediction for each sentence chunk
    :param request: a list of articles
    :return:  a list of predictions, one prediction for each 3 sentences chunk
    """
    article_level_predictions = []
    for article in request.articles:
        chunks = compute_sentence_chunks(article)
        article_level_predictions.append(MODEL.predict(chunks))
    return InferenceResponse(predictions=article_level_predictions)
