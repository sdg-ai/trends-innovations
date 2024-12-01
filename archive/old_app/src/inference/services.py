import nltk
from .models import Article, ArticleChunk

def compute_sentence_chunks(article: Article):
    sentences = nltk.sent_tokenize(article.text)
    chunks = [ArticleChunk(
        id=str(i),
        text="".join(sentences[i:i+3]),
        article_id=article.id,
        start_pos=len("".join(sentences[:i])),
        end_pos=len("".join(sentences[:i+3]))
    ) for i in range(0, len(sentences), 3)]
    return chunks



