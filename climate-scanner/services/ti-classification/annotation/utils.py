from dataclasses import dataclass
from typing import List
from nltk.tokenize import sent_tokenize
import logging
# configure logger to got to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# configure logger to got to stdout
import os
import json
import ast
import functools
import time


@dataclass
class ArticleSection:
    id: int
    article_id: int
    category: str
    text: str
    keywords: List[str]
    annotation: str = ""
    reasoning: str = ""


@dataclass
class Article:
    id: int
    title: str
    text: str
    category: str
    sections: List[ArticleSection]


def load_raw_data(data_dir: str):
    """ 
    Loads the raw data from the given directory. 
    The raw data is a jsonl file where each line is a json object i.e. one article.

    Args:
        data_dir (str): The directory where the raw data is stored.

    Returns:
        List[Article]: A List of Article objects representing the raw data.
    """
    print(os.getcwd())
    file_path = os.path.join(data_dir, "raw_data.jsonl")
    articles = []
    with open(file_path) as file:
        json_list = list(file)
        for json_str in json_list:
            article = json.loads(json_str)
            if article["climate_scanner"] is False:
                continue
            if article["category"] == "3D printed apparel":
                article["category"] = "3D printed clothes"
            articles.append(
                preprocess_raw_article(
                    Article(id=article["id"],
                            title=article["title"],
                            text=article["text"],
                            category=article["category"],
                            sections=[])))
    logging.info("Loaded raw data with articles [%d]", len(articles))
    # remove duplicated articles based on title
    articles = [
        article for idx, article in enumerate(articles)
        if article.title not in [a.title for a in articles[idx + 1:]]
    ]
    logging.info("Removed duplicated articles based on title [%d]",
                 len(articles))
    # check if there are duplicate ids by count
    if len(articles) != len(set([article.id for article in articles])):
        raise logging.error("There are duplicate ids")
    return pre_process_categories(articles)


def preprocess_raw_article(article: Article, section_length: int = 3):
    """Splits the text of an article into sections.

    Args:
        article (Article): The article to preprocess.
        section_length (int, optional): The length of each section as a number of sentences. Defaults to 3.

    Returns:
        Article: The article with the sections added.
    """
    tokens = sent_tokenize(article.text)
    sections = [
        tokens[i:i + section_length]
        for i in range(0, len(tokens), section_length)
    ]
    article.sections = [
        ArticleSection(
            id=idx,
            article_id=article.id,
            category="",
            text=" ".join(s),
            keywords=[],
        ) for idx, s in enumerate(sections)
    ]
    return article


def load_human_annotated_data(data_dir: str):
    """
    Loads the human annotated data from the given directory. Every category has its own jsonl file. Every line of the jsonl file corresponds to a section of the article.

    Args:
        data_dir (str): The directory where the human annotated data is stored.

    Returns:
        List[Article]: A List of Article objects representing the human annotated data.
    """
    data_dir = os.path.join(data_dir, "human_annotated_data")
    final_articles = []
    for f in [f for f in os.listdir(data_dir) if f.endswith('jsonl')]:
        with open(f'{data_dir}/{f}', 'r') as json_file:
            json_list = list(json_file)
        articles = []
        article_sections = []
        for json_str in json_list:
            result = json.loads(json_str)
            if result["meta"]["doc_id"] not in [
                    article.id for article in articles
            ]:
                articles.append(
                    Article(id=result["meta"]["doc_id"],
                            title=result["meta"]["title"],
                            text="",
                            category="",
                            sections=[]))
            try:
                keywords = [s["text"] for s in result["spans"]]
            except:
                # logging.warning(f"Could not parse keywords for article {result['meta']['doc_id']}")
                keywords = None
            article_sections.append(
                ArticleSection(article_id=result["meta"]["doc_id"],
                               id=result["meta"]["sent_id"],
                               category=result["label"],
                               text=result["text"],
                               keywords=keywords if keywords else [],
                               annotation=result["answer"]))
        for article in articles:
            article.sections = [
                s for s in article_sections if s.article_id == article.id
            ]
            article.text = " ".join([s.text for s in article.sections])
            # majority vote on category
            article.category = max(
                set([s.category for s in article.sections]),
                key=lambda x: [s.category for s in article.sections].count(x))
        final_articles.extend(articles)
    logging.info("Loaded human annotated data with articles [%d]",
                 len(final_articles))
    # check that there are no duplicate ids
    if len(final_articles) != len(
            set([article.id for article in final_articles])):
        raise logging.error("There are duplicate ids")
    return pre_process_categories(final_articles)


def pre_process_categories(articles: List):
    """Standardizes the category names.

    Args:
        articles (List): List of articles.

    Returns:
        List: List of articles with standardized categories.
    """
    standardize_category = lambda x: x.lower().replace(" ", "_").replace(
        "-", "_").replace("&", "and").replace("/", "_")
    for article in articles:
        article.category = standardize_category(article.category)
        for section in article.sections:
            section.category = standardize_category(section.category)
    return articles


def match_raw_data_with_human_annotated_data(raw_data: List,
                                             human_annotated_data: List):
    """
    Matches the raw data to the human annotated data.

    Args:
        raw_data (List): List of raw articles.
        human_annotated_data (List): List of human annotated articles.

    Returns:
        dict: Dictionary mapping raw article ids to human annotated article ids.
    """
    raw_to_annotation_key = {}
    for article in raw_data:
        for h_article in human_annotated_data:
            if article.title == h_article.title and article.id == h_article.id:
                raw_to_annotation_key[article.id] = h_article.id
                break
    return raw_to_annotation_key


def get_categories(articles: List):
    return list(set([article.category for article in articles]))


def retry_on_condition(max_retries=3, retry_condition=None, delay=1):
    """
    Retries a function if the retry_condition is True.

    Args:
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
        retry_condition (_type_, optional): The condition to retry on. Defaults to None.
        delay (int, optional): The delay between retries. Defaults to 1.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    section = func(*args, **kwargs)
                    # If retry_condition is provided, check if we should retry
                    if retry_condition and retry_condition(section):
                        retries += 1
                        if retries < max_retries:
                            time.sleep(delay)
                            continue
                    return section
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"Failed after {max_retries} retries: {str(e)}")
                        section.category = "unsure"
                        section.reasoning = str(e)
                    time.sleep(delay)
            return section
        return wrapper
    return decorator