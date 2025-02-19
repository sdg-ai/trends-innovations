import os
from utils import (load_human_annotated_data, load_raw_data, get_categories,
                   Article, ArticleSection, retry_on_condition)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
from collections import Counter
import getpass
from dotenv import load_dotenv
import pandas as pd
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
import pickle
import datetime
import random
random.seed(42)

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)


class AnnotationResponse(BaseModel):
    category: str = Field(description="The chosen category of the section")
    reasoning: str = Field(
        description="The reasoning behind the category choice")
    relevant_keywords: list = Field(
        description="The relevant keywords in the section, if applicable")


ANNOTATION_SYSTEM_PROMPT = """
As an expert in article classification, your task is to categorize sections of news articles into specific topics. Follow these guidelines:

1. Choose the most appropriate category from the list below:
<categories>
{categories}
</categories>

2. Focus on the primary topic of the section, even if multiple topics are mentioned.
3. If artificial intelligence, for example, is only a supporting element, it should not be the main category.
4. If you're unsure or the section lacks relevant information for any category, select "unsure".

When you choose a category:
- Provide a brief, clear explanation for your choice.
- Ensure the category name matches exactly with one from the list above.
- Identify and list relevant keywords that support your decision.
Remember: Accuracy in category selection and reasoning is crucial for this task.
"""

ANNOTATION_USER_PROMPT = """
Please categorize the following article section by selecting the most appropriate category from the provided list. If uncertain or if the section lacks relevant information for any category, reply with "unsure." Don't forget to include the reasoning behind your decision and relvant keywords if applicable.
<article-title>{title}</article-title>
<section>
{text}
</section>
"""


def run_simple_annotation(
        data_dir: str,
        num_category_groups: int = 1,
        num_majority_vote: int = 1,
        conflict_resolution_strategy: str = "majority_vote",  # or 'unsure'
        benchmark: bool = True,
        openai_api_key: str = None,
        openai_model: str = "gpt-4",
        azure_openai_api_key: str = None,
        azure_openai_endpoint: str = None,
        azure_openai_deployment: str = "gpt-4",
        azure_openai_api_version: str = "2023-07-01-preview"):
    """
    Run simple annotation process on the dataset.
    """
    # Initialize the appropriate model based on available credentials
    if azure_openai_api_key and azure_openai_endpoint:
        model = AzureChatOpenAI(azure_deployment=azure_openai_deployment,
                            api_version=azure_openai_api_version,
                            temperature=0.0)
    else:
        model = init_chat_model(
            openai_model,
            model_provider="openai",
            temperature=0.0)

    human_annotated_data = load_human_annotated_data(data_dir)
    raw_data = load_raw_data(data_dir)
    raw_data_categories = get_categories(raw_data)
    # choose the data to run on
    if benchmark:
        logging.info("Running in benchmark mode")
        data = human_annotated_data.copy()
    else:
        data = raw_data
    # setup category groups
    # categories are always from raw_data, because there are more and thus more difficult
    category_groups = []
    if num_category_groups > 1:
        random.shuffle(raw_data_categories)
        category_groups = [
            raw_data_categories[
                i:i + int(len(raw_data_categories) / num_category_groups)]
            for i in range(0, len(raw_data_categories),
                           int(len(raw_data_categories) / num_category_groups))
        ]
    else:
        category_groups = [raw_data_categories]
    pb = tqdm(data, total=len(data) * len(category_groups) * num_majority_vote)
    for article in data:
        for category_group in category_groups:
            for _ in range(num_majority_vote):
                annotate_article(article, category_group, model)
                pb.update(1)
    pb.close()
    for article in data:
        conclude_annotation(article, conflict_resolution_strategy)
    if benchmark:
        evaluate_annotation_results(data, human_annotated_data)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"dataset/ai_annotated_data/openai_{openai_model}_{timestamp}.pkl", "wb") as f:
        pickle.dump(data, f)


def annotate_article(article: Article, categories: List[str], model):
    @retry_on_condition(max_retries=3,
                        retry_condition=lambda section: section.category not in
                        list(categories) + ["unsure"])
    def annotate_section(section: ArticleSection):
        messages = [
            SystemMessage(
                ANNOTATION_SYSTEM_PROMPT.format(categories=categories)),
            HumanMessage(
                ANNOTATION_USER_PROMPT.format(title=article.title,
                                              text=section.text)),
        ]
        model_with_tool = model.bind_tools([AnnotationResponse])
        try:
            response = model_with_tool.invoke(messages)
            response_args = response.tool_calls[0]["args"]
            section.category = response_args[
                "category"] if section.category == "" else section.category + "/" + response_args[
                    "category"]
            section.reasoning = response_args["reasoning"]
            section.relevant_keywords = response_args["relevant_keywords"]
            return section
        except Exception as e:
            print(e)
            return section
    thread_pool = ThreadPoolExecutor(max_workers=16)
    sections = thread_pool.map(annotate_section, article.sections)
    article.sections = list(sections)


def conclude_annotation(article: Article,
                        conflict_resolution_strategy: str = "majority_vote"):
    for section in article.sections:
        if "/" in section.category:
            annotations = section.category.split("/")
        else:
            annotations = [section.category]
        if len(annotations) > 1:
            if len(set(annotations)) == 1:
                # votes are unambiguous
                section.category = annotations[0]
            else:
                # there are different votes
                if conflict_resolution_strategy == "majority_vote":
                    counts = Counter(annotations)
                    if not counts:
                        section.category = "unsure"
                        continue
                    most_common = counts.most_common(1)[
                        0]  # Returns (category, count)
                    majority_category, majority_count = most_common
                    majorities = [
                        category for category, count in counts.items()
                        if count == majority_count
                    ]
                    if len(majorities) > 1:
                        section.category = "unsure"
                    else:
                        section.category = majority_category
                elif conflict_resolution_strategy == "unsure":
                    section.category = "unsure"
                else:
                    raise ValueError(
                        "conflict_resolution_strategy must be 'majority_vote' or 'unsure'"
                    )
        else:
            # only one vote
            pass
    # article level category received by majority vote, even if tied
    article.category = Counter([
        section.category for section in article.sections
    ]).most_common(1)[0][0]


def evaluate_annotation_results(ai_annotated_articles: List[Article],
                                human_annotated_articles: List[Article]):
    # sort articles by id
    ai_annotated_articles.sort(key=lambda x: x.id)
    human_annotated_articles.sort(key=lambda x: x.id)
    new_rows = []
    for ai_article, human_article in zip(ai_annotated_articles,
                                         human_annotated_articles):
        assert ai_article.id == human_article.id
        # sort sections by id
        ai_article.sections.sort(key=lambda x: x.id)
        human_article.sections.sort(key=lambda x: x.id)
        for ai_section, human_section in zip(ai_article.sections,
                                             human_article.sections):
            assert ai_section.article_id == human_section.article_id
            new_row = {
                "article_id": ai_section.article_id,
                "section_id": ai_section.id,
                "ai_category": ai_section.category,
                "human_category": human_section.category,
                "ai_reasoning": ai_section.reasoning,
                "human_reasoning": human_section.reasoning,
                "ai_relevant_keywords": ai_section.keywords,
                "human_relevant_keywords": human_section.keywords,
            }
            new_rows.append(new_row)
    results = pd.DataFrame(new_rows)
    results["correct"] = results["ai_category"] == results["human_category"]
    # overall accuracy
    accuracy = results["correct"].mean()
    print("Overall accuracy: %f", accuracy)
    results_per_category = [
        results.loc[results["ai_category"] == category]
        for category in results.human_category.unique()
    ]
    # accuracy per category
    for results_category in results_per_category:
        accuracy = results_category["correct"].mean()
        print(f"Accuracy for category {results_category.iloc[0]['human_category']}: {accuracy:.4f}")
    results.to_csv(f"annotation/results_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", index=False)
