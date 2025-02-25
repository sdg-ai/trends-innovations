from dotenv import load_dotenv

load_dotenv()
import openai
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
import json
import os
import random
from tqdm import tqdm
import tiktoken
import numpy as np

random.seed(42)
system_template = "You are a language model with expertise in {domain}."
prompt_template = """
Please rewrite the following article while incorporating fresh category-specific vocabulary, phrasing, and domain knowledge. The goal is to maintain the integrity of the original category and ensure that the rewritten article remains relevant to the domain of {domain}. Please include additional domain-specific keywords, recent trends, or emerging topics to be highlighted in the rewritten article, enhancing its focus on {domain}. Avoid providing fake information or wrong statements; prioritize accuracy and reliability. Feel free to explore diverse writing styles as long as they suit the subject matter."

Assigned Category:
{domain}

Title:
{title}

Article:
{article}
"""

# Set up your OpenAI API credentials
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or ""
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or ""
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME") or "gpt-35-turbo"
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-35-turbo"

OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION") or "2023-06-01-preview"

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com/"
openai.api_version = OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY

NUM_GENERATIONS_PER_CATEGORY = 50
MIN_FRACTION_LEFT_FOR_TESTING = 0.3

CONTEXT_LENGTH = 4096 if AZURE_OPENAI_MODEL_NAME == "gpt-3.5-turbo" else 16384


def get_article_ids_from_annotated_data():
    ids = []
    for f in [f for f in os.listdir("../data/annotated_data") if f.endswith('jsonl')]:
        with open(f'../data/annotated_data/{f}', 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            ids.append(result["meta"]["doc_id"])
    return set(ids)


def load_articles(filename):
    articles = []
    annotated_article_ids = get_article_ids_from_annotated_data()
    with open(filename) as file:
        json_list = list(file)
        for json_str in json_list:
            article = json.loads(json_str)
            if article["id"] in annotated_article_ids and "bbc" not in article["category"]:
                articles.append(article)
    return articles


def randomly_sample_articles_from_each_category(articles, category_subset=None):
    categories = set([article["category"] for article in articles]) if not category_subset else category_subset
    sampled_articles = {}
    for category in categories:
        category_articles = [article for article in articles if article["category"] == category]
        num_samples = int(np.floor(len(category_articles) * (1 - MIN_FRACTION_LEFT_FOR_TESTING)))
        sampled_articles[category] = random.sample(category_articles, num_samples)
    return sampled_articles


def generate_response(prompt):
    try:
        completion = openai.ChatCompletion.create(engine=AZURE_OPENAI_DEPLOYMENT_NAME, messages=prompt, temperature=1,
                                                  stop=None, presence_penalty=1)
    except Exception as e:
        print("Error generating response:", e)
        return ""
    return completion.choices[0].message.content


def langchain_chatmsgs_to_openaimsgs(messages):
    openai_msgs = []
    for msg in messages:
        new_msg = {}
        if msg.type == "system":
            new_msg["role"] = "system"
        elif msg.type == "human":
            new_msg["role"] = "user"
        elif msg.type == "ai":
            new_msg["role"] = "assistant"
        new_msg["content"] = msg.content
        openai_msgs.append(new_msg)
    return openai_msgs


def construct_input(article, category, title):
    messages = [SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(prompt_template)]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chat_prompt = chat_prompt.format_prompt(domain=category, title=title, article=article)
    messages = langchain_chatmsgs_to_openaimsgs(chat_prompt.to_messages())
    return messages


def truncate_article(article_text, article_id):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    e = encoder.encode(article_text)
    if len(e) >= CONTEXT_LENGTH / 2:
        print(f"Truncating article [{article_id}] to {CONTEXT_LENGTH / 2} tokens")
        return encoder.decode(e[:CONTEXT_LENGTH // 2])
    return article_text


def main():
    filename = '../data/raw_data.jsonl'
    articles = load_articles(filename)
    selected_categories = ["Biking", "Drone", "Car-sharing"]
    sampled_articles = randomly_sample_articles_from_each_category(articles, selected_categories)
    progress_bar = tqdm(total=len(selected_categories) * NUM_GENERATIONS_PER_CATEGORY, desc="Processing",
        unit="iteration")
    generated_articles = {}
    for category in selected_categories:
        generated_articles[category] = []
        while len(generated_articles[category]) < NUM_GENERATIONS_PER_CATEGORY:
            for article_id, title, text in ((a["id"], a["title"], a["text"]) for a in sampled_articles[category]):
                text = truncate_article(text, article_id)
                messages = construct_input(text, category, title)
                response = generate_response(messages)
                if response != "":
                    generated_articles[category].append({"id": article_id, "text": response})
                    progress_bar.update(1)
                if len(generated_articles[category]) >= NUM_GENERATIONS_PER_CATEGORY:
                    break

    # dump generated articles to json
    with open(f'./generated_articles_{NUM_GENERATIONS_PER_CATEGORY}.json', 'w') as outfile:
        json.dump(generated_articles, outfile)


if __name__ == '__main__':
    main()
