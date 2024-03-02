from dotenv import load_dotenv
load_dotenv()
import openai
import json
import os
import argparse
import random
random.seed(42)
from tqdm import tqdm
import time
import tiktoken
from nltk import sent_tokenize, wordpunct_tokenize


# Set up your OpenAI API credentials
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or ""
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or ""
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME") or "gpt-35-turbo"
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-35-turbo"

OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION") or "2023-07-01-preview"

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com/"
openai.api_version = OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY

CONTEXT_LENGTH = 16384
COST_PER_1k_TOKENS = 0.0015


def generate_response(messages):
    try:
        completion = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0,
            stop=None,
        )
    except Exception as e:
        print("Error generating response:", e)
        return ""
    # check if completion has content[0].message.content attribute
    if hasattr(completion, "choices") and len(completion.choices) > 0 and hasattr(completion.choices[0], "message") and hasattr(completion.choices[0].message, "content"):
        return completion.choices[0].message.content
    else:
        return ""

def get_templates(categories, title, section):
    categories_enumerated = "\n".join([f"- {cat}" for idx, cat in enumerate(categories)])
    system_template = f'''Your task is to classify sections of a news article into specific categories. These categories encompass a range of topics, such as artificial intelligence, autonomous transport, sustainable fabrics, and more. However, it can be challenging to clearly differentiate between these categories at times, as some articles may cover multiple topics. For instance, artificial intelligence might play a significant role in autonomous driving, drones, and various other applications. However, an article should only be labeled as "artificial intelligence" if this topic is the primary focus and not just a supporting element for another topic. A similar example can be made with "3D Printed Clothes" and "Sustainable fabrics". These are the available categories:
    {categories_enumerated}'''
    prompt_template = f'''Please categorize the following article segment by selecting the most appropriate category from the provided list. Your response should only include the category name. If you are uncertain, please reply with "unsure". If you believe the section lacks relevant information for any category, respond with "irrelevant". Do not use any other responses.
    Article title: {title}
    Segment: {section}'''
    return system_template, prompt_template


def split_article_into_sections(article, section_length=3, by="sentences"):
    # get the first three sentences in the article
    raw_sections = []
    if by == "sentences":
        tokens = sent_tokenize(article["text"])

    elif by == "words":
        tokens = wordpunct_tokenize(article["text"])
    else:
        raise ValueError("by must be either 'sentences' or 'words'")
    sections = [tokens[i:i + section_length] for i in range(0, len(tokens), section_length)]
    # join sections
    sections = [" ".join(s) for s in sections]
    sections_formatted = []
    for idx, s in enumerate(sections):
        section_formatted = {"text": s, "article_id": article["id"], "sentence_id": idx, "title": article["title"]}
        sections_formatted.append(section_formatted)
    return sections_formatted


def truncate_messages(messages):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_length = 0
    for msg in messages:
        total_length += len(encoder.encode(msg['content']))
    if total_length >= CONTEXT_LENGTH - 100:
        messages[-1]['content'] = encoder.decode(encoder.encode(messages[-1]['content'])[:(CONTEXT_LENGTH-total_length)-100])
    return messages, total_length


def classify_section(sec, categories):
    first_response_is_invalid = False
    second_response_is_invalid = False
    system_template, prompt_template = get_templates(categories, sec["title"], sec["text"])
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": prompt_template}
    ]
    messages, tokens = truncate_messages(messages)
    response = generate_response(messages).strip().replace('.', '').lower()
    second_response = response
    if response not in categories and response not in ['unsure', 'irrelevant']:
        first_response_is_invalid = True
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": 'Your response is not valid. It does not match any of the provided categories, "unsure" or "irrelevant". Please try again.'})
        second_response = generate_response(messages).replace('.', '').lower()
        second_response_is_invalid = second_response not in categories and second_response not in ['unsure', 'irrelevant']
    return response, second_response, first_response_is_invalid, second_response_is_invalid, tokens


def load_raw_articles(filename):
    articles = []
    with open(filename) as file:
        json_list = list(file)
        for json_str in json_list:
            article = json.loads(json_str)
            articles.append(article)
    return articles


def run():
    # classify sections
    predictions = []
    progressbar = tqdm(total=num_sections, position=0, leave=True)
    total_tokens = 0
    start_time = time.time()
    failed_predictions = []
    for article_id, article_sections in sections_to_classify.items():
        if article_id < last_article_id:
            progressbar.update(len(article_sections))
            continue
        elif article_id == last_article_id:
            article_sections = article_sections[last_sentence_id + 1:]
            if len(article_sections) == 0:
                progressbar.update(len(article_sections))
                continue
        progressbar.set_description(f'{article_id} - {article_sections[0]["title"]}')
        for idx, section in enumerate(article_sections):
            prediction, second_prediction, response_is_invalid, second_response_is_invalid, tokens = classify_section(
                section, all_categories)
            if prediction == "":
                failed_predictions.append(section)
                continue
            prediction_formatted = {
                "text": section,
                "label": prediction,
                "label2": second_prediction,
                "spans": [],
                "article_id": article_id,
                "sentence_id": idx,
                "answer": "",
                "priority": None,
                "score": None,
                "title": section["title"],
                "gpt_first_response_is_invalid": response_is_invalid,
                "gpt_second_response_is_invalid": second_response_is_invalid,
            }
            total_tokens += tokens
            seconds_elapsed = time.time() - start_time
            estimated_tokens_per_minute = total_tokens / (seconds_elapsed / 60)
            current_cost = (total_tokens / 1000) * COST_PER_1k_TOKENS
            estimated_cost = current_cost * progressbar.total / (progressbar.n + 1)
            if (seconds_elapsed > 60.0) and (estimated_tokens_per_minute > 100000.0):
                time.sleep(30)
            progressbar.set_postfix({
                "tokens": total_tokens,
                "est_tokens_per_minute": str(round(estimated_tokens_per_minute, 2)),
                "cost": f"€{round(current_cost, 2)}",
                "est_cost": f"€{round(estimated_cost, 2)}"
            })
            predictions.append(prediction_formatted)
            progressbar.update(1)
        # save all intermediate results
        with open('../data/raw_data_annotated_with_chatgpt.json', 'w') as outfile:
            json.dump(predictions, outfile)
    return predictions, failed_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh_start', action='store_true')
    args = parser.parse_args()
    # get raw data
    raw_articles = load_raw_articles('../data/raw_data.jsonl')
    # where climate scanner True
    raw_articles = [a for a in raw_articles if a["climate_scanner"]]
    all_categories = set([a["category"].lower().replace(" ", "_") for a in raw_articles])

    # drop duplicates by title
    raw_articles = [article for idx, article in enumerate(raw_articles) if article["title"] not in [a["title"] for a in raw_articles[idx+1:]]]
    # re-id articles
    for idx, article in enumerate(raw_articles):
        article["old_id"] = article["id"]
        article["id"] = idx
    # save to json
    with open('../data/raw_data_no_duplicates.jsonl', 'w') as outfile:
        for article in raw_articles:
            json.dump(article, outfile)
            outfile.write('\n')
    # setup sections to classify
    sections_to_classify = {article["id"]: split_article_into_sections(article, section_length=3, by="sentences") for article in raw_articles}
    # check if the file already exists
    if args.fresh_start:
        os.remove('../data/raw_data_annotated_with_chatgpt.json')
    elif os.path.isfile('../data/raw_data_annotated_with_chatgpt.json'):
        # load file
        with open('../data/raw_data_annotated_with_chatgpt.json') as file:
            predictions_so_far = json.load(file)
            last_article_id = predictions_so_far[-1]["article_id"]
            last_sentence_id = predictions_so_far[-1]["sentence_id"]
            print("Found existing file, continuing from last article id:", last_article_id, "and last sentence id:", last_sentence_id)
    else:
        print("No existing file found do delete or continue from, starting from scratch")
    num_sections = sum([len(sections) for _, sections in sections_to_classify.items()])
    predictions, failed_predictions = run()


