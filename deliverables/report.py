
import os
import re
import sys
sys.path.append(os.getcwd())
import nltk
#nltk.download('punkt_tab')
#nltk.download('stopwords')
import json
import torch
import joblib   
import urllib.parse, urllib.request, json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv('deliverables/.env')
from typing import List, Tuple, Any, Dict
from nltk.corpus import stopwords
from dataclasses import dataclass
from torch.nn.functional import softmax
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from deliverables.entity_networks_master.app.entity_extraction import EntityExtractor


CHECKPOINT = os.getenv('CHECKPOINT', 'default_checkpoint_value')
SEED = os.getenv('SEED', 'default_seed_value')
TRENDSCANNER_ENTITY_EXTRACTION_KEY = os.getenv('TRENDSCANNER_ENTITY_EXTRACTION_KEY', '')


MODEL_DIR = f'deliverables/tic_checkpoints/{CHECKPOINT}/{SEED}/'
TOKENIZER_DIR = f'deliverables/tic_checkpoints/{CHECKPOINT}'


DOMAIN_COUNTRY_MAPPING = {
    "se": "Sweden",
    "uk": "United Kingdom",
}


@dataclass
class Document:
    text: str
    country: str = ''
    trend: str = ''
    entities: List = None


def load_model_ti_model() -> Tuple[DistilBertForSequenceClassification, AutoTokenizer, Any]:
    """
    Load the model, tokenizer, and label encoder for the TI model.

    This function loads a pre-trained DistilBert model for sequence classification, 
    a tokenizer, and a label encoder from specified directories. The model is set 
    to evaluation mode after loading.

    Returns:
        Tuple[DistilBertForSequenceClassification, AutoTokenizer, Any]: 
            A tuple containing the loaded model, tokenizer, and label encoder.
    """
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    label_encoder = joblib.load(f"{TOKENIZER_DIR}/label_encoder.pkl") 
    model.eval()
    return model, tokenizer, label_encoder
MODEL, TOKENIZER, LABEL_ENCODER = load_model_ti_model()


def load_documents(documents_dir: str = "deliverables/data") -> List[Document]:
    """
    Load and clean documents from a specified directory.

    This function reads all files from the given directory, extracts the top-level domain (TLD) from filenames
    that match a specific URL pattern, and processes the content of each file by stripping whitespace and 
    joining lines into a single string. It then creates a list of `Document` objects with the cleaned text and 
    associated metadata.

    Args:
        documents_dir (str): The directory containing the documents to be loaded. Defaults to "deliverables/data".

    Returns:
        List[Document]: A list of `Document` objects with cleaned text and metadata.
    """
    documents = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]
    documents = documents[:3]
    url_pattern = r"https~____(www\..+\.[a-z]{2,3})__.*"
    cleaned_documents = []
    for doc in documents:
        match = re.search(url_pattern, doc)
        if match:
            tld = match.group(1).split('.')[-1]
        else:
            tld = ''
        with open(os.path.join(documents_dir, doc), 'r') as file:
            lines = file.readlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            cleaned_text = ' '.join(cleaned_lines)
            cleaned_documents.append(Document(text=cleaned_text, entities=[], trend=None, country=DOMAIN_COUNTRY_MAPPING.get(tld, '')))
    return cleaned_documents


def call_wikifier(text:str, lang="en", threshold:float=0.8):
    """
    Annotate text using the Wikifier API.

    This function splits the input text into chunks if it exceeds 20,000 characters,
    sends each chunk to the Wikifier API for annotation, and merges annotations with
    the same title. The merged annotations are returned.

    Args:
        text (str): The text to be annotated.
        lang (str): The language of the text. Defaults to "en".
        threshold (float): The PageRank square threshold for filtering annotations. Defaults to 0.8.

    Returns:
        List[Dict]: A list of merged annotations.
    """
    def merge_annotations(annotations):
        merged_annotations = []
        seen_titles = set()
        for annotation in annotations:
            if annotation["title"] in seen_titles:
                continue
            same_title = [a for a in annotations if a["title"] == annotation["title"]]
            if len(same_title) > 1:
                merged_annotation = same_title[0]
                try:
                    for a in same_title[1:]:
                        merged_annotation["support"] += a["support"]
                        merged_annotation["supportLen"] += a["supportLen"]
                        merged_annotation["pageRank"] += a["pageRank"]
                        if "wikiDataClasses" in merged_annotation:
                            merged_annotation["wikiDataClasses"] += a["wikiDataClasses"]
                            merged_annotation["wikiDataClassIds"] += a["wikiDataClassIds"]
                        merged_annotation["dbPediaTypes"] += a["dbPediaTypes"]
                except Exception as e:
                    print("Error merging annotations:", e)
                merged_annotations.append(merged_annotation)
            else:
                merged_annotations.append(annotation)
            seen_titles.add(annotation["title"])
        return merged_annotations
    annotations = []
    chunks = [] 
    if len(text) > 20000:
        for i in range(0, len(text), 20000):
            chunks.append(text[i:i+20000])
    else:
        chunks.append(text)
    for chunk in chunks:
        data = urllib.parse.urlencode([
                ("text", chunk),
                ("lang", lang),
                ("userKey", TRENDSCANNER_ENTITY_EXTRACTION_KEY),
                ("pageRankSqThreshold", "%g" % threshold),
                ("applyPageRankSqThreshold", "true"),
                ("nTopDfValuesToIgnore", "200"),
                ("nWordsToIgnoreFromList", "200"),
                ("wikiDataClasses", "true"),
                ("wikiDataClassIds", "true"),
                ("support", "true"),
                ("ranges", "false"),
                ("minLinkFrequency", "2"),
                ("includeCosines", "false"),
                ("maxMentionEntropy", "3")
            ])
        url = "http://www.wikifier.org/annotate-article"
        req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout = 60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        annotations += response["annotations"]
    annotations = merge_annotations(annotations)
    return annotations


def predict_ti(text, min_confidence=0.3) -> Tuple[str, float]:
    """
    Predict the text classification with a minimum confidence threshold.

    Args:
        text (str): The input text to classify.
        min_confidence (float, optional): The minimum confidence threshold for a prediction to be considered valid. Defaults to 0.5.

    Returns:
        tuple or None: A tuple containing the predicted class name and its probability if the highest probability exceeds the minimum confidence threshold, otherwise None.
    """
    inputs = TOKENIZER(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = MODEL(**inputs)
    probabilities = softmax(outputs.logits, dim=1).squeeze(0)
    # Convert logits to class names with probabilities
    results = [(LABEL_ENCODER.inverse_transform([i])[0], prob.item()) for i, prob in enumerate(probabilities)]
    # Sort results by probability in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results[0] if sorted_results[0][1] > min_confidence else None


def get_sentence_chunks(text: str) -> List[str]:
    """
    Split the input text into chunks of sentences.
    
    Args:
        text (str): The input text to be split into sentence chunks.
    
    Returns:
        list: A list of text chunks, where each chunk contains up to three consecutive sentences from the input text.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = ["".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return chunks


def get_trend(doc_chunks:List[str]) -> str:
    """
    Analyze document chunks to determine the most frequent trend.
    
    Args:
        doc_chunks (List[str]): A list of text chunks to analyze for trends.
    
    Returns:
        str: The most frequent trend found in the document chunks. Returns an empty string if no trends are found.
    """
    trends = []
    for chunk in doc_chunks:
        ti = predict_ti(chunk)
        if ti:
            trends.append(ti)
    if len(trends) == 0:
        return ''
    else:
        return max(set(trends), key=trends.count)[0]


def run_entity_extraction_via_wikifier(doc:Document) -> Tuple[List[Dict], str]:
    """
    Run entity extraction on the given document using Wikifier and identify the most mentioned country.

    Args:
        doc (Document): The document object containing the text to be analyzed.

    Returns:
        tuple: A tuple containing:
            - entities (list): A list of entities extracted from the document.
            - country (str): The most mentioned country in the document. Returns an empty string if no countries are mentioned.
    """
    entities = call_wikifier(doc.text)
    mentioned_countries = []
    for entity in entities:
        if 'Country' in entity['dbPediaTypes']:
            mentioned_countries.append((entity['title'], entity['supportLen']))
    if len(mentioned_countries) == 1:
        country = mentioned_countries[0]
    elif len(mentioned_countries) > 1:
        country = max(mentioned_countries, key=lambda x: x[1])[0]
    else:
        country = ''
    return entities, country


def process_documents(documents:List[Document]) -> List[Document]:
    """
    Process a list of documents to extract trends and entities, and save the results to a JSON file.
    
    Args:
        documents (List[Document]): A list of Document objects to be processed.
    
    Returns:
        List[Document]: The list of processed Document objects with updated trends and entities.
    """
    for doc in tqdm(documents):
        doc.trend = get_trend(get_sentence_chunks(doc.text))
        doc.entities, doc.country = run_entity_extraction_via_wikifier(doc)
    with open('deliverables/parsed_documents.json', 'w') as f:
        json.dump([doc.__dict__ for doc in documents], f)
    return documents


def conduct_analysis_a(documents:List[Document]) -> pd.DataFrame:
    """
    Conduct analysis A to count trends per country and calculate their percentages.

    Args:
        documents (List[Document]): A list of Document objects to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results with columns ["country", "ti", "ti_count", 'ti_percentage'].
    """
    print("Conducting analysis A")
    countries = set([doc.country for doc in documents])
    ti_counts_per_country = {country: {} for country in countries}
    for doc in documents:
        ti_counts_per_country[doc.country][doc.trend] = ti_counts_per_country[doc.country].get(doc.trend, 0) + 1
    analysis_output = []
    for country, counts in ti_counts_per_country.items():
        country_total = sum(counts.values())
        for ti, count in counts.items():
            analysis_output.append([country, ti, count, count/country_total])
    data = pd.DataFrame(analysis_output, columns=["country", "ti", "ti_count", 'ti_percentage'])
    data.to_csv('deliverables/analysis_a.csv', index=False)
    print("Analysis A completed. Results saved to analysis_a.csv")
    return data


def conduct_analysis_b(documents:List[Document]) -> pd.DataFrame:
    """
    Conduct analysis B to extract entities and their types for each trend in each country.

    Args:
        documents (List[Document]): A list of Document objects to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results with columns ["country", "ti", "entity_type", "entity", 'support_len'].
    """
    print("Conducting analysis B")
    analysis_output = []
    for country in set([doc.country for doc in documents]):
        docs_country = [doc for doc in documents if doc.country == country]
        for ti in set([doc.trend for doc in docs_country]):
            docs_with_country_and_ti = [doc for doc in docs_country if doc.trend == ti]
            for entity in docs_with_country_and_ti[0].entities:
                if len(entity["dbPediaTypes"]) > 0:
                    entity_type = entity["dbPediaTypes"][-1]
                else:
                    entity_type = ''
                analysis_output.append([country, ti, entity_type, entity['title'], entity['supportLen']])
    data = pd.DataFrame(analysis_output, columns=["country", "ti", "entity_type", "entity", 'support_len'])
    data.to_csv('deliverables/analysis_b.csv', index=False)
    print("Analysis B completed. Results saved to analysis_b.csv")
    return data


def conduct_analysis_c(documents:List[Document]) -> pd.DataFrame:
    """
    Conduct analysis C to perform word frequency analysis for each trend.

    Args:
        documents (List[Document]): A list of Document objects to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results with columns ["ti", "word", "word_count", 'rank'].
    """
    print("Conducting analysis C")
    tis = set([doc.trend for doc in documents])
    analysis_output = []
    for ti in tis:
        docs_ti = [doc for doc in documents if doc.trend == ti]
        words = nltk.word_tokenize(" ".join([doc.text for doc in docs_ti]))
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_words = [word for word in filtered_words if word.lower() not in ['.', ',', '(', ')', '[', ']', '`', '\'', ';', '%', '{', '}', ':', '"', '!', '?', '-', '_', '/', '\\', '|', '@', '#', '$', '^', '&', '*', '+', '=', '<', '>', '~', '\'s']]
        filtered_words = [word for word in filtered_words if not word.replace(',', '').replace('.', '').isdigit()]
        filtered_words = [word for word in filtered_words if len(word) > 1]
        word_counts = {word: filtered_words.count(word) for word in set(filtered_words)}
        for word, count in word_counts.items():
            rank = sorted(word_counts.values(), reverse=True).index(count) + 1
            analysis_output.append([ti, word, count, rank])
    data = pd.DataFrame(analysis_output, columns=["ti", "word", "word_count", 'rank'])
    data.to_csv('deliverables/analysis_c.csv', index=False)
    print("Analysis C completed. Results saved to analysis_c.csv")
    return data


if __name__ == "__main__":
    raw_documents = load_documents('deliverables/data')
    if os.path.exists('deliverables/parsed_documents.json'):
        i = input('Documents seem to be already parsed. Do you want to reparse them? (y/n): ')
        if i.lower() == 'y':
            parsed_documents = process_documents(raw_documents)
        else:
            with open('deliverables/parsed_documents.json', 'r') as f:
                parsed_documents = [Document(**doc) for doc in json.load(f)]
    else:
        parsed_documents = process_documents(raw_documents)
    results_a = conduct_analysis_a(parsed_documents)
    results_b = conduct_analysis_b(parsed_documents)
    results_c = conduct_analysis_c(parsed_documents)
    