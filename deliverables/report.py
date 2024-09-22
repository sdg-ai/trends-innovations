import csv
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
from warnings import warn
from dotenv import load_dotenv
load_dotenv('deliverables/.env')
from typing import List, Tuple, Any, Dict
from nltk.corpus import stopwords
from dataclasses import dataclass, asdict
from torch.nn.functional import softmax
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from deliverables.entity_networks_master.app.entity_extraction import EntityExtractor
import concurrent.futures

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)


CHECKPOINT = os.getenv('CHECKPOINT', 'default_checkpoint_value')
CHECKPOINT = "checkpoint"
SEED = os.getenv('SEED', 'default_seed_value')
SEED = "seed_1"
TRENDSCANNER_ENTITY_EXTRACTION_KEY = os.getenv('TRENDSCANNER_ENTITY_EXTRACTION_KEY', '')


MODEL_DIR = f'deliverables/tic_checkpoints/{CHECKPOINT}/{SEED}/'
TOKENIZER_DIR = f'deliverables/tic_checkpoints/{CHECKPOINT}'


DOMAIN_COUNTRY_MAPPING = {
    "se": "Sweden",
    "uk": "United Kingdom",
    "org": "United Kingdom"
}

@dataclass
class Trend:
    title:str
    probability:float


@dataclass
class DocumentChunk:
    text: str
    trend: Trend
    entities: List = None


@dataclass
class Document:
    id: str = ''
    text: str = ''
    country: str = ''
    chunks: List[DocumentChunk] = None


def load_needs_mapping():
    needs_mapping = {}
    with open(get_data('Database_1.0_for_AI-Driven_Transformative.csv'), 'rt') as csv_f:
        csv_f = csv.reader(f)
        for row in f:
            innovation = row[4]
            if row[1]:
                level_1 = row[1]
            if row[2]:
                level_2 = row[2]
            if 'N/A' in row[3]:
                level_3 = None
            else:
                level_3 = row[3]
            needs_mapping[innovation] = {'level_1': level_1, 'level_2': level_2, 'level_3': level_3}

    return needs_mapping





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
    #documents = documents[:3]
    url_pattern = r"https~____(www\..+\.[a-z]{2,3})__.*"
    cleaned_documents = []
    for doc in documents:
        match = re.search(url_pattern, doc)
        if match:
            tld = match.group(1).split('.')[-1]
        else:
            warn(f"Could not extract TLD from filename: {doc}")
            tld = ''
        with open(os.path.join(documents_dir, doc), 'r') as file:
            lines = file.readlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            cleaned_text = ' '.join(cleaned_lines)
            chunks = get_sentence_chunks(cleaned_text)
            cleaned_documents.append(Document(
                id=doc,
                text=cleaned_text,
                chunks=chunks,
                country=DOMAIN_COUNTRY_MAPPING.get(tld, '')
            ))
    print(f"Loaded {len(cleaned_documents)} documents.")
    print(f"Total number of chunks: {sum([len(doc.chunks) for doc in cleaned_documents])}")
    return cleaned_documents

def get_sentence_chunks(text: str) -> List[str]:
    """
    Split the input text into chunks of sentences.
    
    Args:
        text (str): The input text to be split into sentence chunks.
    
    Returns:
        list: A list of text chunks, where each chunk contains up to three consecutive sentences from the input text.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = [DocumentChunk(text="".join(sentences[i:i+3]), trend=None) for i in range(0, len(sentences), 3)]
    return chunks


def call_wikifier_on_chunks(chunks:List[DocumentChunk], lang="en", threshold:float=0.8, **kwargs) -> List[DocumentChunk]:
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
                    # TODO: Look at how this is affected by spacy model change
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
    def fetch_annotations(chunk):
        # TODO: replace
        data = urllib.parse.urlencode([
            ("text", chunk),
            ("lang", lang),
            ("userKey", TRENDSCANNER_ENTITY_EXTRACTION_KEY),
            ("pageRankSqThreshold", "-1"),
            ("applyPageRankSqThreshold", "false"),
            ("nTopDfValuesToIgnore", "200"),
            ("nWordsToIgnoreFromList", "200"),
            ("wikiDataClasses", "true"),
            ("wikiDataClassIds", "false"),
            ("support", "true"),
            ("ranges", "false"),
            ("minLinkFrequency", "1"),
            ("includeCosines", "false"),
            ("maxMentionEntropy", "-1"),
            ("maxTargetsPerMention", "20")
        ])
        url = "http://www.wikifier.org/annotate-article"
        req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        return response["annotations"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_chunk = {executor.submit(fetch_annotations, chunk.text): chunk for chunk in chunks}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
            kwargs['pb'].set_postfix_str(f"Running entity extraction via Wikifier. Processing chunk {idx+1}/{len(chunks)}")
            try:
                chunks[idx].entities = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
    return chunks


# TODO: replace
def run_entity_extraction_via_wikifier(doc_chunks:List[DocumentChunk], **kwargs) -> Tuple[List[DocumentChunk], List[str]]:
    kwargs['pb'].set_postfix_str(f"Running entity extraction via Wikifier. Processing {len(doc_chunks)} chunks.")
    doc_chunks = call_wikifier_on_chunks(doc_chunks, pb=kwargs['pb'])
    kwargs['pb'].set_postfix_str(f"Extracting countries from entities.")
    countries = []
    for idx, doc_chunk in enumerate(doc_chunks):
        mentioned_countries = []
        for entity in doc_chunk.entities:
            if 'Country' in entity['dbPediaTypes']:
                mentioned_countries.append((entity['title'], entity['supportLen']))
        if len(mentioned_countries) == 1:
            country = mentioned_countries[0][0]
        elif len(mentioned_countries) > 1:
            country = max(mentioned_countries, key=lambda x: x[1])[0]
        else:
            country = ''
        doc_chunk.entities = doc_chunk.entities
        countries.append(country)
    return doc_chunks, countries


def predict_ti(text, min_confidence=0.5) -> Tuple[str, float]:
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
    return Trend(title=sorted_results[0][0], probability=sorted_results[0][1]) if sorted_results[0][1] > min_confidence else Trend(title='uncertain', probability=0.0)


def get_trends_in_document(doc_chunks:List[DocumentChunk], **kwargs) -> List[DocumentChunk]:
    """
    Analyze document chunks to determine the most frequent trend.
    
    Args:
        doc_chunks (List[DocumentChunk]): A list of DocumentChunk objects to be analyzed.
    
    Returns:
        List[DocumentChunk]: A list of DocumentChunk objects with updated trend information.
    """

    for idx, chunk in enumerate(doc_chunks):
        kwargs['pb'].set_postfix_str(f"Getting trends in document. Processing chunk {idx+1}/{len(doc_chunks)}")
        chunk.trend = predict_ti(chunk.text)
    else:
        return doc_chunks


def process_documents(documents:List[Document]) -> List[Document]:
    """
    Process a list of documents to extract trends and entities, and save the results to a JSON file.
    
    Args:
        documents (List[Document]): A list of Document objects to be processed.
    
    Returns:
        List[Document]: The list of processed Document objects with updated trends and entities.
    """
    pb = tqdm(total=len(documents), desc="Processing documents")
    for idx, doc in enumerate(documents):
        try:
            pb.set_description(f"Processing document {idx+1}/{len(documents)}")
            doc.chunks = get_trends_in_document(doc.chunks, pb=pb)
            doc.chunks, countries = run_entity_extraction_via_wikifier(doc.chunks, pb=pb)
            if doc.country == '':
                doc.country = max(set(countries), key=countries.count)
            pb.update(1)
            with open('deliverables/parsed_documents.json', 'w') as f:
                json.dump([asdict(doc) for doc in documents], f)
        except Exception as e:
            print(f"An error occurred, while processing document {idx+1} with name {doc.id}: {e}")
    return documents


def conduct_analysis_a(documents:List[Document]) -> pd.DataFrame:
    print("Conducting analysis A")
    countries = set([doc.country for doc in documents])
    ti_counts_per_country = {country: {} for country in countries}
    for doc in documents:
        for chunk in doc.chunks:
            if chunk.trend.title != 'uncertain' and chunk.trend.title != 'irrelevant':
                ti_counts_per_country[doc.country][chunk.trend.title] = ti_counts_per_country[doc.country].get(chunk.trend.title, 0) + 1
    analysis_output = []
    for country, counts in ti_counts_per_country.items():
        country_total = sum(counts.values())
        for ti, count in counts.items():
            if ti != 'uncertain' and ti != 'irrelevant':
                analysis_output.append([country, ti, count, count/country_total])
    data = pd.DataFrame(analysis_output, columns=["country", "ti", "ti_count", 'ti_percentage'])
    # compute accumulative sum of ti_percentage for each country
    data['ti_percentage_accumulative'] = data.groupby('country')['ti_percentage'].cumsum()
    data.to_csv('deliverables/analysis_a.csv', index=False)
    print("Analysis A completed. Results saved to analysis_a.csv")
    return data


def conduct_analysis_b(documents:List[Document]) -> pd.DataFrame:
    print("Conducting analysis B")
    analysis_output = []
    for country in set([doc.country for doc in documents]):
        docs_country = [doc for doc in documents if doc.country == country]
        for chunk in [chunk for doc in docs_country for chunk in doc.chunks]:
            if chunk.entities is not None:
                for entity in chunk.entities:
                    if len(entity["dbPediaTypes"]) > 0:
                        entity_type = entity["dbPediaTypes"][-1]
                    else:
                        entity_type = ''
                    analysis_output.append([country, chunk.trend.title, entity_type, entity['title'], entity['supportLen']])
   
    data = pd.DataFrame(analysis_output, columns=["country", "ti", "entity_type", "entity", 'support_len'])
    print("Length of data before grouping:", len(data))
    data = data.groupby(['country', 'ti', 'entity_type', 'entity']).agg({'support_len': 'sum'}).reset_index()
    print("Length of data after grouping:", len(data))
    # drop rows with ti = uncertain or ti = irrelevant
    data = data[data['ti'] != 'uncertain']
    data = data[data['ti'] != 'irrelevant']
    data.to_csv('deliverables/analysis_b.csv', index=False)
    print("Analysis B completed. Results saved to analysis_b.csv")
    return data


def conduct_analysis_c(documents:List[Document]) -> pd.DataFrame:
    print("Conducting analysis C")
    tis = set([chunk.trend.title for doc in documents for chunk in doc.chunks])
    analysis_output = []
    for ti in tis:
        chunks_ti = [chunk for doc in documents for chunk in doc.chunks if chunk.trend.title == ti]
        words = nltk.word_tokenize(" ".join([chunk.text for chunk in chunks_ti]))
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_words = [word for word in filtered_words if word.lower() not in ['.', ',', '(', ')', '[', ']', '`', '\'', ';', '%', '{', '}', ':', '"', '!', '?', '-', '_', '/', '\\', '|', '@', '#', '$', '^', '&', '*', '+', '=', '<', '>', '~', '\'s']]
        filtered_words = [word for word in filtered_words if not word.replace(',', '').replace('.', '').isdigit()]
        filtered_words = [word for word in filtered_words if len(word) > 1]
        # words that contain �
        filtered_words = [word for word in filtered_words if '�' not in word]
        filtered_words = [word for word in filtered_words if word != "''"]
        filtered_words = [word for word in filtered_words if '\\' not in word]
        word_counts = {word: filtered_words.count(word) for word in set(filtered_words)}
        for word, count in word_counts.items():
            rank = sorted(word_counts.values(), reverse=True).index(count) + 1
            if isinstance(ti, str) and "built-in" not in ti:
                analysis_output.append([ti, word, count, rank])
    data = pd.DataFrame(analysis_output, columns=["ti", "word", "word_count", 'rank'])
    data = data[data['ti'] != 'uncertain']
    data = data[data['ti'] != 'irrelevant']
    # drop rows where "built in" is in the wti column
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
                # load parsed documents from file to dataclass objects with nested dataclass objects
                parsed_documents = [Document(**doc) for doc in json.load(f)]
                for doc in parsed_documents:
                    doc.chunks = [DocumentChunk(**chunk) for chunk in doc.chunks]
                    for chunk in doc.chunks:
                        if chunk.entities is not None:
                            chunk.entities = [dict(entity) for entity in chunk.entities]
                        if chunk.trend is not None and chunk.trend != '':
                            chunk.trend = Trend(**chunk.trend)

    else:
        parsed_documents = process_documents(raw_documents)
    results_a = conduct_analysis_a(parsed_documents)
    results_b = conduct_analysis_b(parsed_documents)
    results_c = conduct_analysis_c(parsed_documents)
    