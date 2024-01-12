import numpy as np
import os
import pandas as pd
import json
import ast

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_json_data(data_dir: str) -> pd.DataFrame:
    """
    Load all jsonl files in the data directory and returns a dataframe
    :param data_dir: the path to the directory containing the json files (one per clas)
    :return: the data as a pd.Dataframe
    """
    data_dir = os.path.join(data_dir, "annotated_data")
    rows = []
    for f in [f for f in os.listdir(data_dir) if f.endswith('jsonl')]:
        with open(f'{data_dir}/{f}', 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            label = result["label"]
            text = result["text"]
            new_spans = []
            if result["spans"] and len(result["spans"]) != 0:
                try:
                    new_spans = [s["text"] for s in result["spans"]]
                except:
                    x = ast.literal_eval(result["spans"])
                    new_spans = [s["text"] for s in x]
            new_row = {
                "text": text,
                "label": label,
                "spans": new_spans,
                "article_id": result["meta"]["doc_id"],
                "sentence_id": result["meta"]["sent_id"],
                "answer": result["answer"],
                "priority": result["priority"],
                "score": result["score"],
                "title": result["meta"]["title"],
            }
            rows.append(new_row)
        print(f"Loaded: {f}")
    df = pd.DataFrame(rows)
    # only keep rows with unique article_id and sentence_id
    df.drop_duplicates(subset=["article_id", "sentence_id"], inplace=True)
    return df