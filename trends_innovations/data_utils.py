# Import libraries
import os
import yaml
import nltk
from pathlib import Path

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_full_path(path):
    path = Path(path)
    return os.path.join(_ROOT, path)


#############################################################################
#
# 	Utility functions for the trends and innovations classifier.
#
#############################################################################

def load_params():
    """
    Loads configuration parameters.

    Returns:
        dict: Dictionary containing the configuration parameters and their values.
    """

    with open(get_full_path("data/config.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return params


def doc_to_multisentence(text: str, num_sentences: int):
    """
    Splits text into blocks/groups of sentences.

    The function returns the string indices and corresponding sentences for each of the generated blocks.

    Parameters:
        text (str): Input text.
        num_sentences (int): Number of sentences in a block.

    Returns:
        list: List of dictionaries containing arrays of the string indices and corresponding sentences for each block.
    """

    sentences = nltk.sent_tokenize(text)
    offset = 0
    sent_dict = {"string_indices": (0, 0), "text": ""}
    split_sentences = []
    for j in range(len(sentences) - num_sentences + 1):
        line_indices = []
        line_text = []
        for line in sentences[j:j + num_sentences]:
            offset = text.find(line, offset)
            line_indices.append(tuple([offset, offset + len(line)]))
            line_text.append(str(line))
        sent_dict["string_indices"] = [line_indices[i] for i in range(len(line_indices))]
        sent_dict["text"] = [line_text[i] for i in range(len(line_text))]
        split_sentences.append(sent_dict.copy())
        offset = 0

    return split_sentences
