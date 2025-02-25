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

    with open(get_full_path("../data/config.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return params


def doc_to_multisentence(text_obj:dict, num_sentences:int):
    """
    Splits text into blocks/groups of sentences.

    The function returns the string indices and corresponding sentences for each of the generated blocks.

    Parameters:
        text_obj (dict): Input text object.
        num_sentences (int): Number of sentences in a block.

    Returns:
        list: List of dictionaries containing arrays of the string indices and corresponding sentences for each block.
    """
    
    multi_sentence_obj = []
    n = len(text_obj)
    k = num_sentences
    
    if n < k:
        multi_sentence_obj = text_obj
    else:
        for j in range(0,n-k+1):
            group_indices = [(text_obj[ind]['string_indices']) for ind in range(j,j+k)]
            group_text = [(text_obj[ind]['text']) for ind in range(j, j + k)]
            group_obj = {'string_indices':group_indices, 'text':group_text}
            multi_sentence_obj.append(group_obj)
    
    return multi_sentence_obj
