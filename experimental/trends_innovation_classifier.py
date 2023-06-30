# -*- coding: utf-8 -*-
import os
import pandas as pd
import spacy
from experimental.data_utils import load_params, doc_to_multisentence
from operator import itemgetter
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


# Load configuration parameters
params = load_params()


#############################################################################
#
# 	Trends and Innovations Classifier class.
#
#############################################################################

class TrendsInnovationClassifier:
    """
    The trends and innovations classifier for identifying "climate trend-topics" and "climate innovations".

    Attributes:
        debug (bool): Show debug messages (default = False).
        mode (str): Use of "single" or "multi" label models.
    """

    def __init__(self, debug=False, mode="multi"):
        """
        Constructor for the TrendsInnovationClassifier class.

        Parameters:
            debug (bool): Show debug messages (default = False).
            mode (str): Use of "single" or "multi" label models.
        """

        self.model_paths = None
        self.categories = None
        self.debug = debug
        self.mode = mode.lower()
        self.models = self.load_models(self.mode)

    def list_binary_models(self):
        """
        Lists the available single class models.

        Returns the categories for which there are available binary classification models. These are the identifiable classes.

        Returns:
            (list, list): List of available single class models, list of file paths of the available models.
        """

        categories = []
        filtered_filepaths = []
        filtered_categories = []
        inputfilepath = get_full_path(params['data']['path_to_model'])
        filepaths = [x[0] for x in os.walk(inputfilepath)]

        # Scan the contents of the model directory.
        for fn in filepaths:
            head, tail = os.path.split(fn)
            categories.append(tail)

        # Identify the model names.
        for filepath, category in zip(filepaths, categories):
            if '_model' in category:
                filtered_filepaths.append(filepath)
                filtered_categories.append(category)

        if self.debug:
            print(f"Available single class models: \n\tNo. of individual models:: {len(filtered_categories)} \t"
                  f"Model names: {filtered_categories}")

        return filtered_categories, filtered_filepaths

    def load_models(self, mode):
        """
        Loads the model(s).

        Parameters:
            mode (str): Load "single" or "multi" label model(s).

        Returns:
            list: List of loaded model(s).
        """

        classifiers = []

        # If 'single' mode, load single label classifiers.
        if mode == "single":
            self.categories, self.model_paths = self.list_binary_models()
            for model in self.model_paths:
                try:
                    model_best = os.path.join(model, 'model-best')
                    classifiers.append(spacy.load(model_best))
                except Exception as e:
                    print(e)

        # If 'multi' mode, load multi label classifier.
        elif mode == "multi":
            try:
                model_best = os.path.join(get_full_path(params['data']['path_to_model']),
                                          'balanced_multilabel_classifier', 'model-best')
                classifiers.append(spacy.load(model_best))
            except Exception as e:
                print(e)

        else:
            print("Invalid mode entered. Set mode to either 'single' or 'multi' label.")

        return classifiers

    def predict_block(self, text):
        """
        Make predictions on the given block of sentences.

        The decision threshold and the maximum label count per instance are taken from the configuration parameters.

        Parameters:
            text (str): Input text string.

        Returns:
            list: List of most likely trends and innovations, and their corresponding likelihood.
        """

        all_labels, all_probs, most_likely_trends = [], [], []
        threshold = params['predict']['threshold']
        count = params['predict']['count']

        if self.debug:
            print(f"\nPredicting sentence block: {text}")

        # Make prediction on sentence block using selected models.
        for nlp in self.models:
            doc = nlp(text)
            prediction = doc.cats
            for item in prediction:
                predict_label = item
                predict_prob = prediction[predict_label]
            if self.debug:
                if self.mode == "single":
                    print(f"\tmodel: {self.categories[self.models.index(nlp)]} \t predict_label: {predict_label} \t "
                          f"predict_prob: {predict_prob}")
                elif self.mode == "multi":
                    print(f"\tmodel: {self.models} \t predict_label: {predict_label} \t "
                          f"predict_prob: {predict_prob}")
            all_labels.append(predict_label)
            all_probs.append(predict_prob)

        # Keep the predictions that meet the decision threshold.
        most_likely_trends = [[all_labels[i], all_probs[i]] for i, v in enumerate(all_probs) if v > threshold]
        most_likely_trends = sorted(most_likely_trends, key=itemgetter(1), reverse=True)

        # Restrict the predictions to meet the label count per instance.
        try:
            most_likely_trends = [most_likely_trends[i] for i in range(count)]
        except:
            most_likely_trends = most_likely_trends

        if self.debug:
            print(f"Most likely trends/innovations: \n\tThreshold: {threshold} \tCount: {count} \t"
                  f"Matched: {len(most_likely_trends)} \tPredictions: {most_likely_trends}")

        return most_likely_trends

    def predict(self, text):
        """
        Function to scan over sentence blocks and make predictions.

        This function returns an array of the form: array([{"text": <text-snippet>, "indices": [(
        <start>, <end>)], "prediction": <tag str>}]

        Parameters:
            text (str): Input text string.

        Returns:
            list: List of dictionaries containing identified trends/innovations in text snippets and their string indices.
        """

        enriched_sentence_objects = []
        tags = set()
        sentence_objects = doc_to_multisentence(text, 3)

        # Predict over sentence blocks
        for sentence_block in sentence_objects:
            text_block = ' '.join(sentence_block['text'])
            predictions = self.predict_block(text_block)

            sentence_block['predictions'] = {}
            for item in predictions:
                sentence_block['predictions'][item[0]] = item[1]

            for tag, confidence in predictions:
                tags.add(tag)
            if predictions:
                enriched_sentence_objects.append(sentence_block)

        # Condense predictions
        predictions = []
        block_count = len(enriched_sentence_objects)
        for tag in tags:
            current_index_set = []
            current_sentence_set = []
            current_end_offset = None
            i = 0
            while i < block_count:
                if tag in enriched_sentence_objects[i]['predictions']:
                    if not current_end_offset:
                        current_end_offset = i
                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)

                    elif i - current_end_offset < 3:
                        current_end_offset = i
                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)

                    else:
                        # Format prediction object
                        prediction_obj = {'string_indices': [list(current_index_set)[0][0],
                                                             list(current_index_set)[-1][1]],
                                          'text': ' '.join(current_sentence_set),
                                          'prediction': tag}
                        if self.debug:
                            prediction_obj['text'] = ' '.join(current_sentence_set)
                        predictions.append(prediction_obj)

                        # Reset variables
                        current_end_offset = i
                        current_index_set = []
                        current_sentence_set = []

                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)
                i += 1
            if current_index_set:
                # Format prediction object
                prediction_obj = {'string_indices': [list(current_index_set)[0][0],
                                                     list(current_index_set)[-1][1]],
                                  'text': ' '.join(current_sentence_set),
                                  'prediction': tag}
                if self.debug:
                    prediction_obj['text'] = ' '.join(current_sentence_set)
                predictions.append(prediction_obj)

        return predictions


def run_example():
    """
    Sample code for predicting trends/innovations
    """

    # Create classifier object
    classifier = TrendsInnovationClassifier(mode='multi', debug=False)

    # Fetch the input text
    with open(get_full_path('example_text.txt'), 'rt', encoding='utf-8', errors='ignore') as tf:
        example_text = tf.read()

    # Make predictions
    output_data = classifier.predict(example_text)

    # Print the result
    df = pd.DataFrame(output_data, columns=['string_indices', 'prediction', 'text'])
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    run_example()
