import argparse
import ast
import json

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.sentence_level.transformer_config import transformer_args
from experiments.token_level.print_stat import print_information
from text_classification.text_classification_model import TextClassificationModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="sinhala-nlp/xlm-t-sold-si")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

RANDOM_STATE = 777

df = pd.read_csv('../data/CASE2021/subtask4-token/without_duplicates/en-train.csv', encoding='utf-8')
train, test = train_test_split(df, test_size=0.8, random_state=RANDOM_STATE)


def _tokenizer(text):
    return text.split()


def _predict_probabilities(test_sentences):
    predictions, raw_outputs = model.predict(test_sentences)
    probabilities = softmax(raw_outputs, axis=1)
    return probabilities


model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args, use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)
explainer = LimeTextExplainer(split_expression=_tokenizer, class_names=[0, 1])

train_sentence_id = 0
train_token_df = []
for index, row in train.iterrows():
    exp = explainer.explain_instance(row["tokens"], _predict_probabilities, num_features=200)
    explanations = exp.as_list()
    tokens = ast.literal_eval(row["tokens"])
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [train_sentence_id, token, 0, explanation[1]]
                    train_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [train_sentence_id, token, label, explanation[1]]
                    train_token_df.append(processed_row)
    train_sentence_id = train_sentence_id + 1

train_data = pd.DataFrame(
    train_token_df, columns=["sentence_id", "words", "labels", "explanations"])

test_sentence_id = 0
test_token_df = []
for index, row in test.iterrows():
    exp = explainer.explain_instance(row["tokens"], _predict_probabilities, num_features=200)
    explanations = exp.as_list()
    tokens = ast.literal_eval(row["tokens"])
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [test_sentence_id, token, 0, explanation[1]]
                    test_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [test_sentence_id, token, label, explanation[1]]
                    test_token_df.append(processed_row)
    test_sentence_id = test_sentence_id + 1

test_data = pd.DataFrame(test_token_df, columns=["sentence_id", "words", "labels", "explanations"])

X = np.array(train_data['explanations'].tolist()).reshape(-1, 1)
Y = np.array(train_data['labels'].tolist())

clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE))
clf.fit(X, Y)
predictions = clf.predict(np.array(test_data['explanations'].tolist()).reshape(-1, 1))

test_data["predictions"] = predictions
print_information(test_data, "labels", "predictions")

