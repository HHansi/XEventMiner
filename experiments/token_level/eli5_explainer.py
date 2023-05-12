import argparse
import os

import numpy as np
import pandas as pd
import torch
from eli5.lime import TextExplainer
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from experiments.sentence_level.transformer_config import transformer_args
from experiments.token_level.print_stat import print_information
from text_classification.text_classification_model import TextClassificationModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="sinhala-nlp/xlm-t-sold-si")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--language', required=False, help='language of data', default="en")
parser.add_argument('--output_folder', required=False, help='output folder', default="outputs")
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
language = arguments.language
output_folder = os.path.join(arguments.output_folder, 'eli5', MODEL_NAME.split('/')[1], language)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

RANDOM_STATE = 777


def _tokenizer(text):
    return text.split()


def _predict_probabilities(test_sentences):
    predictions, raw_outputs = model.predict(test_sentences)
    probabilities = softmax(raw_outputs, axis=1)
    return probabilities


df = None
if language == "en":
    df = pd.read_csv('experiments/data/CASE2021/subtask4-token/without_duplicates/en-train.csv', encoding='utf-8')
elif language == "es":
    df = pd.read_csv('experiments/data/CASE2021/subtask4-token/without_duplicates/es-train.csv', encoding='utf-8')
elif language == "pr":
    df = pd.read_csv('experiments/data/CASE2021/subtask4-token/without_duplicates/pr-train.csv', encoding='utf-8')
else:
    print('No valid language is given.')

if df is not None:
    train, test = train_test_split(df, test_size=0.8, random_state=RANDOM_STATE)
    print(f'train shape: {train.shape}')
    print(f'test shape: {test.shape}')

    model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args, use_cuda=torch.cuda.is_available(),
                                    cuda_device=cuda_device)
    # explainer = LimeTextExplainer(split_expression=_tokenizer, class_names=[0, 1])1
    explainer = TextExplainer(random_state=RANDOM_STATE, token_pattern=r"\S+")

    train_sentence_id = 0
    train_token_df = []
    train_explanations_df = []
    print(f'processing train set..')
    for index, row in train.iterrows():
        print(row["text"])
        explainer.fit(row["text"], _predict_probabilities)
        explainer.show_prediction()

        break

