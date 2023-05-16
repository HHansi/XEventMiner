import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import shap
import transformers
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.token_level.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="sinhala-nlp/xlm-t-sold-si")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--language', required=False, help='language of data', default="en")
parser.add_argument('--output_folder', required=False, help='output folder', default="outputs")
arguments = parser.parse_args()

MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
language = arguments.language
output_folder = os.path.join(arguments.output_folder, 'shap', MODEL_NAME.split('/')[1], language)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

RANDOM_STATE = 777

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

    # load the model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).cuda()

    # build a pipeline object to do predictions
    pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=cuda_device,
                                 return_all_scores=True)
    pmodel = shap.models.TransformersPipeline(pred, rescale_to_logits=True)

    masker = shap.maskers.Text(tokenizer=r"\s+")
    explainer = shap.Explainer(pmodel, masker=masker)

    train_sentence_id = 0
    train_token_df = []
    train_explanations_df = []
    print(f'processing train set..')
    for index, row in train.iterrows():
        shap_values = explainer([row["text"]])
        data_tokens = shap_values.data[0].tolist()
        explanation_values = shap_values[:, :, 1].values[0].tolist()
        # print(f'data length: {len(data_tokens)}, value length: {len(explanation_values)}')

        # get (token, value) pairs in value descending order
        sorted_explanations = sorted(tuple(zip(shap_values.data[0], shap_values[:, :, 1].values[0])),
                                     key=lambda x: x[1], reverse=True)
        train_explanations_df.append([train_sentence_id, row["text"], sorted_explanations])

        tokens = ast.literal_eval(row["tokens"])
        labels = json.loads(row["rationales"])

        if [x.strip() for x in data_tokens] != tokens:
            print(f'index{index}: found mismatch in tokens.')
            continue

        for i in range(0, len(tokens)):
            processed_row = [train_sentence_id, tokens[i], labels[i], explanation_values[i]]
            train_token_df.append(processed_row)
        train_sentence_id = train_sentence_id + 1

    train_data = pd.DataFrame(train_token_df, columns=["sentence_id", "words", "labels", "explanations"])
    train_explanations = pd.DataFrame(train_explanations_df, columns=["sentence_id", "sentence", "explanation"])
    train_explanations.to_csv(os.path.join(output_folder, 'train_explanation.csv'), index=False, encoding='utf-8')

    test_sentence_id = 0
    test_token_df = []
    test_explanations_df = []
    print(f'processing test set..')
    for index, row in test.iterrows():
        shap_values = explainer([row["text"]])
        data_tokens = shap_values.data[0].tolist()
        explanation_values = shap_values[:, :, 1].values[0].tolist()
        # print(f'data length: {len(data_tokens)}, value length: {len(explanation_values)}')

        # get (token, value) pairs in value descending order
        sorted_explanations = sorted(tuple(zip(shap_values.data[0], shap_values[:, :, 1].values[0])),
                                     key=lambda x: x[1], reverse=True)
        test_explanations_df.append([test_sentence_id, row["text"], sorted_explanations])

        tokens = ast.literal_eval(row["tokens"])
        labels = json.loads(row["rationales"])

        if [x.strip() for x in data_tokens] != tokens:
            print(f'index{index}: found mismatch in tokens.')
            continue

        for i in range(0, len(tokens)):
            processed_row = [test_sentence_id, tokens[i], labels[i], explanation_values[i]]
            test_token_df.append(processed_row)
        test_sentence_id = test_sentence_id + 1

    test_data = pd.DataFrame(test_token_df, columns=["sentence_id", "words", "labels", "explanations"])
    test_explanations = pd.DataFrame(test_explanations_df, columns=["sentence_id", "sentence", "explanation"])
    test_explanations.to_csv(os.path.join(output_folder, 'test_explanation.csv'), index=False, encoding='utf-8')

    X = np.array(train_data['explanations'].tolist()).reshape(-1, 1)
    Y = np.array(train_data['labels'].tolist())

    print(f'training SGDClassifier..')
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE))
    clf.fit(X, Y)
    predictions = clf.predict(np.array(test_data['explanations'].tolist()).reshape(-1, 1))

    test_data["predictions"] = predictions
    print_information(test_data, "predictions", "labels")

    train_data.to_csv(os.path.join(output_folder, 'train.csv'), index=False, encoding='utf-8')
    test_data.to_csv(os.path.join(output_folder, 'test.csv'), index=False, encoding='utf-8')
