import argparse
import ast
import json

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
arguments = parser.parse_args()

MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

RANDOM_STATE = 777

df = pd.read_csv('experiments/data/CASE2021/subtask4-token/without_duplicates/en-train.csv', encoding='utf-8')
train, test = train_test_split(df, test_size=0.8, random_state=RANDOM_STATE)
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

# load the model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).cuda()

# build a pipeline object to do predictions
pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=cuda_device,
                             return_all_scores=True)

masker = shap.maskers.Text(tokenizer=r"\s+")
explainer = shap.Explainer(pred, masker=masker)
shap_values = explainer(train["text"][:2])
print(shap_values)

train_sentence_id = 0
train_token_df = []
print(f'processing train set..')
for index, row in train.iterrows():
    shap_values = explainer(row["text"])
    data_tokens = shap_values.data[0]
    explanation_values = shap_values[:, :, 1].values[0]

    # print (token, value) pairs in value descending oder
    print(sorted(tuple(zip(shap_values.data[0], shap_values[:, :, 1].values[0])), key=lambda x: x[1], reverse=True))

    tokens = ast.literal_eval(row["tokens"])
    labels = json.loads(row["rationales"])

    if data_tokens != tokens:
        print(f'index{index}: found mismatch in tokens.')
        continue

    for i in range(0, len(tokens)):
        processed_row = [train_sentence_id, tokens[i], labels[i], explanation_values[i]]
    train_sentence_id = train_sentence_id + 1

train_data = pd.DataFrame(train_token_df, columns=["sentence_id", "words", "labels", "explanations"])

test_sentence_id = 0
test_token_df = []
print(f'processing test set..')
for index, row in test.iterrows():
    shap_values = explainer(row["text"])
    data_tokens = shap_values.data[0]
    explanation_values = shap_values[:, :, 1].values[0]

    # print (token, value) pairs in value descending oder
    print(sorted(tuple(zip(shap_values.data[0], shap_values[:, :, 1].values[0])), key=lambda x: x[1], reverse=True))

    tokens = ast.literal_eval(row["tokens"])
    labels = json.loads(row["rationales"])

    if data_tokens != tokens:
        print(f'index{index}: found mismatch in tokens.')
        continue

    for i in range(0, len(tokens)):
        processed_row = [test_sentence_id, tokens[i], labels[i], explanation_values[i]]
    test_sentence_id = test_sentence_id + 1

test_data = pd.DataFrame(test_token_df, columns=["sentence_id", "words", "labels", "explanations"])

X = np.array(train_data['explanations'].tolist()).reshape(-1, 1)
Y = np.array(train_data['labels'].tolist())

print(f'training SGDClassifier..')
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE))
clf.fit(X, Y)
predictions = clf.predict(np.array(test_data['explanations'].tolist()).reshape(-1, 1))

test_data["predictions"] = predictions
print_information(test_data, "labels", "predictions")
