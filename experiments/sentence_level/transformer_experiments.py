import argparse

import pandas as pd
import numpy as np
import sklearn
import torch

from sklearn.model_selection import train_test_split

from experiments.sentence_level.transformer_config import transformer_args, SEED
from experiments.sentence_level.evaluation import macro_f1, weighted_f1, print_evaluation
from text_classification.text_classification_model import TextClassificationModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--language', required=False, help='language', default="en")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)

arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)
language = arguments.language

# en_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/en-train.json', lines=True)
# es_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/es-train.json',
#                                  lines=True)
# pr_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/pr-train.json',
#                                  lines=True)

en_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/en-train.json',
                                    lines=True)

en_sentence_data = en_sentence_data.rename(columns={'sentence': 'text', 'label': 'labels'})
en_sentence_data = en_sentence_data[['text', 'labels']]
en_train, en_test_df = train_test_split(en_sentence_data, test_size=0.1, random_state=777)
en_test_sentences = en_test_df['text'].tolist()

es_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/es-train.json',
                                    lines=True)
es_sentence_data = es_sentence_data.rename(columns={'sentence': 'text', 'label': 'labels'})
es_sentence_data = es_sentence_data[['text', 'labels']]
es_train, es_test_df = train_test_split(es_sentence_data, test_size=0.1, random_state=777)
es_test_sentences = es_test_df['text'].tolist()


pr_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/pr-train.json',
                                 lines=True)
pr_sentence_data = pr_sentence_data.rename(columns={'sentence': 'text', 'label': 'labels'})
pr_sentence_data = pr_sentence_data[['text', 'labels']]
pr_train, pr_test_df = train_test_split(pr_sentence_data, test_size=0.1, random_state=777)
pr_test_sentences = pr_test_df['text'].tolist()

# sentence_data = sentence_data.rename(columns={'sentence': 'text', 'label': 'labels'})
# sentence_data = sentence_data[['text', 'labels']]

# train, test_df = train_test_split(sentence_data, test_size=0.1, random_state=777)

# test_sentences = test_df['text'].tolist()

model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args,
                                use_cuda=torch.cuda.is_available())

frames = [en_train, es_train]
train = pd.concat(frames)


train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

print("English")
predictions, raw_outputs = model.predict(en_test_sentences)
en_test_df["predictions"] = predictions
print_evaluation(en_test_df, "predictions", "labels")
en_test_df.to_csv("en_results_mBERT-en-es.tsv", sep='\t', encoding='utf-8', index=False)

print("Spanish")
predictions, raw_outputs = model.predict(es_test_sentences)
es_test_df["predictions"] = predictions
print_evaluation(es_test_df, "predictions", "labels")
es_test_df.to_csv("es_results_mBERT-en-es.tsv", sep='\t', encoding='utf-8', index=False)

print("Portuguese")
predictions, raw_outputs = model.predict(pr_test_sentences)
pr_test_df["predictions"] = predictions
print_evaluation(pr_test_df, "predictions", "labels")
pr_test_df.to_csv("pr_results_mBERT-en-es.tsv", sep='\t', encoding='utf-8', index=False)







