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

en_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/en-train.json', lines=True)
es_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/es-train.json',
                                 lines=True)
pr_sentence_data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/pr-train.json',
                                 lines=True)

if language == "en":
    sentence_data = en_sentence_data
elif language == "es":
    sentence_data = es_sentence_data
elif language == "pr":
    sentence_data = pr_sentence_data

sentence_data = sentence_data.rename(columns={'sentence': 'text', 'label': 'labels'})
sentence_data = sentence_data[['text', 'labels']]

train, test_df = train_test_split(sentence_data, test_size=0.1, random_state=777)

test_sentences = test_df['text'].tolist()

model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args,
                                use_cuda=torch.cuda.is_available())

train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)
predictions, raw_outputs = model.predict(test_sentences)
test_df["predictions"] = predictions

print_evaluation(test_df, "predictions", "labels")
test_df.to_csv("results_BERT.tsv", sep='\t', encoding='utf-8', index=False)







