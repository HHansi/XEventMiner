# Created by Hansi at 5/9/2023
import argparse
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split

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
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args, use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)
explainer = shap.Explainer(model)
shap_values = explainer(train[:1])
print(shap_values)
