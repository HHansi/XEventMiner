import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert")
parser.add_argument('--model_type', required=False, help='model type', default="bert-large-cased")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)

arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

data = pd.read_json('experiments/data/CASE2021/subtask2-sentence/without_duplicates/en-train.json', lines=True)

data = data.rename(columns={'sentence': 'text', 'label': 'labels'})
data = data[['text', 'labels']]

# train_df, test_df = train_test_split(data, test_size=0.1, random_state=SEED * i)