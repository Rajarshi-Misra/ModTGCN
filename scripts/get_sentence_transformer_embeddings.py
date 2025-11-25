"""
We need to specify the model correctly in this file for generating embeddings around line 64
"""
import re, os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='name of dataset')
args = parser.parse_args()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def encode_labels(y_train, y_val, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)
    return y_train_enc, y_val_enc, y_test_enc
    
def load(dataset: str):
    df_train = pd.read_csv('../inputs/' + dataset + '/train.csv')
    df_val = pd.read_csv('../inputs/' + dataset + '/val.csv')
    df_test = pd.read_csv('../inputs/' + dataset + '/test.csv')

    df_train['text'] = df_train['text'].transform(lambda x: clean_str(x))
    df_val['text'] = df_val['text'].transform(lambda x: clean_str(x))
    df_test['text'] = df_test['text'].transform(lambda x: clean_str(x))
    train_size = df_train.shape[0]
    val_size = df_val.shape[0]
    test_size = df_test.shape[0]

    Y_train, Y_val, Y_test = encode_labels(df_train['label'].values, df_val['label'].values, df_test['label'].values)

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    print("data loaded")
    return df_train, df_val, df_test, Y_train, Y_val, Y_test

df_train, df_val, df_test, Y_train, Y_val, Y_test = load(args.dataset)

sbert_model = SentenceTransformer(f'../models/sbert/{args.dataset}/checkpoint')##NOTE: This changes the model used
train_embeddings = sbert_model.encode(df_train['text'].values, convert_to_tensor=True)
val_embeddings = sbert_model.encode(df_val['text'].values, convert_to_tensor=True)
test_embeddings = sbert_model.encode(df_test['text'].values, convert_to_tensor=True)

os.makedirs(f'../embeddings/sbert_fine_tuned/{args.dataset}', exist_ok=True) ##NOTE: This also needs to be changed
path = f'../embeddings/sbert_fine_tuned/{args.dataset}' ##NOTE: This also needs to be changed
np.save(f'{path}/train_embeddings.npy', train_embeddings.cpu().numpy())
np.save(f'{path}/val_embeddings.npy', val_embeddings.cpu().numpy())
np.save(f'{path}/test_embeddings.npy', test_embeddings.cpu().numpy())
np.save(f'{path}/Y_train.npy', np.array(Y_train, dtype = np.int64))
np.save(f'{path}/Y_val.npy', np.array(Y_val, dtype = np.int64))
np.save(f'{path}/Y_test.npy', np.array(Y_test, dtype = np.int64))
print(f"Embeddings saved for dataset {args.dataset}")

