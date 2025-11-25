from nltk.corpus import stopwords
import nltk
from scipy.sparse import vstack
import scipy.io
import numpy as np
import pandas as pd
from utils import *
from build_adj import *

#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))

def load(dataset: str):
    # data_dir = os.path.dirname(os.path.abspath(__file__))
    # print(data_dir)
    df_train = pd.read_csv('./inputs/' + dataset + '/train.csv')
    df_val = pd.read_csv('./inputs/' + dataset + '/val.csv')
    df_test = pd.read_csv('./inputs/' + dataset + '/test.csv')

    df_train['text'] = df_train['text'].transform(lambda x: clean_str(x))
    df_val['text'] = df_val['text'].transform(lambda x: clean_str(x))
    df_test['text'] = df_test['text'].transform(lambda x: clean_str(x))
    df_main = pd.concat([df_train, df_val, df_test])
    one_hot = pd.get_dummies(df_main.label).to_numpy()
    train_size = df_train.shape[0]
    val_size = df_val.shape[0]
    test_size = df_test.shape[0]

    Y_train = one_hot[0:train_size, :]
    Y_val = one_hot[train_size:train_size + val_size, :]
    Y_test = one_hot[train_size + val_size:, :]
    return df_train, df_val, df_test, Y_train, Y_val, Y_test, train_size, val_size, test_size