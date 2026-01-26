import torch
import re
import pandas as pd
from math import log
import scipy.sparse as sp

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

def count_freq(df: pd.DataFrame, ):
    word_freq = {}
    for text in df['text']:
        words = text.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


def remove_less_freq_words(words, word_freq: dict, stop_words: set, dataset):
    words = words.split()
    doc_words = []
    for word in words:
        if dataset in ("MR","mr"):
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)
    return ' '.join(doc_words).strip()


def get_word_doc_list(df_main: pd.DataFrame, word_doc_list):
    for i in range(len(df_main)):
        doc_words = df_main.iloc[i]['text']
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

def find_word_pair_count(windows, word_id_map):
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    return word_pair_count


def get_pmi(windows, word_pair_count, word_window_freq, vocab, train_size, val_size):
    row = []
    col = []
    weight = []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + val_size + i)
        col.append(train_size + val_size + j)
        weight.append(pmi)
    return row, col, weight

def add_tfid(row, col, weight, df_main, word_id_map, train_size, val_size, test_size, vocab_size, word_doc_freq, vocab):
    doc_word_freq = {}
    for doc_id in range(len(df_main)):
        doc_words = df_main.iloc[doc_id]["text"]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(df_main)):
        doc_words = df_main.iloc[i]["text"]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size + val_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + val_size + j)
            idf = log(1.0 * len(df_main) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    node_size = train_size + val_size + vocab_size + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    #adj = torch.sparse_coo_tensor([row, col], weight, size=(node_size, node_size)) # this does not work, as strided operations are not supported
    return adj

def normalize_adj_pw(adj, colsum, rowsum, cpw, rpw, EPS=1e-9):
    col_pw = torch.pow(colsum + EPS, cpw).flatten()
    row_pw = torch.pow(rowsum + EPS, rpw).flatten()

    col_pw = torch.sparse.spdiags(col_pw.cpu(), torch.tensor([0]), (len(col_pw), len(col_pw))).to(adj.device)
    row_pw = torch.sparse.spdiags(row_pw.cpu(), torch.tensor([0]), (len(row_pw), len(row_pw))).to(adj.device)

    ret = col_pw.mm(adj).mm(row_pw)
    return ret

def gaussian_dist(emb, sigma):
    sq_dists = torch.cdist(emb, emb, p=2) ** 2
    weight = torch.exp(-sq_dists / (2 * sigma ** 2))
    weight = weight - torch.diag(torch.diag(weight))
    return weight