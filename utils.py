import re
from collections import Counter
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from math import log
import os
from sentence_transformers import SentenceTransformer

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
    word_counter = Counter()
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

def mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def gaussian_dist(emb, sigma):
    sq_dists = torch.cdist(emb, emb, p=2) ** 2
    weight = torch.exp(-sq_dists / (2 * sigma ** 2))
    weight = weight - torch.diag(torch.diag(weight))
    return weight

class AdjacencyMatrixCreatorForModularity:
    """
    A class for creating adjacency matrices using different methods.
    """
    def __init__(self, method="from_symmetric"):
        """
        Initializes the AdjacencyMatrixCreator with the specified method.
        """
        self.method = method

    def create(self, args=None, doc_word_matrix=None, word_word_matrix=None, df_train = None, df_test = None, df_val = None):
        """
        Creates the adjacency matrix using the specified method.
        """
        if self.method == "from_symmetric":
            return self._from_symmetric(doc_word_matrix, df_train, df_val, df_test, args)
        elif self.method == "from_doc_word_symmetric":
            return self._from_doc_word_symmetric(doc_word_matrix, word_word_matrix)
        elif self.method == "from_sbert_embeddings":
            return self._from_SBERT_embeddings(args, df_train, df_val, df_test)
        elif self.method == "from_sbert_embeddings_gaussian":
            return self._from_SBERT_embeddings_gaussian(args, df_train, df_val, df_test)
        elif self.method == "from_gpt_embeddings_cosine":
                return self._from_GPT_embeddings(args, df_train, df_val, df_test)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _from_symmetric(self, doc_word_matrix, df_train, df_val, df_test, args):
        """
        Create doc-doc adjacency matrix by multiplying normalized doc-word with its transpose.
        """
        # Compute degree matrices
        D_left = torch.sum(doc_word_matrix, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(doc_word_matrix, dim=0, keepdim=True).to_dense()
        D_lpw = -0.5
        D_rpw = -0.5
        # Normalize doc-word matrix
        normalized_mat = normalize_adj_pw(doc_word_matrix, D_left, D_right, D_lpw, D_rpw).to_dense()
        total_size = normalized_mat.shape[0]
        # Cosine-like similarity: doc-doc adjacency
        sim_matrix = torch.matmul(normalized_mat, normalized_mat.T)
        if args.modify_graph:
            adj_matrix = torch.zeros((total_size, total_size), dtype=torch.float32, device=sim_matrix.device)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            # Build adjacency with top-k
            for i in range(total_size):
                # torch.topk instead of np.argsort
                _, top_k_indices = torch.topk(sim_matrix[i], args.k)

                for j in top_k_indices.tolist():
                    if i == j:
                        continue
                    sim = sim_matrix[i, j]

                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease

                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim

            return adj_matrix

        return sim_matrix


    def _from_doc_word_symmetric(self, doc_word_matrix, word_word_matrix):
        D_left = torch.sum(doc_word_matrix, dim=1, keepdim=True).to_dense()
        D_right = torch.sum(doc_word_matrix, dim=0, keepdim=True).to_dense() + torch.sum(word_word_matrix, dim=0, keepdim=True).to_dense()
        D_lpw = -0.5
        D_rpw = -0.5
        normalized_mat = normalize_adj_pw(doc_word_matrix, D_left, D_right, D_lpw, D_rpw).to_dense()
        return torch.matmul(normalized_mat, normalized_mat.T)

    def _from_SBERT_embeddings(self, args, df_train, df_val, df_test):
        documents = np.concatenate([df_train['text'].values, df_val['text'].values, df_test['text'].values])
        sbert_model = SentenceTransformer(args.llm)
        embeddings = sbert_model.encode(documents, convert_to_tensor=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # [N, D]
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        if args.modify_graph:
            total_size = sim_matrix.size(0)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.float32)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            _, top_k_indices = torch.topk(sim_matrix, args.k, dim=1)

            for i in range(total_size):
                for j in top_k_indices[i]:
                    if i == j:  
                        continue
                    sim = sim_matrix[i, j]
                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim
            return adj_matrix

        return sim_matrix

    def _from_GPT_embeddings(self, args, df_train, df_val, df_test):
        train_embeddings = np.load(f"./embeddings/gpt/{args.dataset}/train.npy")
        val_embeddings = np.load(f"./embeddings/gpt/{args.dataset}/val.npy")
        test_embeddings = np.load(f"./embeddings/gpt/{args.dataset}/test.npy")
        
        # Concatenate to single [N, D] array, then convert to torch tensor
        embeddings = np.concatenate([train_embeddings, val_embeddings, test_embeddings])
        print(embeddings.shape)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)  # [N, D]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalize
        
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        if args.modify_graph:
            total_size = sim_matrix.size(0)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.float32)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            _, top_k_indices = torch.topk(sim_matrix, args.k, dim=1)

            for i in range(total_size):
                for j in top_k_indices[i]:
                    if i == j:  
                        continue
                    sim = sim_matrix[i, j]
                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim
            return adj_matrix

        return sim_matrix
    
    def _from_SBERT_embeddings_gaussian(self, args, df_train, df_val, df_test):
        documents = np.concatenate([df_train['text'].values, df_val['text'].values, df_test['text'].values])
        sbert_model = SentenceTransformer(args.llm)
        embeddings = sbert_model.encode(documents, convert_to_tensor=True)
        sim_matrix = gaussian_dist(embeddings, args.sigma)
        if args.modify_graph:
            total_size = sim_matrix.size(0)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.float32)

            train_labels = df_train['label'].values
            train_size = len(train_labels)

            _, top_k_indices = torch.topk(sim_matrix, args.k, dim=1)

            for i in range(total_size):
                for j in top_k_indices[i]:
                    if i == j:  
                        continue
                    sim = sim_matrix[i, j]
                    if i < train_size and j < train_size:
                        if train_labels[i] == train_labels[j]:
                            sim *= args.increase
                        else:
                            sim *= args.decrease
                    adj_matrix[i, j] = sim
                    adj_matrix[j, i] = sim

            return adj_matrix
        return sim_matrix