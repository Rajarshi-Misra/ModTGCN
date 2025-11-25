import pandas as pd
from utils import *
from scipy.sparse import vstack
import scipy
import nltk
from nltk.corpus import stopwords

def get_adj(df_main: pd.DataFrame, word_id_map, vocab, train_size, val_size, test_size, word_doc_freq):
    window_size = 20
    windows = []
    for doc_words in df_main["text"]:
        words = doc_words.split()
        length = len(words)   
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = find_word_pair_count(windows, word_id_map)

    row, col, weight = get_pmi(windows, word_pair_count, word_window_freq, vocab, train_size, val_size)
    adj = add_tfid(row, col, weight, df_main, word_id_map, train_size, val_size, test_size, len(vocab), word_doc_freq,
                   vocab)

    return adj


def build_adj_pmi_tfidf(df_train, df_val, df_test, dataset):

    df_main = pd.concat([df_train, df_val, df_test])

    word_freq = count_freq(df_main)

    if 1:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        df_train['text'] = df_train['text'].transform(
            lambda x: remove_less_freq_words(x, word_freq, stop_words, dataset))
        df_test['text'] = df_test['text'].transform(lambda x: remove_less_freq_words(x, word_freq, stop_words, dataset))
        df_val['text'] = df_val['text'].transform(lambda x: remove_less_freq_words(x, word_freq, stop_words, dataset))
        df_main['text'] = df_main['text'].transform(lambda x: remove_less_freq_words(x, word_freq, stop_words, dataset))

    word_freq2 = count_freq(df_main)
    train_size = len(df_train)
    test_size = len(df_test)
    val_size = len(df_val)

    vocab_size = len(word_freq2)
    vocab = list(word_freq2)
    print("No. of words:",vocab_size)
    ##get indices of the sentences in which each of the word is present
    word_doc_list = {}
    get_word_doc_list(df_main, word_doc_list)

    ##get the number of occurences of each word
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    adj = get_adj(df_main, word_id_map, vocab, train_size, val_size, test_size, word_doc_freq)

    D1 = adj[0:train_size + val_size, train_size + val_size:train_size + val_size + vocab_size]
    D2 = adj[train_size + val_size + vocab_size:train_size + val_size + vocab_size + test_size,
         train_size + val_size:train_size + val_size + vocab_size]
    D = vstack((D1, D2))
    W = adj[train_size + val_size:train_size + val_size + vocab_size,
        train_size + val_size:train_size + val_size + vocab_size]

    # add self loop
    W = W + scipy.sparse.eye(W.shape[0]) #why adding self loop in words

    D = D.tocoo()
    W = W.tocoo()

    D = torch.sparse_coo_tensor([D.row, D.col], D.data, size=(D.shape[0], D.shape[1]))
    W = torch.sparse_coo_tensor([W.row, W.col], W.data, size=(W.shape[0], W.shape[1]))
    return D, W
