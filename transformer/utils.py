import torch
import torch.nn as nn
import pickle
import time
import math
import re
import string
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def preprocessing(data):
    stemmer = PorterStemmer()

    raw = []

    for entry in data:
        entry = re.sub('\d+', 'NUM', entry)  # remove numbers
        entry = entry.translate(
            entry.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # remove punctuation
        entry = word_tokenize(entry)  # tokenize
        entry = [stemmer.stem(word) for word in entry]  # stemming
        entry = [x if x != 'num' else 'NUM' for x in entry]

        raw.append(entry)

    return raw


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair)]


def load_emb_from_file(lang, emb_file):

    with open(emb_file,'rb') as f:
        embs = pickle.load(f)
    first_name=list(embs.keys())[0]
    emb_dim = embs[first_name].shape[0]

    vocab = lang.token2id

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))

    for word in vocab:

        i = vocab[word]

        if word in embs:
            weights_matrix[i] = embs[word]
        else:
            weights_matrix[i] = np.zeros_like(embs[first_name])

    return weights_matrix


def create_emb_layer(weights_matrix, non_trainable=False):

    weights_matrix = torch.Tensor(weights_matrix)

    num_embeddings, embedding_dim = weights_matrix.size()

    emb_layer = nn.Embedding(num_embeddings, embedding_dim)

    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else lang.word2index['UNK'] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.word2index["EOS"])
    indexes = padding(indexes,lang)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, device):

    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


def padding(indexes, lang):
    return indexes + [lang.word2index['PAD']]* ( lang.max_length - len(indexes))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
