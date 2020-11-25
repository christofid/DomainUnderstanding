from train import *
from utils import *
from dataset import *
from Encoder import *
from Decoder import *
from seq2seq import *
from translate import *
from train import *
from order_labels import *
from configure import parse_args_attention_extractor

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import pickle
import random
import spacy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def analyze_txt(txt, head, nlp, model, txt_lang, labels_lang, device, max_label_len):

    doc = nlp(' '.join(txt))

    tokens = [token.text for token in doc if not token.is_punct]

    adj = np.eye(len(tokens) + 2) * 2
    adj[0, 0] = 0
    adj[-1, -1] = 0

    for token in doc:
        if not token.is_punct:
            x = tokens.index(token.text) + 1
            for child in token.children:
                if not child.is_punct:
                    y = tokens.index(child.text) + 1
                    adj[x, y] = 1
                    adj[y, x] = 1

    translation, attention, _ = translate_sentence(txt, txt_lang, labels_lang, model, device, max_len=max_label_len)

    _attention = attention.squeeze(0)[head].cpu().detach().numpy()

    _attention = np.matmul(adj, np.transpose(_attention))
    _attention = np.transpose(_attention)

    extract_relations(txt, translation, _attention, doc)

    display_attention(txt, translation, attention, adj, head=head)



def display_attention(sentence, translation, attention, adj, head):

    _attention = attention.squeeze(0)[head].cpu().detach().numpy()

    if _attention.shape[1]==len(sentence)+2:

        fig,ax = plt.subplots(figsize=(20,10))

        _attention = np.matmul(adj, np.transpose(_attention))
        _attention = np.transpose(_attention)

        ax.matshow(_attention, cmap='gist_yarg')

        ax.tick_params(labelsize=12)
        ax.set_yticklabels([''] + translation)


        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=90)


        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()


def extract_relations(sentence, translation, attention, doc):

    print()
    print('-------------------')

    for i,x in enumerate(translation[:-1]):
        print()
        print(x)

        chunks=[]

        noun_chunks = [x for x in doc.noun_chunks]

        for chunk in noun_chunks:

            val=[]

            chunk_txt = chunk.text.split(' ')

            for t_idx,token in enumerate(chunk):

                token_str=chunk_txt[t_idx]
                j = sentence.index(token_str)

                if token.is_stop:
                    val.append(attention[i,j+1])
                else:
                    val.append(2*attention[i, j + 1])

            chunks.append(np.sum(val))

        print('### nouns:      ', noun_chunks)
        print('### attentions: ', chunks)
        print('### entities:')
        for j in np.argsort(chunks)[-2:]:
            print(noun_chunks[j].text)

    print('-------------------')



def execute(model_path, data_path, order_type='random', root_node=None, embedding_path=None, words_path='words_dictionary',
            enc_layers=3, enc_heads=10, dec_layers=3, dec_heads=10, enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.1, dec_dropout=0.1, head=0):

    nlp = spacy.load("en_core_web_lg")

    with open(data_path, 'rb') as f:
        loaded = pickle.load(f)

    with open(words_path, 'rb') as f:
        dictionary = pickle.load(f)

    txts = loaded['txt']

    words = loaded['words']

    labels = loaded['labels']

    if order_type == 'bfs':
        new_order = find_ordering(words, order_type, root_node)
        labels = reorder_labels(labels, new_order)
        words = new_order

    if order_type == 'random':

        for i, x in enumerate(labels):
            random.shuffle(x)
            labels[i] = x

    max_txt_len = max([len(x) for x in preprocessing(txts)]) + 2
    max_label_len = max([len(x) for x in labels]) + 2

    train_ids, eval_ids = train_test_split(list(range(len(labels))), test_size=0.4, random_state=1)
    valid_ids, test_ids = train_test_split(list(range(len(eval_ids))), test_size=0.5, random_state=0)

    test_txts = preprocessing([txts[eval_ids[i]] for i in test_ids])

    txt_lang = build_vocabulary(dictionary)
    labels_lang = build_vocabulary(words)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weight_matrix = load_emb_from_file(txt_lang, embedding_path)
    pretrained_emb, _, hidden_size = create_emb_layer(weight_matrix)

    INPUT_DIM = len(dictionary) + 4  # input vocab
    OUTPUT_DIM = len(words) + 4  # output vocab

    enc = Encoder(INPUT_DIM,
                  hidden_size,
                  enc_layers,
                  enc_heads,
                  enc_pf_dim,
                  enc_dropout,
                  device,
                  pretrained_emb=pretrained_emb,
                  max_length=max_txt_len)

    dec = Decoder(OUTPUT_DIM,
                  hidden_size,
                  dec_layers,
                  dec_heads,
                  dec_pf_dim,
                  dec_dropout,
                  device,
                  10)

    SRC_PAD_IDX = txt_lang.token2id['<PAD>']
    TRG_PAD_IDX = labels_lang.token2id['<PAD>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))

    for txt in test_txts:
        analyze_txt(txt, head, nlp, model, txt_lang, labels_lang, device, max_label_len)


if __name__ == '__main__':

    args = parse_args_attention_extractor()
    execute(**args)