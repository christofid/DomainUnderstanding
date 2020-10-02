from train import *
from utils import *
from dataset import *
from Encoder import *
from Decoder import *
from seq2seq import *
from translate import *
from train import *
from order_labels import *

from configure import parse_args_visualizer

import torch
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='gist_yarg')

        ax.tick_params(labelsize=12)

        if i % n_cols != 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([''] + translation)

        if i>=2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                               rotation=90)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



def execute(model_path, data_path, order_type='random', root_node=None, embedding_path=None, words_path='words_dictionary',
            enc_layers=3, enc_heads=10, dec_layers=3, dec_heads=10, enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.1, dec_dropout=0.1):

    with open(data_path, 'rb') as f:
        loaded = pickle.load(f)

    with open(words_path, 'rb') as f:
        dictionary = pickle.load(f)

    words = loaded['words']
    if order_type == 'bfs':
        new_order = find_ordering(words, order_type, root_node)
        words = new_order

    max_txt_len = max([len(x) for x in preprocessing(loaded['txt'])])+2

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

    while 1:

        print()
        txt = input()

        txt_pro = preprocessing([txt])[0]

        translation, attention,_ = translate_sentence(txt_pro, txt_lang, labels_lang, model, device, max_len=10)

        print('Found relation types: {}'.format(translation[:-1]))

        display_attention(txt.split(' '), translation, attention, n_heads=enc_heads, n_rows=5, n_cols=2)


if __name__ == '__main__':

    args = parse_args_visualizer()
    execute(**args)
