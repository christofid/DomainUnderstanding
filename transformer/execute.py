from train import *
from utils import *
from dataset import *
from Encoder import *
from Decoder import *
from seq2seq import *
from translate import *
from train import *
from order_labels import *
from configure import parse_args
from evaluator import evaluate_case

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pickle
import random
import math

def execute_case(data_path, order_type='random', root_node=None, output_path=None, model_path=None, embedding_path=None, words_path='words_dictionary',
                 batch_size=128, enc_layers=3, enc_heads=10, dec_layers=3, dec_heads=10, enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.1,
                 dec_dropout=0.1, learning_rate=0.0005, epochs=20, clip_value=1, seed_value=1):

    with open(data_path, 'rb') as f:
        loaded = pickle.load(f)

    with open(words_path, 'rb') as f:
        dictionary = pickle.load(f)

    txt = loaded['txt']
    txt = preprocessing(txt)

    words = loaded['words']

    labels = loaded['labels']

    if order_type == 'bfs':
        new_order = find_ordering(words, order_type, root_node)
        labels = reorder_labels(labels, new_order)
        words = new_order

    if order_type=='random':

        for i,x in enumerate(labels):
            random.shuffle(x)
            labels[i]=x

    max_txt_len = max([len(x) for x in txt])+2
    max_label_len = max([len(x) for x in labels])+2

    train_ids, eval_ids = train_test_split(list(range(len(labels))), test_size=0.4, random_state=seed_value)
    valid_ids, test_ids = train_test_split(list(range(len(eval_ids))), test_size=0.5, random_state=0)

    train_txt = [txt[i] for i in train_ids]
    train_labels = [labels[i] for i in train_ids]

    valid_txt = [txt[ eval_ids[i] ] for i in valid_ids]
    valid_labels = [labels[ eval_ids[i] ] for i in valid_ids]

    test_txt = [txt[ eval_ids[i] ] for i in test_ids]
    test_labels = [labels[ eval_ids[i] ] for i in test_ids]


    txt_lang = build_vocabulary(dictionary)
    labels_lang = build_vocabulary(words)

    train_data = NMTDataset(train_txt, train_labels, txt_lang, labels_lang)
    valid_data = NMTDataset(valid_txt, valid_labels, txt_lang, labels_lang)
    test_data = NMTDataset(test_txt, test_labels, txt_lang, labels_lang)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn)

    valid_iterator = DataLoader(dataset=valid_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)

    test_iterator = DataLoader(dataset=test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)

    weight_matrix = load_emb_from_file(txt_lang, embedding_path)
    pretrained_emb, _, hidden_size = create_emb_layer(weight_matrix)

    INPUT_DIM = len(dictionary)+4 #input vocab
    OUTPUT_DIM = len(words)+4     #output vocab

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
                  max_label_len)

    SRC_PAD_IDX = txt_lang.token2id['<PAD>']
    TRG_PAD_IDX = labels_lang.token2id['<PAD>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    print('The model has {} trainable parameters'.format(count_parameters(model)))

    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    trainer(model, train_iterator, valid_iterator, optimizer, criterion, epochs, clip_value,model_path)

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    evaluate_case(test_txt, test_labels, words, txt_lang, labels_lang, model, device, max_label_len, save_pred=output_path)


if __name__ == '__main__':

    args = parse_args()
    execute_case(**args)