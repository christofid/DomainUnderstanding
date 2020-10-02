from utils import *
import pickle
import torch
from collections import Counter
from torch.utils.data import Dataset
from tqdm import tqdm

PAD = 0
BOS = 1
EOS = 2
UNK = 3

class AttrDict(dict):

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class NMTDataset(Dataset):
    def __init__(self, source, target, src_vocab=None, tgt_vocab=None, max_vocab_size=50000, share_vocab=True):

        print('=' * 100)
        print('Dataset preprocessing log:')

        print('- Loading and tokenizing source sentences...')
        self.src_sents = source
        print('- Loading and tokenizing target sentences...')
        self.tgt_sents = target

        if src_vocab is None or tgt_vocab is None:
            print('- Building source counter...')
            self.src_counter = self.build_counter(self.src_sents)
            print('- Building target counter...')
            self.tgt_counter = self.build_counter(self.tgt_sents)

            if share_vocab:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter + self.tgt_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.src_vocab
            else:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.build_vocab(self.tgt_counter, max_vocab_size)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            share_vocab = src_vocab == tgt_vocab

        print('=' * 100)
        print('Dataset Info:')
        print('- Number of source sentences: {}'.format(len(self.src_sents)))
        print('- Number of target sentences: {}'.format(len(self.tgt_sents)))
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.token2id)))
        print('- Target vocabulary size: {}'.format(len(self.tgt_vocab.token2id)))
        print('- Shared vocabulary: {}'.format(share_vocab))
        print('=' * 100 + '\n')

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, append_BOS=True, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token2id, append_BOS=True, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq

    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter

    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update(
            {token: _id + 4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        return vocab

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq


def collate_fn(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)

    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    data.sort(key=lambda x: len(x[0]), reverse=True)

    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)

    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)

    return  src_seqs, tgt_seqs


def build_vocabulary(words):
    vocab = AttrDict()
    vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
    vocab.token2id.update(
        {word: _id + 4 for _id, word in tqdm(enumerate(words))})
    vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
    return vocab
