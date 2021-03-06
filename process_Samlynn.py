import pandas as pd
import torchtext
from torchtext import data
from Tokenize_Samlynn import tokenize
import numpy as np
from torch.autograd import Variable
# from Batch import MyIterator, batch_size_fn
import os
import dill as pickle


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def read_data(src_data, trg_data):
    if src_data is not None:
        try:
            src_data = open(src_data, 'r', encoding='UTF-8').read().strip().split('\n')
        except IOError as e:
            print(e)
            quit()

    if trg_data is not None:
        try:
            trg_data = open(trg_data, 'r', encoding='UTF=8').read().strip().split('\n')
        except IOError as e:
            print(e)
            quit()

    return src_data, trg_data

def create_fields(src_lang, trg_lang):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs)
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")

    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    # if load_weights is not None:
    #     try:
    #         print("loading presaved fields...")
    #         SRC = pickle.load(open(f'{load_weights}/SRC.pkl', 'rb'))
    #         TRG = pickle.load(open(f'{load_weights}/TRG.pkl', 'rb'))
    #     except:
    #         print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
    #         quit()

    return (SRC, TRG)


def create_dataset(src_data,trg_data,batchsize, device, max_strlen, SRC, TRG):
    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=batchsize, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # if opt.load_weights is None:
    #     SRC.build_vocab(train)
    #     TRG.build_vocab(train)
    #     if opt.checkpoint > 0:
    #         try:
    #             os.mkdir("weights")
    #         except:
    #             print("weights folder already exists, run program with -load_weights weights to load them")
    #             quit()
    #         pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
    #         pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    train_len = get_len(train_iter)

    return train_iter, train_len, src_pad, trg_pad


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i