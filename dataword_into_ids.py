import os
import numpy as np
import re

def tokenizer(sentence):
    tokens = []
    for word in sentence.strip().split():
        tokens.extend(re.split(" ", word))

    return [w for w in tokens if w]

def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, "r") as f:
            for line in f:
                rev_vocab.append(line.strip('\n'))
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    else:
        return ValueError("File {} does not found".format(vocab_path))

def create_vocabulary(vocab_dir, data_paths):
    if not os.path.exists(os.path.join(vocab_dir, 'vocab.dat')):
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        vocab = {}
        for path in data_paths:
            with open(path, 'r') as f:
                counter = 0
                for line in f:
                    words = tokenizer(line)
                    for word in words:
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        with open(os.path.join(vocab_dir, 'vocab.dat'), 'w') as fh:
            for w in vocab_list:
                fh.write(w + b"\n")
                    
if __name__=='__main__':

    train_dir = os.path.join('data/processed/squad', 'train')
    dev_dir = os.path.join('data/processed/squad', 'dev')
    glove_dir = os.path.join('data/download', 'glove')
    vocab_dir = os.path.join('data', 'processed', 'vocab')

    create_vocabulary(vocab_dir, [train_dir + '/contexts',
                                  train_dir + '/answers',
                                  dev_dir + '/contexts',
                                  dev_dir + '/answers'])

    vocab, _ = initialize_vocab(vocab_dir + '/vocab.dat')

