import os
import numpy as np


def create_vocabulary(vocab_dir, data_paths):
    if not os.path.exists(os.path.join(vocab_dir, 'vocab.dat')):
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        vocab = {}
        for path in data_paths:
            with open(path, 'r') as f:
                counter = 0
                for line in f:
                    words = line.split(' ')
                    for word in words:
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        with open(os.path.join(vocab_dir, 'vocab.dat'), 'w') as fh:
            for w in vocab_list:
                fh.write(w + "\n")
                    
if __name__=='__main__':

    train_dir = os.path.join('data/processed/squad', 'train')
    dev_dir = os.path.join('data/processed/squad', 'dev')
    glove_dir = os.path.join('data/download', 'glove')
    vocab_dir = os.path.join('data', 'processed', 'vocab')

    create_vocabulary(vocab_dir, [train_dir + '/contexts',
                                  train_dir + '/answers',
                                  dev_dir + '/contexts',
                                  dev_dir + '/answers'])

