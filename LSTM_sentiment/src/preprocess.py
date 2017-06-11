# -*- coding: utf-8 -*-

"""
author: Ted
data: 2017-06-10
des: a script to convert raw data into number seq

todo:

- build dictionary
- get pos/neg train/test dataset
- 

"""

from src.token_tools import Tokenizer
from os.path import realpath, split, join
import glob
import numpy

import pickle as pkl

from collections import Counter


class Config:
    work_dir = split(split(realpath(__file__))[0])[0]
    raw_txt_dir = join(work_dir, 'data', 'raw_corpus')


class TextPreprocess:

    def __init__(self, ):
        self.tokenize = Tokenizer().get_token
        self.dictionary = self.build_dict()

    def build_dict(self):

        sentences = []
        file_ns = glob.glob(join(Config.raw_txt_dir, '*.txt'))

        for file_n in file_ns:
            with open(file_n, 'rt') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    sentences.append(self.tokenize(line))

        print("Building dictionary..")
        wordcount = Counter()
        for idx, words in enumerate(sentences):
                wordcount.update(words)

        counts = list(wordcount.values())
        keys = list(wordcount.keys())

        sorted_idx = numpy.argsort(counts)[::-1]

        worddict = dict()

        for idx, ss in enumerate(sorted_idx):
            worddict[keys[ss]] = idx + 2  # leave 0 and 1 (UNK) for later use

        print(numpy.sum(counts), ' total words ', len(keys), ' unique words')
        return worddict

    def grab_data(self, file_name):

        sentences = []
        with open(join(Config.raw_txt_dir, file_name), 'rt') as f:
            for line in f.readlines():
                line = line.strip('\n')
                sentences.append(self.tokenize(line))

        seqs = [None] * len(sentences)
        for idx, words in enumerate(sentences):
            seqs[idx] = [self.dictionary[w] if w in self.dictionary else 1 for w in words]

        return seqs

    def main(self):

        train_x_pos = self.grab_data('pos_train.txt')
        train_x_neg = self.grab_data('neg_train.txt')
        train_x = train_x_pos + train_x_neg
        train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

        test_x_pos = self.grab_data('pos_test.txt')
        test_x_neg = self.grab_data('pos_test.txt')
        test_x = test_x_pos + test_x_neg
        test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

        with open(join(Config.work_dir, 'data', 'comment.pkl'), 'wb') as f:
            pkl.dump((train_x, train_y), f, -1)
            pkl.dump((test_x, test_y), f, -1)

        with open(join(Config.work_dir, 'data', 'comment.dict.pkl'), 'wb') as f:
            pkl.dump(self.dictionary, f, -1)


if __name__ == '__main__':
    text_preprocess = TextPreprocess()
    text_preprocess.main()