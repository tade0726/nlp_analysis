# -*- coding: utf-8 -*-

"""
Date: 2017-04-19
Author: Ted
e-mail: zp913913@gmail.com

Here is some tokenizers:
- jieba
- thulac

"""


import jieba
import string
from gensim import utils
from os.path import realpath, split, join
from functools import partial


class Config:
    work_dir = split(split(realpath(__file__))[0])[0]


class Tokenizer:

    def __init__(self):

        self.stw_path = join(Config.work_dir, 'data', 'tok', 'stopwords.txt')
        self.dict_path =  join(Config.work_dir, 'data', 'tok', 'dictionary.txt')
        self.stopwords = self._read_txt(self.stw_path)
        # loading dictionary
        jieba.load_userdict(self.dict_path)
        self._jieba_token = partial(jieba.lcut, cut_all=0, HMM=1, )

    @staticmethod
    def _read_txt(file_path):
        """
        des: reading file
        :param file_path: self explain
        :return: return list of words in utf-8 encoding
        """
        with open(file_path, 'rt', encoding='utf-8') as file:
            ret_list = [x.strip('\n') for x in file.readlines()]
        return ret_list

    @staticmethod
    def _filter_punctuation(line):
        # cleaning punctuation
        punctuation = \
            set(''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳丨﹐､﹒\
                ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠\
                々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻\
                ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''') | \
            set(string.punctuation)

        return list(filter(lambda x: x not in punctuation, line))

    def _filter_stopwords(self, line):
        # cleaning punctuation and stop-words
        stopwords = \
            set(self.stopwords)

        return list(filter(lambda x: x not in stopwords, line))

    @staticmethod
    def _get_stem(line: str):
        """
        des: clean the encoding and convert to lower case
        :param line: 
        :return: 
        """
        line = utils.to_unicode(line)
        return line.lower().strip()

    def jieba_token(self, line):
        token_lst = self._jieba_token(line)
        # clear the punctuation
        token_lst_keep = self._filter_punctuation(token_lst)
        return token_lst_keep

    def get_token(self, line):
        line = self._get_stem(line)
        return self._filter_stopwords(self.jieba_token(line))

if __name__ == '__main__':
    txt = \
        "开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率"
    token = Tokenizer()
    print(token.get_token(txt))
