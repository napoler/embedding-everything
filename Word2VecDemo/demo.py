# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

https://radimrehurek.com/gensim/models/word2vec.html
"""
import os
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


mode_path="/tmp/word2vec.model"
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
model.save(mode_path)



model = Word2Vec.load(mode_path)
# tain
model.train([["hello", "world"]], total_examples=1, epochs=1)



# 多词 ngram 的嵌入
# 有一个gensim.models.phrases模块可以让您使用搭配统计自动检测长度超过一个单词的短语。使用短语，您可以学习 word2vec 模型，其中“单词”实际上是多词表达式，例如new_york_times或


from gensim.models import Phrases

# Train a bigram detector.
bigram_transformer = Phrases(corpus)

# Apply the trained MWE detector to a corpus, using the result to train a Word2vec model.
model = Word2Vec(bigram_transformer[common_texts], min_count=1)









from gensim.models import Word2Vec
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model = Word2Vec(min_count=1)
model.build_vocab(sentences)  # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors


vector = model.wv['meow']
print(vector)
if __name__ == '__main__':
    pass
