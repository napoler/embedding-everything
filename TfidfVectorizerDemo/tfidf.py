# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

将原始文档集合转换为 TF-IDF 特征矩阵。
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# from spacy.lang.en.stop_words import STOP_WORDS
# Vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)

Vectorizer = TfidfVectorizer()
X = ["hello word", "Sorry, I'll call later", "Ard 6 like dat lor."]
X = Vectorizer.fit_transform(X)
# X = Vectorizer.fit_transform(X, y)
print(X.shape)
print(X)

z = Vectorizer.transform(['Hi Abdelrahman Abozied, I have a job offer for you as a Machine Learning Engineer'])
print("z", dir(z))
print(z.shape)









if __name__ == '__main__':
    pass
