# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
https://www.kaggle.com/abdelrahmanzied/ham-or-spam-sms-text-classification
"""
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# use pipline
class TextCleaning():
    def __init__(self):
        print("call init")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        documents = []
        for sent in X:
            # Remove all single characters
            sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', sent)

            # Substituting multiple spaces with single space
            sent = re.sub(r'\s+', ' ', sent, flags=re.I)

            doc = nlp(sent)

            document = [token.lemma_ for token in doc]

            document = ' '.join(document)

            documents.append(document)
        return documents
EmailClassification = Pipeline([('TextCleaning', TextCleaning()),
                                ('Vectorizer', TfidfVectorizer()),
                                # ('NB', MultinomialNB())
                                ])
if __name__ == '__main__':
    pass
