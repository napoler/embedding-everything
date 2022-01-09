# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
https://fasttext.cc/docs/en/cheatsheet.html
https://fasttext.cc/docs/en/unsupervised-tutorial.html

download https://fasttext.cc/docs/en/crawl-vectors.html
"""
import os
import fasttext
import fasttext.util
ft = fasttext.load_model('cc.en.300.bin')
ft.get_dimension()

fasttext.util.reduce_model(ft, 100)
ft.get_dimension()

vec=ft.get_word_vector('hello').shape
print(vec)
ft.get_nearest_neighbors('hello')



if __name__ == '__main__':
    pass
