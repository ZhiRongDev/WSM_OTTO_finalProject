import os, sys, time
import polars as pl
import pandas as pd
import numpy as np

start = time.time()

# train_df = pd.read_parquet('./data/train/train.parquet')
test_df = pl.read_parquet('./data/test/test.parquet')

# list must be string
sentences_df = pl.concat([test_df]).groupby('session').agg(
    pl.col('aid').alias('sentence').cast(pl.Utf8, strict=False)
)
sentences = sentences_df['sentence'].to_list()

# print(sentences)

from gensim.models import TfidfModel
from gensim.corpora import Dictionary, dictionary

dictionary = Dictionary(sentences)

print(dictionary)


bow_corpus = [dictionary.doc2bow(text) for text in sentences]


# train the model
tfidf = TfidfModel(bow_corpus)

print(tfidf[dictionary.doc2bow(sentences[40])])








# corpus = [dictionary.doc2bow(line) for line in sentences]
# model = TfidfModel(corpus)

# vector = model[corpus[0]]
# print(vector)

# end = time.time()
# print(f'\ntime = {start - end}')


