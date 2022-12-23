import os, sys, time
import polars as pl
import pandas as pd
import numpy as np

start = time.time()

train_df = pl.read_parquet('./data/train/train.parquet')
test_df = pl.read_parquet('./data/train/test.parquet')

# list must be string
sentences_df = pl.concat([train_df, test_df]).groupby('session').agg(
    pl.col('aid').alias('sentence').cast(pl.Utf8, strict=False)
)
sentences = sentences_df['sentence'].to_list()


from gensim.models import TfidfModel
from gensim.corpora import Dictionary, dictionary

dictionary = Dictionary(sentences)

# print(dictionary.token2id)

bow_corpus = [dictionary.doc2bow(text) for text in sentences]

# train the model
tfidf = TfidfModel(bow_corpus)

# print(tfidf[dictionary.doc2bow(sentences[40])])

# index the tfidf vector of corpus as sparse matrix
from gensim import similarities
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))

def get_closest_n(query, n):
    '''get the top matching docs as per cosine similarity
    between tfidf vector of query and all docs'''
    print(f'{query=}')

    query_document = query
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    top_idx = sims.argsort()[-1*n:][::-1]
    return [sentences[i] for i in top_idx]

for d in get_closest_n(sentences[2],2):
    print(f'\n {d}')


end = time.time()
print(f'\ntime = {end - start}')
