import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

train = pl.read_parquet('../data/test/train.parquet')
test = pl.read_parquet('../data/test/test.parquet')


sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)


######
from gensim.similarities.annoy import AnnoyIndexer
# 100 trees are being used in this example
annoy_index = AnnoyIndexer(w2vec, 10)
# Derive the vector for the word "science" in our model
vector = w2vec.wv[1]

print(w2vec.wv.index_to_key[1])

# The instance of AnnoyIndexer we just created is passed
approximate_neighbors = w2vec.wv.most_similar([vector], topn=11, indexer=annoy_index)
# Neatly print the approximate_neighbors and their corresponding cosine similarity values

print("Approximate Neighbors")
for neighbor in approximate_neighbors:
    print(neighbor)

normal_neighbors = w2vec.wv.most_similar([vector], topn=11)
print("\nExact Neighbors")
for neighbor in normal_neighbors:
    print(neighbor)

#####
from annoy import AnnoyIndex

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
index = AnnoyIndex(32, 'euclidean')

for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx])
    
index.build(10)

print('\nKaggle Notebook')
# print(index.get_nns_by_item(1, 11))
for idx in index.get_nns_by_item(1, 11):
    print(w2vec.wv.index_to_key[idx])
