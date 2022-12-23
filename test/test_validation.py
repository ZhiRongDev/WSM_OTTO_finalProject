import polars as pl

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

train = pl.read_parquet('./data/local_validation/train.parquet')
test = pl.read_parquet('./data/local_validation/test.parquet')

print(test)
print(train)
breakpoint()
