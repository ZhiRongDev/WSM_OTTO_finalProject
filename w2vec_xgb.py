import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

# train = pl.read_parquet('./data/local_validation/test.parquet')
train = pl.read_parquet('./data/local_validation/test.parquet')
test = pl.read_parquet('./data/test/test.parquet')

train = train.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32),
    pl.col('ts').cast(pl.datatypes.Int64)
])

test = test.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32),
    pl.col('ts').cast(pl.datatypes.Int64)
])

print(train)
print(test)

sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)

from annoy import AnnoyIndex

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
index = AnnoyIndex(32, 'euclidean')

for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx])
    
index.build(10)

import pandas as pd
import numpy as np

from collections import defaultdict

sample_sub = pd.read_csv('./data/sample_submission.csv')

session_types = ['clicks', 'carts', 'orders']
train_session_AIDs = train.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
train_session_types = train.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

# labels = []
candidates = pl.DataFrame({
    "session": [],
    "aid": []
})
candidates = candidates.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('aid').cast(pl.datatypes.Int32)
])

type_weight_multipliers = {0: 1, 1: 6, 2: 3}
for idx, (AIDs, types) in enumerate(zip(train_session_AIDs, train_session_types)):
    session_num = train_session_AIDs.index[idx]

    if len(AIDs) >= 20:
        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types): 
            aids_temp[aid]+= w * type_weight_multipliers[t]
            
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        # labels.append(sorted_aids[:20])
        for aid in sorted_aids[:20]:
            df = pl.DataFrame({
                "session": [session_num],
                "aid": [aid]
            })
            df = df.with_columns([
                pl.col('session').cast(pl.datatypes.Int32),
                pl.col('aid').cast(pl.datatypes.Int32)
            ])
            candidates = pl.concat(
                [
                    candidates,
                    df
                ],
                how="vertical",
            )

    else:
        # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        # let's grab the most recent aid
        most_recent_aid = AIDs[0]
        # and look for some neighbors!
        nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
                        
        # labels.append((AIDs+nns)[:20])

# candidates = pd.DataFrame(data={'session': train_session_AIDs.index, 'aid': labels})
# candidates = pl.DataFrame(candidates)
print(candidates)
candidates = candidates.with_column(pl.col('aid').cumcount().over('session').alias('word2vec_rank') + 1)
print(candidates)

train = train.join(candidates, on=['session', 'aid'], how='outer')
print(train)
