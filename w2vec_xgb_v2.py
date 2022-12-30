import time
import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

# train = pl.read_parquet('./data/local_validation/test.parquet')
train = pl.read_parquet('./data/test/train.parquet')
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

# print(f'{train=}')
# print(f'{test=}')

sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

print("w2vec start")
start = time.time()

w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)

print("w2vec end")
end = time.time()
print(f'執行時間: {end - start} 秒\n')

from annoy import AnnoyIndex

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
index = AnnoyIndex(32, 'euclidean')

for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx])
    
index.build(10)

import pandas as pd
import numpy as np

from collections import defaultdict

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

print("candidate calc start")
start = time.time()

type_weight_multipliers = {0: 1, 1: 6, 2: 3}
for idx, (AIDs, types) in enumerate(zip(train_session_AIDs, train_session_types)):
    session_num = train_session_AIDs.index[idx]    

    # print(session_num)

    if len(AIDs) >= 20:

        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types): 
            aids_temp[aid]+= w * type_weight_multipliers[t]
            
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]

        if len(sorted_aids) < 20:
            AIDs = list(dict.fromkeys(AIDs[::-1]))
            # let's grab the most recent aid
            most_recent_aid = AIDs[0]
            # and look for some neighbors!
            nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
            sorted_aids = (sorted_aids + nns)[:20]

        if session_num < 8:
            print(session_num)
            print(f'{sorted_aids[:20]=}')

        session_arr = np.full((20), session_num)
        df = pl.DataFrame({
            "session": session_arr,
            "aid": sorted_aids[:20]
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

        # for aid in sorted_aids[:20]:
        #     df = pl.DataFrame({
        #         "session": [session_num],
        #         "aid": [aid]
        #     })
        #     df = df.with_columns([
        #         pl.col('session').cast(pl.datatypes.Int32),
        #         pl.col('aid').cast(pl.datatypes.Int32)
        #     ])
        #     candidates = pl.concat(
        #         [
        #             candidates,
        #             df
        #         ],
        #         how="vertical",
        #     )

    else:
        # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        # let's grab the most recent aid
        most_recent_aid = AIDs[0]
        # and look for some neighbors!
        nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
        
        session_arr = np.full((20), session_num)

        df = pl.DataFrame({
            "session": session_arr,
            "aid": (AIDs+nns)[:20]
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
        # for aid in (AIDs+nns)[:20]:
        #     df = pl.DataFrame({
        #         "session": [session_num],
        #         "aid": [aid]
        #     })
        #     df = df.with_columns([
        #         pl.col('session').cast(pl.datatypes.Int32),
        #         pl.col('aid').cast(pl.datatypes.Int32)
        #     ])
        #     candidates = pl.concat(
        #         [
        #             candidates,
        #             df
        #         ],
        #         how="vertical",
        #     )
        
        # labels.append((AIDs+nns)[:20])  
    # print(candidates)

# candidates = pd.DataFrame(data={'session': train_session_AIDs.index, 'aid': labels})
# candidates = pl.DataFrame(candidates)
print("candidate calc end")
end = time.time()
print(f'執行時間: {end - start} 秒\n')


# print(f'{candidates=}')
candidates = candidates.with_column(pl.col('aid').cumcount().over('session').alias('word2vec_rank') + 1)
# print(f'{candidates=}')

train = train.join(candidates, on=['session', 'aid'], how='outer').sort("session")
# print(f'{train=}')
print(f'{train.filter(pl.col("session") == 7)=}')
