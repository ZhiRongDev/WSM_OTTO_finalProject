import time
import pandas as pd
import polars as pl
import numpy as np
from collections import defaultdict
import cudf
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)

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

##### for Word2Vec pretrain
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
#####

###
train_labels = pl.read_parquet('./data/local_validation/test_labels.parquet')

def word2vec_candidate(df): 
    global index
    session_types = ['clicks', 'carts', 'orders']
    df_session_AIDs = df.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
    df_session_types = df.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)
    
    candidates = cudf.DataFrame({"session": [], "aid":[]})

    print("candidate calc start")
    start = time.time()

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    for idx, (AIDs, types) in enumerate(zip(df_session_AIDs, df_session_types)):
        session_num = df_session_AIDs.index[idx]    
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

            session_arr = np.full((20), session_num)        

            session_df = cudf.DataFrame({
                "session": session_arr,
                "aid": sorted_aids[:20] 
            })
            candidates = cudf.concat([candidates, session_df])
        else:
            # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
            AIDs = list(dict.fromkeys(AIDs[::-1]))
            # let's grab the most recent aid
            most_recent_aid = AIDs[0]
            # and look for some neighbors!
            nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
            session_arr = np.full((20), session_num)
            session_df = cudf.DataFrame({
                "session": session_arr,
                "aid": (AIDs+nns)[:20] 
            })
            
            candidates = cudf.concat([candidates, session_df])

    print("candidate calc end")
    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

    candidates = candidates.to_pandas()
    candidates = pl.DataFrame(candidates)
    candidates = candidates.with_column(pl.col('aid').cumcount().over('session').alias('word2vec_rank') + 1)
    candidates = candidates.with_columns([
        pl.col('session').cast(pl.datatypes.Int32),
    ])
     
    df = df.join(candidates, on=['session', 'aid'], how='outer').sort("session")
    return df

def add_action_num_reverse_chrono(df):
    return df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
    ])

def add_session_length(df):
    return df.select([
        pl.col('*'),
        pl.col('session').count().over('session').alias('session_length')
    ])

def add_log_recency_score(df):
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    return df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)

def add_type_weighted_log_recency_score(df):
    type_weights = {0:1, 1:6, 2:3}
    type_weighted_log_recency_score = pl.Series(df['log_recency_score'] / df['type'].apply(lambda x: type_weights[x]))
    return df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df

pipeline = [add_action_num_reverse_chrono, add_session_length, add_log_recency_score, add_type_weighted_log_recency_score, word2vec_candidate]

train = apply(train, pipeline)

# print(train.columns)

type2id = {"clicks": 0, "carts": 1, "orders": 2}

train_labels = train_labels.explode('ground_truth').with_columns([
    pl.col('ground_truth').alias('aid'),
    pl.col('type').apply(lambda x: type2id[x])
])[['session', 'type', 'aid']]

train_labels = train_labels.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32)
])

train_labels = train_labels.with_column(pl.lit(1).alias('gt'))

train = train.join(train_labels, how='left', on=['session', 'type', 'aid']).with_column(pl.col('gt').fill_null(0))

def get_session_lenghts(df):
    return df.groupby('session').agg([
        pl.col('session').count().alias('session_length')
    ])['session_length'].to_numpy()

session_lengths_train = get_session_lenghts(train)

from lightgbm.sklearn import LGBMRanker

ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=20,
    importance_type='gain',
)

feature_cols = ['aid', 'type', 'action_num_reverse_chrono', 'session_length', 'log_recency_score', 'type_weighted_log_recency_score', 'word2vec_rank']
target = 'gt'

ranker = ranker.fit(
    train[feature_cols].to_pandas(),
    train[target].to_pandas(),
    group=session_lengths_train,
)

test = apply(test, pipeline)
print(f'{test.columns=}')

scores = ranker.predict(test[feature_cols].to_pandas())

test = test.with_columns(pl.Series(name='score', values=scores))
test_predictions = test.sort(['session', 'score'], reverse=True).groupby('session').agg([
    pl.col('aid').limit(20).list()
])

session_types = []
labels = []

for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid'].to_numpy()):
    l = ' '.join(str(p) for p in preds)
    for session_type in ['clicks', 'carts', 'orders']:
        labels.append(l)
        session_types.append(f'{session}_{session_type}')

submission = pl.DataFrame({'session_type': session_types, 'labels': labels})
submission.write_csv('submission.csv')

