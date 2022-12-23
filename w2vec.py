import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

train = pl.read_parquet('./data/test/train.parquet')
test = pl.read_parquet('./data/test/test.parquet')

sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

w2vec = Word2Vec(sentences=sentences, vector_size=100, min_count=1, workers=4)

########
from gensim.similarities.annoy import AnnoyIndexer

# 100 trees are being used in this example
annoy_index = AnnoyIndexer(w2vec, 100)

########

import pandas as pd
import numpy as np

from collections import defaultdict

sample_sub = pd.read_csv('./data/sample_submission.csv')

session_types = ['clicks', 'carts', 'orders']
test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

labels = []

type_weight_multipliers = {0: 1, 1: 6, 2: 3}
for AIDs, types in zip(test_session_AIDs, test_session_types):
    if len(AIDs) >= 20:
        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types): 
            aids_temp[aid]+= w * type_weight_multipliers[t]
            
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:20])
    else:
        # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        
        # let's grab the most recent aid
        most_recent_aid = AIDs[0]
        
        # print(most_recent_aid)
    
        # and look for some neighbors!
        # nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
        vector = w2vec.wv[most_recent_aid]
        nns = [i for i, j in w2vec.wv.most_similar([vector], topn=21, indexer=annoy_index)]
        nns.pop(0)

        labels.append((AIDs+nns)[:20])

labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv('submission.csv', index=False)