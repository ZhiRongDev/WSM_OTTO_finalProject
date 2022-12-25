import numpy as np
import pandas as pd
import time

id2type = ['clicks', 'carts', 'orders']
type2id = {a: i for i, a in enumerate(id2type)}

# print(id2type, type2id)

pd.to_pickle(id2type, '../data/test/id2type.pkl')
pd.to_pickle(type2id, '../data/test/type2id.pkl')

def json_to_df(fn):
    start = time.time()

    sessions = []
    aids = []
    tss = []
    types = []
 
    """
        chunksize doesn't means it "only chunk n rows in a file",
        but "chunks n rows in a file once a time, and keep chunking".

        You need to use "break" to only chunk n rows.

    """
    chunks = pd.read_json(fn, lines=True, chunksize=1000)
     
    for chunk in chunks:
        # print(chunk)
        # breakpoint()

        for row_idx, session_data in chunk.iterrows():
            num_events = len(session_data.events)
            sessions += ([session_data.session] * num_events)
            for event in session_data.events:
                aids.append(event['aid'])
                tss.append(event['ts'])
                types.append(type2id[event['type']])

        break

    end = time.time()
    print(f'執行時間: {end - start} 秒\n')
    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})



train_df = json_to_df('../data/origin/train.jsonl')
train_df.type = train_df.type.astype(np.uint8)
train_df.to_parquet('../data/test/train.parquet', index=False)
train_df.to_csv('../data/test/train.csv', index=False)

del train_df

test_df = json_to_df('../data/origin/test.jsonl')
test_df.type = test_df.type.astype(np.uint8)
test_df.to_parquet('../data/test/test.parquet', index=False)
test_df.to_csv('../data/test/test.csv', index=False)
