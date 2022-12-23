# Gensim 官方文件學習筆記

## 目錄：
[TOC]
<!-- - Gensim 介紹
- Gensim 操作
- Gensim 核心概念
- 作業練習
- 參考資料
 -->
## Gensim 介紹 (Introduction to Gensim)
### What is Gensim?
Gensim 是一個免費且開放的 Python 套件，用來將文檔document 表示成語意(semantic)向量。

![](https://radimrehurek.com/gensim/_images/gensim_logo_positive_complete_tb.png)

它是使用unsupervised machine learning處理原始的、非結構化的文本（text），藉由統計training documents在語料庫(corpus)中字與字之間組合或共同出現(co-occurrence)的模式，自動理解 documents 的語義結構。

- 資料參考: [1]

### Gensim 安裝
本次使用 Python 3.8 版的環境，安裝版是 Gensim 4.0.1
```
pip install --upgrade gensim
```

## Gensim 核心概念 (Core Concepts of Gensim)

- Gensim 核心概念有 `Document`、`Corpus`、`Vector`、`Model`。

- **Document (文章)**: 一群文字。可以是 140 個字的簡短推文、單個段落（即期刊文章摘要）、新聞文章或書籍中的任何內容。
  ```python
  document = "Human machine interface for lab abc computer applications"
  ```
  
- **Corpus (語料庫)**: Document物件的集合。
  - 它在Gensim中扮演兩個角色：

      1. 當做 model 的輸入。Model 藉由 training corpus 來初始化模型內部的參數。

      2. 用來組織 document。當一個 topic model `(ex:關鍵字提取)` 被訓練後，可以從新的 document `(在training corpus中未看到的document)` 中提取 topic `(ex:新的關鍵字)`。這樣的 corpus 可以為相似性查詢建立索引、為語義相似性提供查詢或聚類等。

  - 舉例一個由9個 documents 所組成的 corpus。每一個 document 是由一段 single sentence 所構成的字串。
      ```python
      text_corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]
      ```
  - 上述是說明語料庫的一個小的例子，此外 corpus 也可以是`莎士比亞所寫的所有戲劇的列表`、`所有維基百科文章的列表`或是`某個特定感興趣的人的所有推文`。
  
  - 收集完 corpus 後，我們通常要進行許多預處理步驟來保持 corpus 的簡單。
    - 刪除一些常用的 word，或稱停用詞`(Stop Words)`，例如: 英文中的`the`)。
    - 刪除在 corpus 中僅出現一次的 word。
    
    在這樣的過程中我們將對資料標記，透過標記化將 documents 拆解成數份 words。
    
    - 在英文句子中最有效率的方法就是從**空白(space)** 的地方切分。
    - 中文句子則需要使用**中文分詞**工具，例如：[結巴](https://github.com/fxsjy/jieba)。

    Example:
    ```python
    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]
    pprint.pprint(texts) # Out 1

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    pprint.pprint(frequency) # Out 2

    # Only keep words that appear more than once
    import pprint
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    pprint.pprint(processed_corpus) # Out 3

    ```
    
    :::spoiler Out 1: `pprint.pprint(texts)`
    ```python
    [['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'],
     ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'management', 'system'],
     ['system', 'human', 'system', 'engineering', 'testing', 'eps'],
     ['relation', 'user', 'perceived', 'response', 'time', 'error', 'measurement'],
     ['generation', 'random', 'binary', 'unordered', 'trees'],
     ['intersection', 'graph', 'paths', 'trees'],
     ['graph', 'minors', 'iv', 'widths', 'trees', 'well', 'quasi', 'ordering'],
     ['graph', 'minors', 'survey']]
    ```
    :::
    
    :::spoiler Out 2: `pprint.pprint(frequency)`
    ```python
    defaultdict(<class 'int'>,
                {'abc': 1,
                 'applications': 1,
                 'binary': 1,
                 'computer': 2,
                 'engineering': 1,
                 'eps': 2,
                 'error': 1,
                 'generation': 1,
                 'graph': 3,
                 'human': 2,
                 'interface': 2,
                 'intersection': 1,
                 'iv': 1,
                 'lab': 1,
                 'machine': 1,
                 'management': 1,
                 'measurement': 1,
                 'minors': 2,
                 'opinion': 1,
                 'ordering': 1,
                 'paths': 1,
                 'perceived': 1,
                 'quasi': 1,
                 'random': 1,
                 'relation': 1,
                 'response': 2,
                 'survey': 2,
                 'system': 4,
                 'testing': 1,
                 'time': 2,
                 'trees': 3,
                 'unordered': 1,
                 'user': 3,
                 'well': 1,
                 'widths': 1})
    ```
    :::

    :::spoiler Out 3: `pprint.pprint(processed_corpus)`
    ```python
    [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'],
     ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]
    ```
    :::

  - 接著，我們希望將 corpus 中的每個 word 能與唯一的整數 ID 相關聯。 我們可以使用 `gensim.corpora.Dictionary` 類來做到這一點。 這本詞典定義了所有處理後得知的 word 的詞彙。
 
    ```python
    from gensim import corpora

    dictionary = corpora.Dictionary(processed_corpus)
    print(dictionary)
    ```
    
    :::spoiler Out:
    ```python
    Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)
    ```
    :::
    
    由於我們的 corpus 較小，在這個 `gensim.corpora.Dictionary` 中僅僅只有 12 個相異的 tokens。 但在更大的 corpus 中，很常見到字典中包含數十萬以上的 tokens。

- **Vector**: 具有數學意義的 document。
    - 為了能推論 corpus 中的潛在結構，我們需要一種方法來表示 document，使其具有數學上的意義。 而帶有 features 的 vector 正好可以滿足這個需求。而一個 single feature 可以被認為是一個 question-answer 對:
    
        1. Document 中單詞`splonge`出現幾次? `0次`
        2. Document 中由多少段落組成? `2個`
        3. Document 使用了多少種字體？ `5種`
        
    - 這些 questions 通常以整數的方式標示(如: `1`, `2`, `3`)，並伴隨著 answer 可呈現一連串的成對`(1, 0.0), (2, 2.0), (3, 5.0)`。這被稱為 *dense vector*，因為它包含對上述每個問題的明確答案。

    - 此外我們也可以將事前已知的所有 questions 隱藏起來，簡單地將 document 表示成 `(0, 2, 5)`，而這串 answers 就是 document 的 vector (在此例中是一個 3 維的 dense vector)。
    
    - 因於實際目的，Gensim 只允許 answer 是（或可以轉換為）*浮點數* (single floating point number)。

    - 而實際上 vector 時常包含著許多的 0 值。 為了節省記憶體，Gensim 會省略所有皆是 `0.0` 的 vector。 上述的範例就因此變成 `(2, 2.0), (3, 5.0)`。 這被稱為 *稀疏向量 (sparse vector)* 或 *詞袋向量 (bag-of-words vector)*。 此稀疏表示中所有缺失特徵值可以明確地解析為零，即 `0.0`。
    
    - 假設所有的 questions 皆相同，我們可以比較兩個不同 documents 的 vector。舉個例子，這有兩個 vectors，`(0.0, 2.0, 5.0)` 和 `(0.1, 1.9, 4.9)`，因為這兩個 vectors 相當的類似，我們可以斷定這兩個所表示的 document 也是相當類似的。 當然，這個問題的**正確性首先取決於我們選擇問題的程度**。

    - 另外一個方法來表示 document 的 vector 是 *bag-of-words model*。 在 bag-of-words model 下，每個 document 都由一個 vector 表示，該 vector 包含著字典中每個 word 的出現頻率次數。 

    - 舉個例子，假設我們有一個字典包含 `['coffee', 'milk', 'sugar', 'spoon']` 等 words，也有一個 document 的組成包含字串 `"coffee milk coffee"`，那就可以將 vector 表示成 `[2, 1, 0, 0]`。 在這個 vector 中依序可以解讀成 "coffee"、"milk"、"sugar"、"spoon" 出現在 document 的次數。因此該 vector 的長度決定於字典內的詞彙量。

    - 而 bag-of-words model 的主要特色是它不會保存原始句子中詞的順序，因此不考慮文法以及詞的順序，該表示法有許多成功的應用，像是郵件過濾上。[3]

    - 我們處理的 corpus 中有 12 個唯一的 words，這意味著每個 document 將在 bag-of-words model 下由一個 12 維向量表示。 我們可以使用字典將標記化的 document 轉換為這些 12 維向量。 

      我們可以看到這些ID對應的是什麼：
    
        ``` python
        pprint.pprint(dictionary.token2id)
        ```
        :::spoiler Out:
        ``` python
        {'computer': 0,
         'eps': 8,
         'graph': 10,
         'human': 1,
         'interface': 2,
         'minors': 11,
         'response': 3,
         'survey': 4,
         'system': 5,
         'time': 6,
         'trees': 9,
         'user': 7}
        ```
        :::
    
    - 舉個例子，有一個想要向量化的片語 `“Human computer interaction”` (需注意的一點，這個片語並未出現在我們原始的 corpus 中)。 透過使用 `doc2bow` 方法，我們可以創造一個以 bag-of-word 的表示法來表示 document。 此方法是由 dictionary 類提供。

        ```python
        new_doc = "Human computer interaction"
        new_vec = dictionary.doc2bow(new_doc.lower().split())
        print(new_vec)
        ```
        ::: spoiler Out: `print(new_vec)`
        ```python
        [(0, 1), (1, 1)]
        ```
        每個 tuple 中的第一個 entry 對應的是字典內標記的 ID， 第二個對應於該標記出現的計數。
        :::
        
    - 有一點需注意的是，`interaction` 並沒有被向量化 (vectorization)，因為他並不包含在原本的 corpus 中。 同樣地，vector 也僅會包含在 document 中才有的詞。

    - 因為任何給定的 document 只包含字典中眾多單詞中的幾個單詞，所以沒有出現在向量化中的單詞被表示為隱式零 (mplicitly zero) 作為節省空間的措施。

    - 接著，將之前的舉例的 corpus 轉換成一串的 vector:
    
        ```python
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        pprint.pprint(bow_corpus)
        ```
    
        :::spoiler Out: `pprint.pprint(bow_corpus)`
        ```python
        [[(0, 1), (1, 1), (2, 1)],
         [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
         [(2, 1), (5, 1), (7, 1), (8, 1)],
         [(1, 1), (5, 2), (8, 1)],
         [(3, 1), (6, 1), (7, 1)],
         [(9, 1)],
         [(9, 1), (10, 1)],
         [(9, 1), (10, 1), (11, 1)],
         [(4, 1), (10, 1), (11, 1)]]
         ```
        :::
    
    - [補充] 有時我們為了使說明上更加的簡潔，會互換使用 vector 和 document。例如: 我們給予任意 doucment, `D`，而不是說 "這是對應某 document, `D`的 vector"。

    - [補充] 根據 document 呈現方式，兩個不同的 documents 可能會被表示成相同的 vector。

- **Model**: 一種將 vector 從一種表示轉換為另一種表示的演算法。
    - 現在我們有了已向量化的 corpus，可以開始著手透過 model 來轉換它了。 model 作為一個抽象術語，指的是從一種 document 表示到另一種 document 表示的轉換。

    - 在 gensim 中，document 被表示為 vector，因此可以將 model 視為兩個 vector space 之間的轉換。 當 model 讀取 training corpus 時，它會在訓練期間學習這種轉換的細節。

    - 有一個簡單的範例 model 是 [tf-idf](https://zh.wikipedia.org/wiki/Tf-idf)。 tf-idf model 將 vector 從 bag-of-words 表示轉換為一個 vector space，其中頻率計數 (frequency count) 根據 corpus 中每個詞的相對稀有度進行加權。 其可用以評估一字詞對於一個 document 或一個 corpus 中的其中一份檔案的重要程度。

    - 以下是一個簡單的範例，我們嘗試初始化 tf-idf 模型並用先前的 corpus 來訓練它。再將字串 `“system minors”` 輸入 model 轉換。
    
        ```python
        from gensim import models

        # train the model
        tfidf = models.TfidfModel(bow_corpus)

        # transform the "system minors" string
        words = "system minors".lower().split()
        print(tfidf[dictionary.doc2bow(words)])
        ```
        
        :::spoiler Out: `print(tfidf[dictionary.doc2bow(words)])`
        ```python
        [(5, 0.5898341626740045), (11, 0.8075244024440723)]
        ```
        :::
        
    - `tfidf` model 再次返回一個 tuple 的 list，其中 tuple 的第一個 entry 指的是 token ID 且第二個指的是 tf-idf 權重。
        > Note: `“system”` 的 ID (此 word 在原本的 corpus 中出現 4 次) 所伴隨的權重**低於** `“minors”` 的 ID 的權重 (其只出現 2 次)。

    - 此外，可以把 trained models 保存至硬碟中，並在稍後將其載回來繼續訓練新的 training document 或是轉換新的 document。

    - Gemsim 提供許多不同的 models 或是 transformations， 更多可以參考這裡，[Topics and Transformations](https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py)。

    - 一旦你創建了 model，你就可以用它做各種很酷的事情。 例如，通過 TfIdf 轉換整個 corpus 並對其進行索引，為相似性查詢 (similarity queries) 做準備：
    
        ```python
        from gensim import similarities

        index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
        ```
        
      並透過 `query_document` 來為欲查詢的 document 與 原本 corpus 內的每筆 documents 進行相似性比對:
      
        ```python
        query_document = 'system engineering'.split()
        query_bow = dictionary.doc2bow(query_document)
        sims = index[tfidf[query_bow]]
        print(list(enumerate(sims)))
        ```
        
        :::spoiler Out: `print(list(enumerate(sims)))`
        ```python
        [(0, 0.0), (1, 0.32448703), (2, 0.41707572), (3, 0.7184812), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]
        ```
        :::

    - 如何去解讀上述的輸出呢? 與 Document 3 相似的程度有 71.8%，與 Document 2 相似程度為 42% 等等。 此外我們可以通過排序其其更具有可讀性:
 
        ```python
        for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
            print(document_number, score)
        ```
        
        :::spoiler Out: `print(document_number, score)`
        ```python
        3 0.7184812
        2 0.41707572
        1 0.32448703
        0 0.0
        4 0.0
        5 0.0
        6 0.0
        7 0.0
        8 0.0
        ```
        :::

- 總結:
  在這個核心課程中，一開始我們了解了 doucment 與 corpus，接下來嘗試將 documents 轉換成 vector space 的表示方式。轉換後我們建立一個 model 來將原始的 vector 轉換成 Tf-idf 的形式。最後，透過建立好的 model 來計算 query document 與所有在 corpus 內的 documents 的相似度。

- 參考資料: [2]


## Corpora 與 Vector Spaces
此段教程演示如何將 text 轉換成 vector space 表示法。

```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

### 從字串(string)到向量(vector)

- 這一次，讓我們從表示為 string 的 document 開始：

    ```python
    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]
    ```

  這是一個包含 9 個 document 的小 corpus，每個 document 僅包含一個句子。

- 首先，我們標記(tokenize) documents，移除常見字(透過自定義一個 stoplist)，並且刪除在 corpus 中只出現一次的文字(word)。

    ```python=
    from pprint import pprint     # 讓輸出更有結構的 print
    from collections import defaultdict    # 可以強制定義key:value 的 value's data type

    # 移除常用字並且標記文字
    stoplist = set('for a of the and to in'.split()) # 自定義的停用字，文字用空格隔開
    texts = [
        [word for word in document.lower().split() if word not in stoplist] for document in documents
    ]

    # 移除僅出現一次的文字
    # 計算每個 word 出現的次數
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    pprint(texts)
    ```

    :::spoiler Out:
    ```python
    [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'],
     ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]
    ```
    :::

- 每個人處理 documents 的方式可能會有所不同；在這裡是只拆分空格以進行標記化，並將每個單詞轉化為小寫。

:::info
**Note:**
- 此外處理 document 的方式，可能會依賴於應用或是語言，因此不建議用任何的形式來限制它們。相反地應該以 document 中所提取的特徵來表示，而不是用其“表面”字符串形式表示。
- 如何獲取 document 的特稱取決於各人，在此以一種常見的方法，bag-of-words，來表示 document。但務必記住，當有不同的應用領域時是需要有不同的特徵，一旦一如既往，那就只是 garbage in, garbage out…
:::

- 為了將 document 轉換成 vector，我們將使用一種 [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) 表示法。 這種表示法會使各個 doucment 以 vector 的方式呈現，其中在 vector 內的元素類似一個 question-answer 對，像是:

    - Question: How many times does the word system appear in the document?
    - Answer: Once.
  
  這樣的好處在於能將問題關聯到一個獨立的整數(ID)。問題與 id 之間的 mapping 稱為字典(dictionary)。

    ```python
    from gensim import corpora
    dictionary = corpora.Dictionary(texts)
    dictionary.save('tmp//deerwester.dict')  # store the dictionary, for future reference
    print(dictionary)
    ```
    
    :::spoiler Out:
    ```python
    2021-07-26 21:25:49,871 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2021-07-26 21:25:49,872 : INFO : built Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...) from 9 documents (total 29 corpus positions)
    2021-07-26 21:25:49,872 : INFO : Dictionary lifecycle event {'msg': "built Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...) from 9 documents (total 29 corpus positions)", 'datetime': '2021-07-26T21:25:49.872831', 'gensim': '4.0.1', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
    2021-07-26 21:25:49,872 : INFO : Dictionary lifecycle event {'fname_or_handle': 'tmp/deerwester.dict', 'separately': 'None', 'sep_limit': 10485760, 'ignore': frozenset(), 'datetime': '2021-07-26T21:25:49.872831', 'gensim': '4.0.1', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'saving'}
    2021-07-26 21:25:49,873 : INFO : saved tmp/deerwester.dict
    ```
    :::
    
  上述程式透過 `gensim.corpora.dictionary.Dictionary` class 給與所有出現在 corpus 中的詞一個獨一無二的整數 id。 這個 `Dictionary` class 還會掃過整個文本，收集字數和統計數據。 最後的結果可看見有 12 個相異的詞在處理後的 corpus 中，這也意味著各個 doucment 將會由 12 個數字 (即 12 維向量)表示。
  
- 透過以下 code 可以得知 dictionary 內文字與 id 的對應關係:
  
    ```python
    print(dictionary.token2id)
    ```
    
    :::spoiler Out:
    ```python
    {'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4, 'system': 5, 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}
    ```
    :::
    
- 實際將標記好的 doucments 轉換成 vectors:

    ```python
    new_doc = "Human computer interaction"
    new_vec = dictionary.doc2bow(new_doc.lower().split())
    print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored
    ```

    :::spoiler Out:
    ```python
    [(0, 1), (1, 1)]
    ```
    :::
    
  函數 `doc2bow()` 簡單地計算每個不同的 word 出現的數量，將詞對應到字典內的整數 id，並將結果以 sparse vector `[(0, 1), (1, 1)]` 呈現。因此在解讀時可知，`computer (id 0)` 與 `human (id 1)` 各在 document 出現一次，而其他 10 個在字典內的 word 出現 0 次 (隱含，因為沒有實際寫出來)。
  
    ```python
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
    print(corpus)
    ```
  
    :::spoiler Out:
    ```python
    2021-07-26 22:09:12,433 : INFO : storing corpus in Matrix Market format to tmp/deerwester.mm
    2021-07-26 22:09:12,433 : INFO : saving sparse matrix to tmp/deerwester.mm
    2021-07-26 22:09:12,433 : INFO : PROGRESS: saving document #0
    2021-07-26 22:09:12,434 : INFO : saved 9x12 matrix, density=25.926% (28/108)
    2021-07-26 22:09:12,434 : INFO : saving MmCorpus index to tmp/deerwester.mm.index
    [[(0, 1), (1, 1), (2, 1)], [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(2, 1), (5, 1), (7, 1), (8, 1)], [(1, 1), (5, 2), (8, 1)], [(3, 1), (6, 1), (7, 1)], [(9, 1)], [(9, 1), (10, 1)], [(9, 1), (10, 1), (11, 1)], [(4, 1), (10, 1), (11, 1)]]
    ```
    :::
    
    現在可以很清楚了得知 id=10 的向量特征代表的問題是 ``“詞 "graph" 在document 中出現了多少次？”`` 並且前六個 document 所給的答案是 “零”，其餘三個 document 的答案是 “一”。

### Corpus Streaming – One Document at a Time
- 上面所使用的 corpus 都是做為一個 Python list 並保存在 memory 中。

- 此章節雖然沒有其他章節的重要，但還是有其需要。因為有時我們的 corpus 中可能有數百萬個 documents 在其中。 而一次將其保存在 menory 中顯然是不可行的。 相反地可以將 corpus 放在 disk 上的一個文件中，而該文件的每行都是一個 document。

- Gensim 僅要求 corpus 一次要能返回一個 document 的 vector。 所以我們可以定義一個 class 來達到我們的需求，程式碼如以下

    ```python=
    file_name = 'mycorpus.txt'
    class MyCorpus:
        def __init__(self, file_name):
            self.file_name = file_name

        def __iter__(self):
            for line in open(file_name):
                # assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(line.lower().split())
    ```

- Gensim 並不限定 corpus 一定得是 Python list, Numpy array, Pandas dataframe，它接受一切能在 iterate 並依序生成 doucment 的對象(object)。 

        # This flexibility allows you to create your own corpus classes that stream the
        # documents directly from disk, network, database, dataframes... The models
        # in Gensim are implemented such that they don't require all vectors to reside
        # in RAM at once. You can even create the documents on the fly!

- 假設今天有一個 `mycorpus.txt` 的檔案，它內部放置著我們一直舉例時使用的那 9 個 documents，而每一個 document 都放在單一行，

    :::spoiler mycorpus 的內容

        Human machine interface for lab abc computer applications
        A survey of user opinion of computer system response time
        The EPS user interface management system
        System and human system engineering testing of EPS
        Relation of user perceived response time to error measurement
        The generation of random binary unordered trees
        The intersection graph of paths in trees
        Graph minors IV Widths of trees and well quasi ordering
        Graph minors A survey

    :::

- 你就可以自行創立一個 class 改寫其 ` __iter__` 方法來滿足你的輸入格式。 此外輸入的獲取也有可能需要透過遍歷目錄(Walking directorie)、解析 XML(parsing XML)、訪問網絡等方法。 因此將這些資料經過處理後，最終你只需要給予一個乾淨的 token list(詞與詞之間分開的 list 且每個詞不能是停用詞也不能只出現過一次) 就可以透過 dictionary 將其轉變成對應 id (token -> id)，並在 `__iter__` 中返回生成的稀疏向量。

    ```python
    corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
    print(corpus_memory_friendly)
    ```

    :::spoiler Out:
    ```python
    <__main__.MyCorpus object at 0x11e77bb38>
    ```
    :::

- 現在 corpus 只是一個 object。 由於我們沒有定義任何 print 方式，所以只是輸出 object 在記憶體中的地址。 這並不是很有用。 因此要查看組成向量，讓我們遍歷 corpus 並 print 每個 document 的 vector（一次一個）： 

    ```python
    for vector in corpus_memory_friendly:  # load one vector into memory at a time
        print(vector)
    ```

    :::spoiler Out:
    ```python
    [(0, 1), (1, 1), (2, 1)]
    [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
    [(2, 1), (5, 1), (7, 1), (8, 1)]
    [(1, 1), (5, 2), (8, 1)]
    [(3, 1), (6, 1), (7, 1)]
    [(9, 1)]
    [(9, 1), (10, 1)]
    [(9, 1), (10, 1), (11, 1)]
    [(4, 1), (10, 1), (11, 1)]
    ```
    :::

- 儘管只是輸出普通的 Python list，但 corpus 現在對記憶體變得更加友好，因為一次最多只有一個向量駐留在 RAM 中。 現在您的 corpus 可以大到您想要的大小了。 

- 同樣地，dictionary 的建立也可以不用一次載入所有文字到 RAM 中。以下我們將修改上述的程式碼來滿足需求。在此也將 stopword 放置到 `stoplist.txt` 中

    ```python
    # collect statistics about all tokens
    dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
    stoplist = [word for stopwords in open('stoplist.txt') for word in stopwords.split()]

    # remove stop words and words that appear only once
    stop_ids = [
        dictionary.token2id[stopword]
        for stopword in stoplist
        if stopword in dictionary.token2id
    ]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    print(dictionary)
    ```
    
    :::spoiler Out:
    ```python
    2021-08-02 11:21:22,997 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2021-08-02 11:21:22,998 : INFO : built Dictionary(42 unique tokens: ['abc', 'applications', 'computer', 'for', 'human']...) from 9 documents (total 69 corpus positions)
    2021-08-02 11:21:22,998 : INFO : Dictionary lifecycle event {'msg': "built Dictionary(42 unique tokens: ['abc', 'applications', 'computer', 'for', 'human']...) from 9 documents (total 69 corpus positions)", 'datetime': '2021-08-02T11:21:22.998851', 'gensim': '4.0.1', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
    Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)
    ```
    :::
    
    :::spoiler stoplist 的內容
    ```
    for a of the and to in
    ```
    :::
    
- 以上是這個章節的內容。雖然我們完全不清楚計算不同單詞的頻率如何有用。 但就目前為止我們已對詞袋(word-of-bag)表示有一定的認知了。然而這些都是必然的流程，我們需要先對這個簡單的表示做應用轉換，然後才能使用它來計算任何有意義的 document 與 document 的相似性。 轉換將在下一個教程（主題和轉換）中介紹，但在此之前，讓我們簡要地將注意力持續放在 corpus 上。 

### Corpus Formats

- 其實對一個連續 Vector Space corpus(~sequence of vectors) 在磁碟上的 file 是存在有各種的格式(format)。 

- Gensim 透過串流的 corpus 介面來讓實作時更加的簡易: document 從磁碟讀取（或存儲），一次一個 document，而不是將整個 corpus 一次讀入 RAM。 因此當我們需要保存或讀取 corpus 時就必須決定使用哪個格式。

- 有一種值得我們注意的檔案格式是 (Matrix Market format)[https://math.nist.gov/MatrixMarket/formats.html] (MM format)，這是一種保存矩陣數據的格式，基於 AscII 的可讀性很強的文件格式，目的是促進數據的交流。其定義的 sparse matrix 和 density matrix 的保存方法。 (保存方法很簡單也很暴力)

- 以下程式碼實作如何讓 corpus 保存成 MM 格式
    ```python=
    # corpus represents as bag-of-word vector
    corpus = [[(0, 2),(1, 3)], # document 1, only 2 tokens, token 0 has 2, token 1 has 3
             [(1, 1)],         # document 2, only 1 tokens, token 1 has 1
             []]               # document 3 has 0 token
    corpora.MmCorpus.serialize('tmp/corpus.mm', corpus)

    print('\nprint file \'tmp/corpus.mm\'')
    for i in open('tmp/corpus.mm'):
        print(i, end='')
    ```

    :::spoiler Out:
    ```python
    2021-08-02 12:10:20,866 : INFO : storing corpus in Matrix Market format to tmp/corpus.mm
    2021-08-02 12:10:20,867 : INFO : saving sparse matrix to tmp/corpus.mm
    2021-08-02 12:10:20,867 : INFO : PROGRESS: saving document #0
    2021-08-02 12:10:20,868 : INFO : saved 3x2 matrix, density=50.000% (3/6)
    2021-08-02 12:10:20,868 : INFO : saving MmCorpus index to tmp/corpus.mm.index

    print file 'tmp/corpus.mm'
    %%MatrixMarket matrix coordinate real general
    3 2 3
    1 1 2
    1 2 3
    2 2 1
    ```
    :::

- 從輸出訊息可知我們保存了一個 3x2 的 matrix，這裡的 3 表示的是 document 數，而 2 則表示 token 數。 由於 matrix 僅有一半的內容有值，故密度是 50%。

- 接著我們檢查輸出的 `tmp/corpus.mm`。第一行 `3 2 3`，此列是預先宣告，前 2 個元素表示著 3x2 的矩陣，後一個則表示該矩陣有 3 個非零元素。接著的每一列，第 1 個元素表示著第幾個 document，第 2 元素表示 token id，第三個元素是 token 出現的次數。 這裡要注意的是 document 3 是 0 token，所以不會有出現任何一列。

- 此外也有其它 format。 如 [Joachim’s SVMlight](http://svmlight.joachims.org/), [Blei’s LDA-C](https://www.cs.princeton.edu/~blei/lda-c/) 和 [GibbsLDA++](http://gibbslda.sourceforge.net/)。 操作方法如下，

    ```python
    corpora.SvmLightCorpus.serialize('tmp/corpus.svmlight', corpus)
    corpora.BleiCorpus.serialize('tmp/corpus.lda-c', corpus)
    corpora.LowCorpus.serialize('tmp/corpus.low', corpus)
    ```

    :::spoiler Out
    ```python
    2021-08-02 12:31:26,502 : INFO : converting corpus to SVMlight format: tmp/corpus.svmlight
    2021-08-02 12:31:26,503 : INFO : saving SvmLightCorpus index to tmp/corpus.svmlight.index
    2021-08-02 12:31:26,504 : INFO : no word id mapping provided; initializing from corpus
    2021-08-02 12:31:26,505 : INFO : storing corpus in Blei's LDA-C format into tmp/corpus.lda-c
    2021-08-02 12:31:26,506 : INFO : saving vocabulary of 2 words to tmp/corpus.lda-c.vocab
    2021-08-02 12:31:26,508 : INFO : saving BleiCorpus index to tmp/corpus.lda-c.index
    2021-08-02 12:31:26,509 : INFO : no word id mapping provided; initializing from corpus
    2021-08-02 12:31:26,509 : INFO : storing corpus in List-Of-Words format into tmp/corpus.low
    2021-08-02 12:31:26,510 : INFO : saving LowCorpus index to tmp/corpus.low.index
    ```
    :::

- 相反地，要從 Matrix Market 檔案中 load corpus iterator，

    ```python
    new_corpus = corpora.MmCorpus('tmp/corpus.mm')
    ```
    :::spoiler Out:
    ```python
    2020-10-28 00:52:04,538 : INFO : loaded corpus index from /tmp/corpus.mm.index
    2020-10-28 00:52:04,540 : INFO : initializing cython corpus reader from /tmp/corpus.mm
    2020-10-28 00:52:04,540 : INFO : accepted corpus with 2 documents, 2 features, 1 non-zero entries
    ```
    :::

- `new_corpus` 物件目前是串流的形式，所以你無法很一般地將其直接印出。 

    ```python
    print(new_corpus)
    ```
    :::spoiler Out:
    ```python
    MmCorpus(3 documents, 2 features, 3 non-zero entries)
    ```
    :::

- 因此需將其先載入 RAM 才能看到內容物

    ```python
    # one way of printing a corpus: load it entirely into memory
    print(list(new_corpus))  # calling list() will convert any sequence to a plain Python list
    ```
    :::spoiler Out:
    ```python
    [[(0, 2.0), (1, 3.0)], [(1, 1.0)], []]
    ```
    :::

    或是

    ```python
    # another way of doing it: print one document at a time, making use of the streaming interface
    for doc in new_corpus:
        print(doc)
    ```
    :::spoiler Out:
    ```python
    [(0, 2.0), (1, 3.0)]
    [(1, 1.0)]
    []
    ```
    :::
    
    第二種方式顯然對 RAM 更加友好，但出於測試和開發目的，沒有什麼比呼叫 `list(new_corpus)` 更簡單了。 
    
- 以上的這些方法，可以使 gensim 作為減省內存和 I/O 格式轉換的工具(ex: MM -> SVMlight)。

### Compatibility with NumPy and SciPy
- Gensim 也內建一些工具函數可以輕鬆的轉換成或讀取至 Numpy matrices。

    ```python
    import gensim
    import numpy as np

    number_of_corpus_features = 5
    number_of_corpus_document = 2

    numpy_matrix = np.random.randint(10, size=[number_of_corpus_features, number_of_corpus_document])  # random matrix as an example
    print('numpy_matrix:\n', numpy_matrix)

    print('\nDense2Corpus:')
    corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
    for i in corpus:
        print(i)


    numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms= number_of_corpus_features)
    print('\nnumpy_matrix:\n',numpy_matrix)
    ```
    :::spoiler Out:
    ```python
    numpy_matrix:
     [[1 0]
     [6 9]
     [4 3]
     [5 5]
     [4 2]]

    Dense2Corpus:
    [(0, 1.0), (1, 6.0), (2, 4.0), (3, 5.0), (4, 4.0)]
    [(1, 9.0), (2, 3.0), (3, 5.0), (4, 2.0)]

    numpy_matrix:
     [[1. 0.]
     [6. 9.]
     [4. 3.]
     [5. 5.]
     [4. 2.]]
    ```
    :::

- 也可以是 scipy.sparse matrices

    ```python
    import scipy.sparse
    scipy_sparse_matrix = scipy.sparse.random(5, 2)  # random sparse matrix as example
    corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    ```
## Topics 與 Transformations

介紹轉換(Transformation) 並演示它們在 corpus 中的使用。

在本章節開始前先宣告以下內容在我們 code 的正上方，方便我們追蹤 log 紀錄。
```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

在這個章節中，將會示範如何將一個 document 從一個 vector 表示法到另一個表示法。

這個過程有兩個目的：
1. 揭示 corpus 中的隱藏結構，發現 word 之間的關係並使用這關係來讓它們以一種新的更具語義化（希望如此）的方式來描述文檔。
2. 為了使 document 的表示方式更加的結實。這既提高了效率(efficiency)（新表示消耗的資源更少）又提高了效率(efficacy)（忽略了邊際數據趨勢，降噪）。

### Creating the Corpus
- 首先，我們需要創建一個 corpus 來使用。 這一步和上一個章節一樣； 如果您已完成，請跳到下一部分。

    ```python
    from collections import defaultdict
    from gensim import corpora

    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ```
    
### Creating a transformation
- Transformation 是標準的 Python object，通常通過訓練 corpus 時初始化：

    ```python
    from gensim import models
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    ```
  
  我們利用之前常用的那個 corpus 來訓練 transformation model。 對於所選擇的 model 不同有不同的初始參數需要給予。 在這個例子中，我們使用 Tfidf model 來訓練，它的訓練很簡單，只要給予它 corpus，它就能計算出 document 內所有 features 的 frequencies。
  
- 此外還有其他的 training model 可以使用，如 Latent Semantic Analysis 和 Latent Dirichlet Allocation，但它們涉及更多資訊，因此需要更多的時間來訓練。
  
    :::info
    :warning: **Note**
    轉換(Transformation) 永遠是從兩個特定的 vector spaces 中做轉變。 必須使用相同的 vector space （指擁有相同 feature ids 集）進行訓練以及後續的向量轉換。 若不使用相同的 feature space 輸入，例如在應用相異字串的預處理時，使用不同的 feature id，或是使用 bag-of-words 向量輸入而不是被預期使用的 Tfidf 向量，將導致轉換調用期間的特徵不匹配，從而導致垃圾輸出和（或）運行時異常。
    :::

### Transforming vectors
- 從現在開始，tfidf 被視為 read-only 物件，可用於將任何向量從舊表示（整數計數的bag-of-words）轉換為新表示（TfIdf 的實數權重）：

    ```python
    doc_bow = [(0, 1), (1, 1)]
    print(tfidf[doc_bow])  # step 2 -- use the model to transform vectors
    ```

    :::spoiler Out:
    ```python
    [(0, 0.7071067811865476), (1, 0.7071067811865476)]
    ```
    - 可以思考看看為何這兩值相同，善用 print 去研究看看。
    :::

- 或是一次將整個 curpos 做轉換

    ```python
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)
    ```
    :::spoiler Out:
    ```python
    [(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)]
    [(0, 0.44424552527467476), (3, 0.44424552527467476), (4, 0.44424552527467476), (5, 0.3244870206138555), (6, 0.44424552527467476), (7, 0.3244870206138555)]
    [(2, 0.5710059809418182), (5, 0.4170757362022777), (7, 0.4170757362022777), (8, 0.5710059809418182)]
    [(1, 0.49182558987264147), (5, 0.7184811607083769), (8, 0.49182558987264147)]
    [(3, 0.6282580468670046), (6, 0.6282580468670046), (7, 0.45889394536615247)]
    [(9, 1.0)]
    [(9, 0.7071067811865475), (10, 0.7071067811865475)]
    [(9, 0.5080429008916749), (10, 0.5080429008916749), (11, 0.695546419520037)]
    [(4, 0.6282580468670046), (10, 0.45889394536615247), (11, 0.6282580468670046)]
    ```
    :::

- 我們現在正在用訓練時的 corpus 來轉換，但這只是偶然的應用。 其實一旦 transformation model 被初始化，它就可以用於任何向量（當然，前提是它們來自相同的 vector space），即使它們根本沒有在 training corpus 中使用。<!--This is achieved by a process called folding-in for LSA, by topic inference for LDA etc. -->

    :::info
    :warning: **Note**
    調用 `model[corpus]` 只會在舊的 corpus document 流周圍創建一個包裝器(wrapper)——實際轉換是在 document 迭代期間即時完成的。 因此我們無法在調用時轉換整個 corpus。
    `corpus_transformed = model[corpus]` 此句語法意味著將結果存儲在 RAM 中，這與 gensim 的內存獨立目標相互矛盾。 如果將 corpus_transformed 迭代轉換多次，則轉換成本很高，因此請先將生成的 corpus 序列化到 disk 並繼續使用它。
    :::

- 轉換也可以序列化，一個接在另一個之上，形成一種鏈狀：
    ```python
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
    corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    ```

- 在這裡，我們通過潛在語義索引(Latent Semantic Indexing) 將我們的 Tf-Idf corpus 轉換為潛在的 2 維空間（為何 2 維，因為我們設置 `num_topics=2`）。 現在你可能想知道：這兩個潛在維度代表什麼？ 讓我們用 `models.LsiModel.print_topics()` 來檢查： 

    ```python
    print(lsi_model.print_topics(2))
    ```
    
    :::spoiler out
    ```python
    [(0, '0.703*"trees" + 0.538*"graph" + 0.402*"minors" + 0.187*"survey" + 0.061*"system" + 0.060*"response" + 0.060*"time" + 0.058*"user" + 0.049*"computer" + 0.035*"interface"'), (1, '-0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"response" + -0.320*"time" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"')]
    ```
    :::

- 從輸出中可得知，LSI 認為 `tree` 、 `graph` 、 `minors` 是相關聯的字詞且在第 一個主題的方向中具有較高的貢獻。 而第二個主題則實際上與所有其他的字詞都有關。

- 正如預期的那樣，前五個 doucment 與第二個主題的相關性更強，而其餘四個 doucment 與第一個主題的相關性更強。

    ```python
    # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    for doc, as_text in zip(corpus_lsi, documents):
        print(doc, as_text)
    ```
    :::spoiler Out:
    ```python
    [(0, 0.06600783396090543), (1, -0.5200703306361846)] Human machine interface for lab abc computer applications
    [(0, 0.19667592859142624), (1, -0.7609563167700041)] A survey of user opinion of computer system response time
    [(0, 0.08992639972446687), (1, -0.7241860626752505)] The EPS user interface management system
    [(0, 0.07585847652178407), (1, -0.6320551586003423)] System and human system engineering testing of EPS
    [(0, 0.10150299184980191), (1, -0.5737308483002952)] Relation of user perceived response time 
    to error measurement
    [(0, 0.7032108939378309), (1, 0.1611518021402597)] The generation of random binary unordered trees
    [(0, 0.8774787673119828), (1, 0.16758906864659642)] The intersection graph of paths in trees  
    [(0, 0.9098624686818575), (1, 0.14086553628719256)] Graph minors IV Widths of trees and well quasi ordering
    [(0, 0.6165825350569281), (1, -0.05392907566389192)] Graph minors A survey
    ```
    :::
    
- 此外也可以透過 `save()` 和 `load()` 來達到 Model 的持久性。（以下示範如何將 model save 到暫存中，又從暫存中讀取）
    ```python
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
        lsi_model.save(tmp.name)  # same for tfidf, lda, ...

    loaded_lsi_model = models.LsiModel.load(tmp.name)

    os.unlink(tmp.name)    # delete the temp model
    ```

### Available transformations
Gensim 內有實作數個知名的向量空間轉換演算法。

- **Term Frequency Inverse Document Frequency [(Tf-Idf)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)**，它預期我們使用 bag-of-word 形式的 train corpus 來做初始化。 而在轉換向量空間的期間，它將保持著向量前後維度的相同。 此外它將會對在 train corpus 中很少見的 features 額外增加其價值。 因此它將整數值向量轉換為實值向量，同時保持維數不變。它還可以選擇將結果的向量 normalize 為（歐幾里德）單位長度（結果介於 0~1 之間）。

    ```python
    model = models.TfidfModel(corpus, normalize=True)
    ```

- **Latent Semantic Indexing [(LSI or sometimes LSA)](https://en.wikipedia.org/wiki/Latent_semantic_indexing)**，它預期我們使用 bag-of-word 或 TfIdf-weighted 形式的 train corpus 來初始化，它將會對輸入向量空間進行降維至一個低維度的潛在空間。 對於先前示範的 corpus，我們只使用了 2 個潛在維度，但在真實 corpus 中，建議將 200-500 的目標維度作為 “黃金標準(golden standard)”。

    ```python
    model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
    ```
    輸入若能附上字典，則可在 print Topic 時有相對應的字詞。
    
    LSI 的獨特之處在於可以在任何時候繼續 “訓練”，只需提供更多 train document 即可。 這是通過在稱為在線訓練（online training）的過程中對底層模型進行增量更新來完成的。由於這個特性，輸入 doucment stream 甚至可能是無限的——只要在 LSI 新的 doucment 到達時繼續輸入它們，同時將計算的轉換模型用作只讀！
    
    詳細的 LSI 介紹可以閱讀 [`gensim.models.lsimode`](https://radimrehurek.com/gensim/models/lsimodel.html#module-gensim.models.lsimodel) 的官方文件。 內有提及如何讓 LSI 逐漸地在無止境的串流中 “忘記” 舊有的觀察。 或是如果你想資料變髒，你也可以調整一些參數，這些參數會影響 LSI 算法的速度、內存佔用和數值精度。
    
    ```python
    model.add_documents(another_tfidf_corpus)  # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
    lsi_vec = model[tfidf_vec]  # convert some new document into the LSI space, without affecting the model

    model.add_documents(more_documents)  # tfidf_corpus + another_tfidf_corpus + more_documents
    lsi_vec = model[tfidf_vec]
    ```

- **Random Projections ([RP](http://www.cis.hut.fi/ella/publications/randproj_kdd.pdf))**，可協助於降低向量空間維數，並且對 menmory 和 cpu 更加的有善與更具效率。 它的方法是通過加入一點隨機性來近似文檔之間的 TfIdf 距離。 推薦使用在數百或數千目標維度的資料集上。

    ```python
    model = models.RpModel(tfidf_corpus, num_topics=500)
    ```

- **Latent Dirichlet Allocation ([LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation))**，是另一種從 bag-of-words 計數到低維主題空間的轉換。 LDA 是 LSA（也稱為多項式 PCA）的概率擴展，因此 LDA 的主題可以解釋為單詞的概率分佈。這些分佈就像 LSA 一樣，是從 training corpus 中自動推斷出來的。文檔反過來被解釋為這些主題的（軟）混合（同樣，就像 LSA 一樣）。

    ```python
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
    ```

- **Hierarchical Dirichlet Process ([HDP](http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf))**，是一種非參數的貝葉斯方法（注意: 此處省略了請求主題(num_topics)的數量）

    ```python
    model = models.HdpModel(corpus, id2word=dictionary)
    ```
    HDP 模型是 gensim 的新增功能，但其在學術上仍然粗糙——請謹慎使用。

值得重申的是，這些 model 都是獨特的**增量**實現，不需要整個 training corpus 一次全部出現在 main memory 中。

## Similarity Queries
本節將會演示如何為 doucment 在 corpus 上做相似度查詢(Similarity Queries)。

### Creating the Corpus
首先，我們需要創建一個 corpus 來使用。 這一步和上一個教程一樣；如果您已完成，請跳到下一部分。

```python
from collections import defaultdict
from gensim import corpora

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
```

### Similarity interface
在先前的章節中有提過 "corpus 和 vector space"、"主題與轉換"，我們介紹了在向量空間模型中創建 corpus 的含義以及如何在不同的向量空間之間進行轉換。 在這之後我們常有需求在於**要確定成對 document 之間的相似性**，或者**特定 document 與一組其他 documents（例如用戶查詢與索引 document）之間的相似性**。

為了展示如何在 gensim 中實現這一點，我們考慮使用與先前示範相同的 corpus。 接著為這個小 corpus 定義一個 2 維的 LSI 空間
```python
from gensim import models
lsi = model.LsiModel(corpus, id2word=dictionary, num_topics=2)
```

為了達成本節的目的，你對於 LSI 必須有 2 件事需要先得知。 首先，它就只是另一種轉換 —— 從一個向量空間到另一個。 第二，LSI 的優勢在於它能能夠辨識出 patterns 和項目（在這個例子中，指在 doucment 中的字詞）與主題之間的關係。 此次的 LSI space 是二維空間（num_topics = 2），所以會分出兩個主題，但這可以任意調整。

現在假設有個使用者嘗試打入 *“Human computer interaction”* 要來做查詢。 我們希望能按照與此查詢的相關性降序對九個 corpus documents 進行排序。 與現代搜索引擎不同，這裡我們只關注可能相似性的一個方面——它們的文本（單詞）的明顯語義相關性。沒有超鏈接，沒有隨機遊走靜態排名，只是對 boolean 關鍵字匹配的語義擴展：

```python
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)
```

:::spoiler Out:
```python
[(0, -0.4618210045327159), (1, -0.07002766527900028)]
```
:::

此外，我們將考慮餘弦相似度來確定兩個向量的相似度。 餘弦相似度是向量空間建模中的標準評估，但在向量表示概率分佈的任何地方，不同的相似度評估可能更合適。

### Initializing query structures
為了準備相似性查詢，我們需要輸入要與後續查詢進行比較的所有 documents。 在我們的例子中，它們與用於訓練 LSI 的 9 個 document 相同，並轉換為 2 維 LSA 空間。 但這只是偶然，我們也可能正在索引一個完全不同的 corpus。

```python
from gensim import similarities
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
```

</br>

:::warning
:warning: **Warning**
`similarities.MatrixSimilarity` 類別僅適用於當整個向量集處在 RAM 時。 例如，當與此類別一起使用時，包含 100 萬個 document 的 corpus 將需要 256 維 LSI 空間，而這個空間將要 2GB RAM。

如果沒有 2GB 的可用 RAM，您將需要使用 `Similarities.Similarity` 類別。 此類別在固定 memory 中運行，通過將索引拆分到磁盤上的多個文件（稱為分片）。 它在內部使用了 `Similarities.MatrixSimilarity` 和 `Similarities.SparseMatrixSimilarity`，所以它仍然很快，雖然稍微複雜一些。 
:::

</br>

為了讓 index 也具持久性，可以透過 `save()` 或 `load()` 函數來達到。
```python
index.save('tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('tmp/deerwester.index')
```

這些方法適用於所有相似性索引類（`similarities.Similarity`、`similarities.MatrixSimilarity` 和 `similarities.SparseMatrixSimilarity`）。 而在載入索引時，索引可能是其中任何一個的對象。 因此如有疑問，請使用 `similarities.Similarity`，因為它是最具擴展性的版本，並且還支持稍後向索引添加更多文檔。 

### Performing queries

為了獲得我們的 query doucment 與九個 indexed document 的相似性：

```python
sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
```

:::spoiler Out:
```python
[(0, 0.998093), (1, 0.93748635), (2, 0.9984453), (3, 0.9865886), (4, 0.90755945), (5, -0.12416792), (6, -0.10639259), (7, -0.09879464), (8, 0.050041765)
```
:::

其評估方式是使用餘弦角度，而餘弦評估後會返回 <-1, 1> 範圍內的相似性（越大，越相似），因此第一個 doucment 的得分為 0.99809301 表示最為相似。

接著在使用一些標準的 Python 語法來實現魔法，令這些相似性依照降序進行排序，並獲得查詢 “Human computer interaction” 的最終答案:

```python
sims = sorted(enumerate(sims), key= lambda item: -item[1])
print('{:8} {:} {:}'.format('Score', '\t', 'Document'))
print('{:8} {:} {:}'.format('---','\t','---'))
for doc_position, doc_score in sims:
    print('{:.6f} {:} {:}'.format(doc_score, '\t', documents[doc_position]))
```

:::spoiler Out:
```python
Score            Document
---              ---
0.998445         The EPS user interface management system
0.998093         Human machine interface for lab abc computer applications
0.986589         System and human system engineering testing of EPS
0.937486         A survey of user opinion of computer system response time
0.907559         Relation of user perceived response time to error measurement
0.050042         Graph minors A survey
-0.098795        Graph minors IV Widths of trees and well quasi ordering
-0.106393        The intersection graph of paths in trees
-0.124168        The generation of random binary unordered trees
```
:::

這裡有需要觀察的地方，第 2 個 document 的內容是 `"The EPS user interface management system"` 與第 4 個 document 的內容是 `"Relation of user perceived response time to error measurement"`，如果是用標準方法（採用布林比對全文是否有匹配的字詞）則將永遠不會標示相似，因為它們的內文未與我們輸入的 `"Human computer interaction"` 有匹配的地方。 然而我們應用過 LSI 方法，所以我們可以觀察出其實這兩者具有很高的相似性（而第 2 個其實是最相似的！）。 這更符合我們的直覺，或許他們就是想查詢 `“computer-human”` 的相關主題。 事實上，擁有語義一般化就是我們首先應用轉換和進行主題建模的原因。

## Word2Vec Model
稍後將介紹 Gensim 上的 Word2Vec model 並將其示範在 [Lee Evaluation Corpus](https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf) 上。

開始學習前可加入以下程式碼來協助我們了解運行的過程
```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

Word2Vec 是一種廣泛使用的基於神經網絡的演算法，通常被稱為 “深度學習”（儘管 word2vec 本身相當淺）。 word2vec 使用大量未註釋的純文本，自動學習單詞之間的關係。 輸出是向量，每個詞一個向量，具有顯著的線性關係，允許我們做如下事情： 

- vec(“king”) - vec(“man”) + vec(“woman”) =~ vec(“queen”)
- vec(“Montreal Canadiens”) – vec(“Montreal”) + vec(“Toronto”) =~ vec(“Toronto Maple Leafs”). 
  "蒙特婁加拿大人" - "蒙特婁" + "多倫多" ~= "多倫多楓葉"

也因此 Word2vec 在自動文本標記、推薦系統和機器翻譯中非常有用。 

本教程：
1. 引入 Word2Vec 作為對傳統 Bag-of-word 的改進
2. 使用預訓練模型展示 Word2Vec 的演示
3. 演示從您自己的數據訓練新模型
4. 演示加載和保存模型
5. 介紹幾個訓練參數並展示它們的效果
6. 討論 RAM 的要求
7. 通過應用降維來可視化 Word2Vec embedding

### 回顧 Bag-of-words
您可能在向量部分中就熟悉了 Bag-of-word 模型。 該模型將每個 Document 轉換為一個固定長度的整數向量。 

例如，給定以下句子： 
- `John likes to watch movies. Mary likes movies too.`
- `John also likes to watch football games. Mary hates football.`

模型輸出向量：
- `[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]`
- `[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]`

每個向量有 10 個元素，其中每個元素計算特定 word 在 Document 中出現的次數。 **元素的順序是任意的**。 在上面的例子中，元素的順序對應於 word： 

    ["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]

Bag-of-words 出奇地有效，但也有一些弱點。

首先，它們丟失了所有關於字詞之間順序的資訊：“John likes Mary” 和 “Mary likes John” 對應於相同的向量。 有一個解決方案：bag of [n-grams](https://en.wikipedia.org/wiki/N-gram) 模型考慮長度為 n 的詞短語將文檔表示為固定長度的向量來捕獲局部詞序，但存在數據稀疏性和高維數的問題。 

其次，該模型不會嘗試學習潛在單詞的含義，因此，向量之間的距離並不能總是反映含義間的差異，但 Word2Vec 模型解決了第二個問題。 

### 介紹：Word2Vec 模型
Word2Vec 是一個近代的模型，它使用淺層神經網絡將單詞嵌入到低維向量空間中。 結果將是一組詞向量，其中向量空間中靠近的向量根據上下文具有相似的含義，而彼此遠離的詞向量具有不同的含義。 例如 `strong` 和 `powerful` 會靠近，而 `strong` 和 `Paris` 會相對較遠。

此模型有兩個版本，Word2Vec 類同時實現了它們 
1. Skip-grams (SG)
2. Continuous-bag-of-words (CBOW)

[Word2Vec Skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model) 模型將通過在文本數據上移動窗口生成的成對 (word1, word2)，並基於給定輸入單詞的合成任務訓練一個 1-hidden-layer 神經網絡，並輸出給我們鄰近詞對輸入的預測概率分佈。 單詞透過虛擬 one-hot 編碼通過 “投影層” 到達隱藏層； 這些投影權重後來被解釋為 word embeddings。 所以如果隱藏層有 300 個神經元，這個網絡會給我們 300 維的 word embeddings。

Continuous-bag-of-words Word2vec 與 skip-gram 模型非常相似。 它也是一個 1-hidden-layer 神經網絡。 合成訓練任務現在使用多個輸入上下文詞的平均值，而不是像 skip-gram 中的單個詞，來預測中心詞。 同樣，將單 one-hot 詞轉換為與隱藏層寬度相同的可平均向量的投影權重被解釋為 word embeddings。 

### Word2Vec Demo
要了解 Word2Vec 可以做什麼，讓我們下載一個預先訓練的模型並使用它。 我們將獲取在部分 Google 新聞數據集上訓練的 Word2Vec 模型，涵蓋大約 300 萬個單詞和短語。 這樣的模型可能需要幾個小時來訓練，但由於它已經可用，用 Gensim 下載和加載它只需要幾分鐘。 

</br>

:::warning
**Important**
該模型大約為 2GB，因此您需要良好的網絡連接才能繼續。 否則，請跳到下面的“訓練你自己的模型”部分。
:::

</br>

該演示在整個 Google 新聞數據集上運行 word2vec，大約 1000 億字。
```python
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
```

一個常見的操作是檢索模型的詞彙表。 
```python
for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")
```
輸出:
```
word #0/3000000 is </s>
word #1/3000000 is in
word #2/3000000 is for
word #3/3000000 is that
word #4/3000000 is is
word #5/3000000 is on
word #6/3000000 is ##
word #7/3000000 is The
word #8/3000000 is with
word #9/3000000 is said
```

</br>

我們可以輕鬆獲得模型熟悉的術語的向量： 
```python
vec_king = wv['king']
```

</br>

不幸的是模型無法為不熟悉的單詞推斷向量。 這是 Word2Vec 的一個限制：如果此限制對您很重要，請查看 FastText 模型。
```python
try:
    vec_cameroon = wv['cameroon']
except KeyError:
    print("The word 'cameroon' does not appear in this model")
```

輸出:
```
The word 'cameroon' does not appear in this model
```

</br>

繼續前進，Word2Vec 支持多種開箱即用的單詞相似性任務。 您可以看到隨著單詞變得越來越不相似，相似度是如何直觀地降低的。 
```python
pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))
```
輸出:
```
'car'	'minivan'	0.69
'car'	'bicycle'	0.54
'car'	'airplane'	0.42
'car'	'cereal'	0.14
'car'	'communism'	0.06
```

</br>

印出與 “car” 或 “minivan” 最相似的 5 個詞 
```python
print(wv.most_similar(positive=['car', 'minivan'], topn=5))
```
輸出:
```
[('SUV', 0.8532192707061768), ('vehicle', 0.8175783753395081), ('pickup_truck', 0.7763688564300537), ('Jeep', 0.7567334175109863), ('Ford_Explorer', 0.7565720081329346)]
```

</br>

下面哪個不屬於這個序列？ 
```python
print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))
```
輸出:
```
car
```


### 
### h3
- [time=Mon, Aug 9, 2021 8:32 PM] 閱讀至[此](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py)
:::spoiler Out:
```python

```
:::


```python

```
:::spoiler Out:
```python

```
:::

## 作業練習

## 參考資料
- [1] [What is Gensim?](https://radimrehurek.com/gensim/intro.html#what-is-gensim) - `radimrehurek.com`
- [2] [Core Concepts](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts) - `radimrehurek.com`
- [3] [詞袋模型](https://zh.wikipedia.org/wiki/%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B) - `wikipedia`

https://www.coder.work/article/92412

```python

```
:::spoiler Out:
```python

```
:::
