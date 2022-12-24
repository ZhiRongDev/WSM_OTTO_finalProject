* Candidate ReRank Model - [LB 0.575] used the "Test Data Leak - LB Boost"
which means the result can't be used in the real life.


* gensim word2vec only use the idea of word2vec to do word embedding, and calculate the cosine similarity 
of each transformed word vector to rank the output. Hence we could implement spotify/Annoy, which is a 
faster way to find the nearest neighbors. Besides, I think we could also consider using the word embedding result 
of gensim word2vec model to carry out another ML experiment (Since it reduce the dimension of vectors).

