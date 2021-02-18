import psycopg2
from nltk import *
import sys
from lemma_module import lemma
import datetime

# look up the last record (article_id) from 'market_news_bow' table
try:
    conn = psycopg2.connect("""dbname = 'finance_news' user = 'yokotani' host = '192.168.0.7'  password = 'Qx26aI5X'""")
except ValueError:
    print("'finance_news' connection fail @ 192.168.0.7 server")
cur = conn.cursor()

# if no record in table 'market_news_bow', return None
date_b = '2018-01'
date_t = str(int(datetime.date.today().year)) + '-' + str(int(datetime.date.today().month))
# date_t = '2021-01'
seq = (date_b, date_t)
cur.execute("""SELECT bow
	FROM public.finance_news_bow
	WHERE to_char(date, 'YYYY-MM') >= %s
	AND to_char(date, 'YYYY-MM') <= %s
	group by date, bow order by date DESC;""", seq)
raw = cur.fetchall()
cur.close()
conn.close()

bow = []
for i in range(len(raw)):
    for j in range(len(raw[i])):
        bow += raw[i]
for i in range(len(bow)):
    bow[i] = lemma(bow[i])

from gensim.models import word2vec

model = word2vec.Word2Vec(bow, size=250, min_count=10, window=5, sg=1)
#model = word2vec.Word2Vec(bow, size=500, min_count=5, window=5, sg=0)
index2word=model.wv.index2word
model.save("word2vec.finance_news_cbow500.model")

similar_words=model.wv.most_similar(positive=['bull'],topn=50)
print(*[" ".join([v, str("{:.2f}".format(s))]) for v, s in similar_words], sep="\n")
#similarity = model.wv.similarity(w1="u.s.", w2="china")
#word_vectors=model.wv.syn0
#print(word_vectors.shape)
#print(word_vectors)
