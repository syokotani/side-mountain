import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import sys

model = word2vec.Word2Vec.load("word2vec.finance_news_cbow500.model")
max_vocab = 500000000000
vocab = list(model.wv.key_to_index.keys())[:max_vocab]
vectors = [model.wv[word] for word in vocab]

from sklearn.mixture import GaussianMixture
gm = GaussianMixture(3,n_init=10).fit(vectors)
labels = gm.predict(vectors)

cluster_to_words = defaultdict(list)
for cluster_id, word in zip(labels, vocab):
    cluster_to_words[cluster_id].append(word)

for i in range(len(cluster_to_words.values())):
    print(list(cluster_to_words.keys())[i], list(cluster_to_words.values())[i][:250])
    print("----------------------------------")

data = {'word': vocab, 'label': labels}
word_df = pd.DataFrame(data)
print(word_df)

"""
cluster_labels = gm.labels_
cluster_to_words = defaultdict(list)

for cluster_id, word in zip(cluster_labels, vocab):
    cluster_to_words[cluster_id].append(word)

if len(sys.argv) >= 2:
    if sys.argv[1] == '-f' and sys.argv[2] == '':
        print("Error: no designated file name.")
        exit()
    if sys.argv[1] == '-f':
        file = sys.argv[2]


for i in range(len(cluster_to_words.values())):
    print(list(cluster_to_words.keys())[i], list(cluster_to_words.values())[i][::])
    print("----------------------------------")
    if len(sys.argv) >= 2 and sys.argv[1] == '-f':
        with open(file, 'a',  newline='') as f:
            writer = csv.writer(f,  lineterminator='\n')
            l = list(cluster_to_words.values())[i][::]
            l.insert(0,list(cluster_to_words.keys())[i])
            writer.writerow(l)
        f.close()



bics = np.array([])
nc = np.arange(1,20)  # クラスタ数：1から100
for k in nc:
    if k%20==0: print('k=',k)
    bics = np.r_[bics, GaussianMixture(k,n_init=10).fit(x).bic(x)]
bic_optk = nc[np.argmin(bics)]
bic_optk               # BICが最も小さくなるクラスタ数
plt.plot(nc,bics)      # クラスタ数に対するBICの値をプロット
plt.show()
"""
