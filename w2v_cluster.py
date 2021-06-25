import numpy as np
from gensim.models import word2vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import sys

model = word2vec.Word2Vec.load("word2vec.finance_news_cbow500.model")
max_vocab = 50000000000
vocab = list(model.wv.key_to_index.keys())[:max_vocab]
vectors = [model.wv[word] for word in vocab]
n_clusters = 14
kmeans_model = KMeans(n_clusters=n_clusters, verbose=1, random_state=42)
kmeans_model.fit(vectors)

cluster_labels = kmeans_model.labels_
cluster_to_words = defaultdict(list)

for cluster_id, word in zip(cluster_labels, vocab):
    cluster_to_words[cluster_id].append(word)

if len(sys.argv) >= 2:
    if sys.argv[1] == '-f' and sys.argv[2] == '':
        print("Error: no designated file name.")
        exit()
    if sys.argv[1] == '-f':
        file = sys.argv[2]
"""
i = 1
for word in cluster_to_words.values():
    print(cluster_to_words.keys(), word[::])
    print("----------------------------------")
    if len(sys.argv) >= 2 and sys.argv[1] == '-f':
        with open(file, 'a',  newline='') as f:
            writer = csv.writer(f,  lineterminator='\n')
#            writer.writerow(list(i))
            writer.writerow(word[::])
        f.close()

    i += 1
"""

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

# https://chusotsu-program.com/scikit-learn-clustering-plot/

import pandas as pd
from sklearn.decomposition import PCA

df = pd.DataFrame(vectors)
df['cluster'] = cluster_labels

x = df
pca = PCA(n_components=3)
pca.fit(x)
sc = StandardScaler()
#x_pca = pca.transform(x)
x_pca = sc.fit_transform(x)
pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = df['cluster']

"""
https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
"""
p = []
colors=['black', 'gray', 'silver', 'rosybrown', 'firebrick', 'red',\
        'darksalmon', 'sienna', 'tan', 'gold', 'darkkhaki', 'olivedrab',\
        'darkgreen', 'seagreen', 'lightseagreen', 'darkturquoise',\
        'deepskyblue', 'royalblue', 'navy', 'blue', 'mediumpurple',\
        'm', 'mediumvioletred', 'palevioletred']
for i in df['cluster'].unique():
    tmp = pca_df.loc[pca_df['cluster'] == i]
    scatter = plt.scatter(tmp[0], tmp[1], color=colors[i], label=i)
#    classes = pca_df['cluster']

plt.legend(loc='upper right')

plt.show()
"""
colors = ['red', 'blue', 'green']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

for i in range(n_clusters):
  ax.scatter(vectors[:, 0][results==i], vectors[:, 1][results==i], color=colors[i], alpha=0.5)
#  ax.scatter(centers[i, 0], centers[i, 1], marker='x', color=colors[i], s=300)

ax.set_title('k-means(Iris)', size=16)
#ax.set_xlabel(iris.feature_names[0], size=14)
#ax.set_ylabel(iris.feature_names[1], size=14)

plt.show()
"""
from sklearn.mixture import GaussianMixture
bics = np.array([])
nc = np.arange(1,20)  # クラスタ数：1から100
for k in nc:
    if k%20==0: print('k=',k)
    bics = np.r_[bics, GaussianMixture(k,n_init=10).fit(x).bic(x)]
bic_optk = nc[np.argmin(bics)]
bic_optk               # BICが最も小さくなるクラスタ数
plt.plot(nc,bics)      # クラスタ数に対するBICの値をプロット
plt.show()
