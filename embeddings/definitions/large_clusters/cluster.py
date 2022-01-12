import pandas as pd
import torch
from sklearn.cluster import KMeans

df = pd.read_pickle(f"/home/ben/datasets/dict_word_embeddings.pkl")

words = list(df.columns)
n_clusters = len(words)//5
kmeans = KMeans(
    init="random",
    n_clusters=n_clusters,
    n_init=10,
    max_iter=300,
    random_state=42
)

data = torch.transpose(torch.tensor(df.values), 0, 1)
data_norm = torch.norm(data, p=2, dim=1).detach()
data = data.div(data_norm.expand_as(data))

kmeans.fit(data)
labels = kmeans.labels_

clusters = []
for i in range(n_clusters):
    clusters.append([])

i = 0
for cluster in labels:
    word = words[i]
    clusters[cluster].append(word)
    i += 1

for cluster in clusters:
    for word in cluster:
        print(word)
    print(f"=========== Cluster size: {len(cluster)} ===============")
    input()

