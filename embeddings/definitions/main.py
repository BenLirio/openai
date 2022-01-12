import torch
import os
from sklearn.cluster import KMeans

with open("words.txt") as f:
    lines = ''.join(f.readlines()).split('\n')
    words = lines[:len(lines)-1]

dataset_folder = '/home/ben/datasets'
embeddings_file = '4k-common-word-embeddings-2048dim.pt'
embeddings = torch.load(os.path.join(dataset_folder, embeddings_file))
n_clusters=5
kmeans = KMeans(
    init="random",
    n_clusters=n_clusters,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(embeddings)
clusters = [[], [], [], [], []]
labels = kmeans.labels_
for i in range(len(labels)):
    clusters[labels[i]].append(words[i])
for cluster in clusters:
    print(', '.join(cluster))
