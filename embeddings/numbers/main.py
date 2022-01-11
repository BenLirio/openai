import pandas as pd
import torch
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_pickle("numbers.pkl")


kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=300,
    random_state=42
)


data = torch.tensor([embedding for (_, embedding) in df.values])
kmeans.fit(data)

clusters = {}
for i in range(len(data)):
    label = kmeans.labels_[i]
    cur = clusters.get(label, [])
    cur.append(i)
    clusters[label] = cur
for key in clusters:
    print(key, clusters[key])
