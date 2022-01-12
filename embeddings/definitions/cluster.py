import pandas as pd
import torch
from sklearn.cluster import KMeans

for idx0 in range(5):
    for idx1 in range(5):
        for idx2 in range(5):
            df = pd.read_pickle(f"/home/ben/datasets/dict_word_embeddings_{idx0}_{idx1}_{idx2}.pkl")
            
            words = list(df.columns)
            n_clusters = min(len(words), 5)
            kmeans = KMeans(
                init="random",
                n_clusters=n_clusters,
                n_init=10,
                max_iter=300,
                random_state=42
            )

            data = torch.transpose(torch.tensor(df.values), 0, 1)

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

            i = 0
            for cluster in clusters:
                embeddings = {}
                for word in cluster:
                    embeddings[word] = df[word]
                sub_df = pd.DataFrame(embeddings)
                sub_df.to_pickle(f"/home/ben/datasets/dict_word_embeddings_{idx0}_{idx1}_{idx2}_{i}.pkl")
                i += 1

