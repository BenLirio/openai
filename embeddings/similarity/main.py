import openai
import numpy as np
from scipy.stats import pearsonr

embed = lambda x, engine: np.array(openai.Engine(id=engine).embeddings(input=x).data[0].embedding)
euclid_sim = lambda a, b: (1-np.linalg.norm(a - b))
pearson_sim = lambda a, b: pearsonr(a,b)[0]

embed_ada = lambda x:  embed(x, "babbage-similarity")

vocab = ["bow", "sling", "Axe", "Dagger"]
embeddings = [embed_ada(x) for x in vocab]


euclid_sim_matrix = np.zeros(shape=(len(embeddings), len(embeddings)))
pearson_sim_matrix = np.zeros(shape=(len(embeddings), len(embeddings)))


for x in range(0, len(embeddings)):
    for y in range(0, len(embeddings)):
        euclid_sim_matrix[x][y] = euclid_sim(embeddings[x], embeddings[y])

for x in range(0, len(embeddings)):
    for y in range(0, len(embeddings)):
        pearson_sim_matrix[x][y] = pearson_sim(embeddings[x], embeddings[y])

round_2 = np.vectorize(lambda x: round(x, 2))
print("Euclid")
print(', '.join(vocab))
print(round_2(euclid_sim_matrix))
print("Pearson")
print(', '.join(vocab))
print(round_2(pearson_sim_matrix))

