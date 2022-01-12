from openai.embeddings_utils import get_embedding
import torch

model = "ada-similarity"

with open("words.txt") as f:
    lines = ''.join(f.readlines()).split('\n')
    words = lines[:len(lines)-1]

embeddings = torch.tensor([get_embedding(word, model) for word in words])

torch.save(embeddings, 'embeddings.pt')
