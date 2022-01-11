import torch
from random import randint
from torch import nn
import pandas as pd
from scipy import spatial

df = pd.read_pickle("numbers.pkl")
embeddings = torch.tensor([embedding for _, embedding in df.values])

embedding_size = 2048
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size*2),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

get_embeddings = lambda a, b: torch.cat((embeddings[a].clone(), embeddings[b].clone()))
tree = spatial.cKDTree(embeddings)

def pred(a, b):
    pred = model(get_embeddings(a, b))
    _, idx = tree.query(pred.detach().numpy(), k=3)
    return idx

N = 10
for _ in range(N):
    num1 = randint(0, N-1)
    num2 = randint(0, N-num1-1)
    num3 = pred(num1, num2)
    print(f"{num1} + {num2} = {num3}")

