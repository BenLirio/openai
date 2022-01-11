import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import torch
from random import randint

embedding_size = 2048
batch_size = 64
df = pd.read_pickle("numbers.pkl")
embeddings = torch.tensor([embedding for _, embedding in df.values])
N = len(df.values)
N_train = 1024<<3
training_data = []
# Train data
for _ in range(N_train):
    num1 = randint(0, N-1)
    num2 = randint(0, N-num1-1)
    num3 = num1 + num2
    x = torch.cat((embeddings[num1].clone(), embeddings[num2].clone()))
    y = embeddings[num3].clone()
    training_data.append((x, y))

train_dataloader = DataLoader(training_data, batch_size=batch_size)

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

learning_rate = 1e-3
epochs = 1

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>6f} [{current:>5d}/{size:>5d}]")

for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done.")

torch.save(model.state_dict(), 'model_weights.pth')
