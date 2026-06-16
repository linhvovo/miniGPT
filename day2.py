import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
... = 5000
... = 1e-3
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all the unique characters that occur in this text
chars = sorted(list(set(text)))
alphabet_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# split data into training and validation data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y

... ...(nn.Module):

    def __init__(self, alphabet_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.... = nn.Embedding(..., ...)

    def forward(self, inputs, targets):
        # idx and targets are both (B,T) tensor of integers
        ... = self.token_embedding_table(...) # (B,T,C)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        ... = F....(..., ...)

        return loss

    def ...(self, inputs, num_new_tokens):
        # inputs is (B, T) array of indices in the current context
        for _ in range(...):
            # get the predictions
            logits = self.token_embedding_table(inputs)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            ... = F....(logits, dim=-1) # (B, C)
            # sample from the distribution
            ... = torch....(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            inputs = torch.cat((inputs, new_tokens), dim=1) # (B, T+1)
        return inputs

model = BigramLanguageModel(alphabet_size)
m = model.to(device)

# create a PyTorch optimizer
... = torch.optim....(model.parameters(), lr=...)

for iter in range(...): # increase number of steps for good results

    # sample a batch of data
    inputs, targets = ...('train')

    # evaluate the loss
    ... = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    .......()
    .......()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, num_new_tokens=500)[0].tolist()))
