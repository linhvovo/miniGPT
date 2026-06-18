import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
iterations = 5000
learning_rate = 1e-3
eval_interval = 500
eval_iterations = 200
n_features = 32
n_heads = 6
dropout = 0.2
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_features, head_size, bias=False)
        self.query = nn.Linear(n_features, head_size, bias=False)
        self.value = nn.Linear(n_features, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = inputs.shape
        k = self.key(inputs)   # (B,T,hs)
        q = self.query(inputs) # (B,T,hs)
        v = self.value(inputs) # (B,T,hs)
        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        outputs = weights @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return outputs
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        head_size = n_features // n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.combined = nn.Linear(head_size * n_heads, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = torch.cat([h(inputs) for h in self.heads], dim=-1)
        outputs = self.dropout(self.combined(outputs))
        return outputs

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(alphabet_size, n_features)
        self.position_embedding_table = nn.Embedding(block_size, n_features)
        self.lm_head = nn.Linear(n_features, alphabet_size)
        self.ln = nn.LayerNorm(n_features)
        self.sa = MultiHeadAttention()

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        # idx and targets are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(inputs) # (B,T,C)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embedding + position_embedding # (B,T,C)
        x = x + self.sa(self.ln(x))
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(iterations): # increase number of steps for good results...

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == iterations - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
