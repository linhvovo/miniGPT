import ...

# hyperparameters
... = 4 # how many independent sequences will we process in parallel?
... = 8 # what is the maximum context length for predictions?
device = '...' ... torch.....is_available() ... '...'
# ------------

torch.manual_seed(1337)

with open('...', 'r', encoding='...') as f:
    text = f.read()

# get all the unique characters that occur in this text
chars = sorted(...(...(text)))
alphabet_size = len(chars)
# create a mapping from characters to integers
... = { ch:i for i,ch in enumerate(chars) }
... = { i:ch for i,ch in enumerate(chars) }
... = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
... = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# split data into training and validation data
data = torch....(...(text), dtype=torch.long)
n = ...(0.9*len(data)) # first 90% will be train, rest val
... = data[:n]
... = data[n:]

# generate a small batch of data of inputs x and targets y
def ...(...):
    data = ... ... split == 'train' ... ...
    indices = torch....(len(data) - ..., (...,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y

# print out and check results
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
