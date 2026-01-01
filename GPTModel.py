import torch 
import torch.nn as nn 
from torch.nn import functional as Fn
from colorama import Fore

torch.manual_seed(42)
block_size = 8 # lenth of one line 
batch_size = 4 # How many lines we need to process in parallel 
Lrate = 0.1
device = torch.device("mps") 
eval_iters = 200
steps = 10000

with open("Shakespeare_text.txt") as f:
    texts = f.read()
# print(len(texts))

chars = sorted(list(set(texts)))
vocab_size = len(chars)
#print(f"number of characters: {chars}")
#print(f"lenght of vocab: {vocab_size}")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}


def encode (s):
    encode_char = []
    for c in list(s):
        encode_char.append(stoi[c])
    return encode_char    

def decode (i):
    decode_int = []
    for k in i:
        decode_int.append(itos[k])
    return "".join(decode_int)


data = torch.tensor(encode(list(texts)),dtype=torch.long)
# print(data.shape,data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
len(train_data)


train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    #print(f"When input is {context}, the target is {target}")



def get_batch (split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix] )
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y = x.to(device) , y.to(device)
    return x,y

xb,yb = get_batch('train')
# print("inputs")
# print(xb.shape)
# print(xb)
# print("--------")
# print("target")
# print(yb.shape)
# print(yb)
# print("--------")

for b in range(batch_size):
    for k in range(block_size):
        context = xb[b,:k+1]
        target = yb[b,k]
        # print(f"for context {context}, target is {target}")

@torch.no_grad()
def extimate_loss():
    out = {}
    GPTmodel.eval()
    for split in ['train_data','test_data']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits ,loss = GPTmodel(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    GPTmodel.train()
    return out 

## Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,target=None):
        logits = self.token_embeding_table(idx)
        if target is None:
            loss = None
        else:
             # (B,T,C) B = batch (4), T = time(8), C = channel (vocab size)
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = Fn.cross_entropy(logits,target)
        return logits,loss
          
        
        
    
    def generate(self,idx, max_new_token):
        for _ in range(max_new_token):
            logits,loss = self(idx)
            logits = logits[:,-1,:] # (B,C)
            probs = Fn.softmax(logits,dim=-1) # (B,C)
            idx_next = torch.multinomial(probs,num_samples=1,replacement=True)#(B,1)
            idx = torch.cat((idx,idx_next), dim=1) # (B,T+1)
        return idx
    

GPTmodel = BigramLanguageModel(vocab_size=vocab_size)
GPTmodel = GPTmodel.to(device)
logits,loss = GPTmodel(xb,yb)
# print(logits.shape)
# print(loss)

# print(decode(GPTmodel.generate(idx = torch.zeros((1,1),dtype=torch.long), max_new_token=100)[0].tolist()))


optimiser = torch.optim.AdamW(GPTmodel.parameters(),lr=Lrate)


for step in range(steps):
    
    if step % steps == 0:
        losses = extimate_loss()
        print(f"loss in train data: {losses['train_data']} and loss in test data: {losses['test_data']}")


    xb,yb = get_batch('train')

    logits,loss = GPTmodel(xb,yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

print(loss.item())
context = torch.zeros((1,1),dtype=torch.long, device=device)
print(Fore.GREEN + decode(GPTmodel.generate(context, max_new_token=500)[0].tolist()))