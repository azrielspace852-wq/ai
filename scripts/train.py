import json, math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- model ----------
class CausalAttn(nn.Module):
    def __init__(self, dim, n_heads, max_seq, drop=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.nh, self.hd = n_heads, dim // n_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_seq, max_seq)).view(1,1,max_seq,max_seq),
            persistent=False,
        )
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).view(B,T,3,self.nh,self.hd).permute(2,0,3,1,4)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, nh, max_seq, drop=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim); self.attn = CausalAttn(dim, nh, max_seq, drop)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.GELU(),
            nn.Linear(4*dim, dim), nn.Dropout(drop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, dim, nh, nl, max_seq):
        super().__init__()
        self.max_seq  = max_seq
        self.tok_emb  = nn.Embedding(vocab, dim)
        self.pos_emb  = nn.Embedding(max_seq, dim)
        self.blocks   = nn.Sequential(*[Block(dim, nh, max_seq) for _ in range(nl)])
        self.ln_f     = nn.LayerNorm(dim)
        self.head     = nn.Linear(dim, vocab, bias=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x); x = self.ln_f(x)
        return self.head(x)

# ---------- dataset ----------
class SeqDS(Dataset):
    def __init__(self, seqs, seq_len):
        self.seqs, self.seq_len = seqs, seq_len
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        need = self.seq_len + 1
        s = (s + [0]*need)[:need]
        return torch.tensor(s[:-1]), torch.tensor(s[1:])

# ---------- main ----------
def main():
    mode   = os.environ.get('MODE', 'pretrain')
    bs     = int(os.environ['BATCH_SIZE'])
    steps  = int(os.environ['STEPS'])
    dim    = int(os.environ['EMBED_DIM'])
    nh     = int(os.environ['NUM_HEADS'])
    nl     = int(os.environ['NUM_LAYERS'])
    sl     = int(os.environ['SEQ_LEN'])
    lr     = float(os.environ['LEARNING_RATE'])
    warmup = int(os.environ.get('WARMUP_STEPS', '100'))

    data  = json.load(open('dataset.json'))
    vocab = json.load(open('vocab.json'))
    V     = len(vocab)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[info] dev={dev} V={V} N={len(data)}")

    dl = DataLoader(SeqDS(data, sl), batch_size=bs, shuffle=True, drop_last=True)
    model = GPT(V, dim, nh, nl, sl + 1).to(dev)
    print(f"[info] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if mode == 'finetune' and os.path.exists('model_path.txt'):
        ck = open('model_path.txt').read().strip()
        if ck and os.path.exists(ck):
            print(f"[info] load {ck}")
            st = torch.load(ck, map_location=dev)
            if isinstance(st, dict) and 'model' in st and isinstance(st['model'], dict):
                st = st['model']
            ms = model.state_dict()
            filt = {k:v for k,v in st.items() if k in ms and v.shape == ms[k].shape}
            print(f"[ok] loaded {len(filt)}/{len(ms)} tensors")
            model.load_state_dict(filt, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.01)

    def sched_fn(s):
        if s < warmup:
            return (s + 1) / max(1, warmup)
        p = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)

    model.train()
    step, acc, cnt = 0, 0.0, 0
    while step < steps:
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, V), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

            acc += loss.item(); cnt += 1
            if step % 25 == 0:
                print(f"step {step}/{steps} | loss {acc/cnt:.4f} | lr {sch.get_last_lr()[0]:.6f}")
                acc = cnt = 0
            step += 1
            if step >= steps: break

    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/model.bin')
    json.dump(vocab, open('output/vocab.json','w', encoding='utf-8'), ensure_ascii=False)
    print("[ok] output/model.bin + output/vocab.json")

if __name__ == '__main__':
    main()
