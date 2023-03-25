#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset,DataLoader
from nets import log
from nets import TransformerModel

torch.manual_seed(0)

# Define special symbols and indices
PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<pad>', '<bos>', '<eos>']

def generate_subsequent_mask(sz: int) -> Tensor:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged.
    If a FloatTensor is provided, it will be added to the attention weight.
    """
    #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return torch.triu(torch.ones(sz, sz), diagonal=1) == 1

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # 右上角是 -inf，是不是错了？
    # 改版后右上角是 True
    tgt_mask = generate_subsequent_mask(tgt_seq_len)
    # 全是 False，表示全 attend
    #src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return tgt_mask, src_padding_mask, tgt_padding_mask

def _construct_dataset(datas,lb_vocab,cn_vocab):
    data_input  = [torch.tensor(lb_vocab(['<bos>']+lb+['<eos>']),dtype=torch.long) for lb,cn in datas]
    data_target = [torch.tensor(cn_vocab(['<bos>']+cn+['<eos>']),dtype=torch.long) for lb,cn in datas]
    data_input  = pad_sequence(data_input,batch_first=True,padding_value=PAD_IDX)
    data_target = pad_sequence(data_target,batch_first=True,padding_value=PAD_IDX)
    dataset     = TensorDataset(data_input, data_target)
    return dataset

def get_word_tokenizer(words):
    "return a tokenizer which yield char except for given words"
    wordstart = set(w[0] for w in words)
    def tokenizer(s):
        s = s.strip()
        mem = None
        for i,c in enumerate(s):
            if c not in wordstart:
                if mem is None:
                    yield c
                else:
                    yield mem+c
                    mem = None
            else:
                mem = c
    return tokenizer


log("loading datas")
with open("2021-5.txt","r") as f:
    lines = f.readlines()

cn_tokenizer = get_word_tokenizer(["睡覺","離開","落下","到達","跳舞","狩獵"])
lb_tokenizer = get_word_tokenizer(["ts",])
datas = [[]]
for l in lines:
    if l.startswith("----"):
        datas.append([])
    else:
        lb,cn = l.split()
        datas[-1].append(([i for i in lb_tokenizer(lb)],[i for i in cn_tokenizer(cn)]))

train = datas[0]
test1 = datas[1]
test2 = datas[2]

lb_vocab = build_vocab_from_iterator([lb for lb,cn in train], specials=special_symbols)
cn_vocab = build_vocab_from_iterator([cn for lb,cn in train], specials=special_symbols)
print(lb_vocab.get_itos())
print(cn_vocab.get_itos())

train_dataset    = _construct_dataset(train,lb_vocab,cn_vocab)
test1_dataset    = _construct_dataset(test1,lb_vocab,cn_vocab)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=True,drop_last=True)
test1_dataloader = DataLoader(dataset=test1_dataset,batch_size=6,shuffle=False,drop_last=True)
log("loaded %d,%d data"%(len(train_dataset),len(test1_dataset)))

ntokens_1 = len(lb_vocab)
ntokens_2 = len(cn_vocab)
model = TransformerModel(len(lb_vocab),len(cn_vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, betas=(0.9, 0.98), eps=1e-9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
#optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
num_params = sum(p.numel() for p in model.parameters())
log("%d parameters"%(num_params))

def train():
    min_test_loss = float('inf')
    for epoch in range(1001):
        train_loss = train_epoch()
        test_loss  = evaluate()

        if epoch<3 or epoch%20==0:
            log("epoch %2d: %7.4f %.4f"%(epoch,train_loss,test_loss))
        if epoch%40==0 and test_loss<min_test_loss:
            min_test_loss = test_loss
            trans = []
            for lb,cn in test1:
                trans.append("%12s %12s %s"%("".join(lb),"".join(cn),translate(model,lb)))
            log("epoch %2d: %7.4f %.4f\n%s"%(epoch,train_loss,test_loss,"\n".join(trans)))

def translate(model: torch.nn.Module, src_sentence: list):
    model.eval()
    src = torch.tensor(lb_vocab(['<bos>']+src_sentence+['<eos>']),dtype=torch.long).unsqueeze(0)
    tgt = _greedy_decode(model,src).flatten()
    return "".join(cn_vocab.lookup_tokens(list(tgt.cpu().numpy()))).replace("<bos>","").replace("<eos>","")

def _greedy_decode(model, src, max_len=10, start_symbol=BOS_IDX):
    memory = model.encode(src) # shape: [1, L, embd_size(d_model)]
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol)
    for i in range(max_len-1):
        tgt_mask = generate_subsequent_mask(ys.size(1))
        out      = model.decode(ys, memory, tgt_mask)
        prob     = model.generator(out[0,-1])
        _, next_word = torch.max(prob, dim=-1)
        next_word    = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1, dtype=torch.long).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys

def train_epoch():
    model.train()  # turn on train mode
    total_loss = 0.0
    for i,(src,tgt) in enumerate(train_dataloader):
        tgt_input = tgt[:,:-1]
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src,tgt_input,tgt_mask,src_padding_mask,tgt_padding_mask)
        tgt_out = tgt[:,1:]
        loss = loss_fn(logits.reshape(-1, ntokens_2), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def evaluate():
    model.eval()
    total_loss = 0.0
    nchar      = 0
    for i,(src,tgt) in enumerate(test1_dataloader):
        tgt_input = tgt[:,:-1]
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src,tgt_input,tgt_mask,src_padding_mask,tgt_padding_mask)
        tgt_out = tgt[:,1:]
        loss = F.cross_entropy(logits.reshape(-1, ntokens_2), tgt_out.reshape(-1), ignore_index=PAD_IDX, reduction='sum')
        total_loss += loss.item()
        nchar      += (tgt_out!=PAD_IDX).sum().item()
    return total_loss/nchar


if __name__=="__main__":
    #load_data()
    train()
    #print(translate(model,"aharapyryk"))
