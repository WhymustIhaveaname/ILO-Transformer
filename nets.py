#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import traceback
import os
import torch
from typing import Tuple
from torch import nn,Tensor
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, ntoken_src, ntoken_tgt, d_model=32, nhead=1, d_ff=128, nlayers=2):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.src_encoder = nn.Embedding(ntoken_src, d_model)
        self.tgt_encoder = nn.Embedding(ntoken_tgt, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers,
                                        dim_feedforward=d_ff, dropout=0.1, batch_first=True)
        self.generator   = nn.Linear(d_model, ntoken_tgt)

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src,tgt,tgt_mask,src_padding_mask,tgt_padding_mask) -> Tensor:
        src = self.src_encoder(src) * math.sqrt(self.d_model)
        tgt = self.tgt_encoder(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        #(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        out = self.transformer(src,tgt,None,tgt_mask,None,src_padding_mask,tgt_padding_mask,src_padding_mask)
        return self.generator(out)

    def encode(self, src: Tensor):
        src = self.src_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = self.tgt_encoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        return self.transformer.decoder(tgt,memory,tgt_mask)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

LOGLEVEL = {0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE  = "net.log"

def log(*msg,l=1,end="\n",logfile=LOGFILE):
    msg=", ".join(map(str,msg))
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    perfix="%s [%s,%s:%03d]"%(now_str,lstr,st.name,st.lineno)
    if l<3:
        tempstr="%s %s%s"%(perfix,str(msg),end)
    else:
        tempstr="%s %s:\n%s%s"%(perfix,str(msg),traceback.format_exc(limit=5),end)
    print(tempstr,end="")
    if l>=1:
        with open(logfile,"a") as f:
            f.write(tempstr)