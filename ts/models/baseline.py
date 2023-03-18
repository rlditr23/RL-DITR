import math
from typing import Optional

import torch
from torch import nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, n_outputs, n_input, n_hidden=256, nlayers=3, max_len=512, dropout: float = 0.0):
        super(LSTM, self).__init__()
        self.model_type = 'LSTM'
        self.n_outputs = n_outputs
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.max_len = max_len
        self.dropout = dropout

        self.rnn = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=nlayers)
        self.decoder = MLP(n_hidden, [n_hidden], n_outputs)
        self.h0 = nn.Parameter(torch.zeros((self.nlayers, 1, self.n_hidden)),requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros((self.nlayers, 1, self.n_hidden)),requires_grad=False)

    # def generate_h0_c0(self):
        # h0 = torch.zeros((self.nlayers, 1, self.n_hidden))
        # c0 = torch.zeros((self.nlayers, 1, self.n_hidden))
        # return h0, c0

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        '''
        X: (B,L,E)
        x_mask: (B,L,E) (bool) 
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        '''
        # if self.h0 is None:
        #     self.h0, self.c0 = self.generate_h0_c0()
        #     self.h0 = self.h0.to(x.device)
        #     self.c0 = self.c0.to(x.device)
        B = x.shape[0]
        h0 = self.h0.data.repeat(1, B, 1)
        c0 = self.c0.data.repeat(1, B, 1)
        output, (_, _) = self.rnn(x.transpose(0, 1), (h0, c0))
        output = output.transpose(0, 1)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        b, l = x.shape[:2]
        shape = list(x.shape)
        shape[-1] = self.pe.shape[-1]
        return self.pe[None, offset:offset + l].expand(shape)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class TransformerCausalEncoder(nn.Module):
    '''
    X[1:T] -> Y[1:T]
    X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]
    '''

    def __init__(self, n_input, n_hidden, nhead, nhid, nlayers,
                 max_len: int = 512, dropout: float = 0.0):
        super(TransformerCausalEncoder, self).__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.nhid = nhid
        self.max_len = max_len
        self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(max_len * 2), requires_grad=False)

        self.input_encoder = nn.Sequential(
            nn.Linear(n_input + n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
        )
        self.pos_encoder = PositionalEncoding(n_hidden)

        encoder_layers = nn.TransformerEncoderLayer(n_hidden, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        # src_mask: If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        '''
        X: (B,L,E)
        x_mask: (B,L,E) (bool) 
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        '''
        b, l, d = x.shape
        x = torch.cat([x, self.pos_encoder(x)], dim=-1)
        x = self.input_encoder(x)
        x = x.transpose(0, 1)
        if dynamics:
            forward_mask = self.generate_square_subsequent_mask(x.shape[0])
        else:
            forward_mask = self.forward_mask[:l, :l]
        feat = self.transformer_encoder(x, forward_mask, padding_mask)
        feat = feat.transpose(0, 1)
        return feat


class TransformerCausalDecoder(nn.Module):
    '''
    X[1:T] -> Y[1:T]
    X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]
    '''
    def __init__(self, n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3,
                 max_len=512, dropout: float = 0.5):
        super(TransformerCausalDecoder, self).__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.nhid = nhid
        self.max_len = max_len
        self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(max_len * 2), requires_grad=False)

        self.input_encoder = nn.Sequential(
            nn.Linear(n_input + n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
        )
        self.pos_encoder = PositionalEncoding(n_hidden)

        decoder_layers = nn.TransformerDecoderLayer(n_hidden, nhead, nhid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        # src_mask: If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, q: torch.Tensor, v: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                dynamics: bool = False, offset=0):
        '''
        X: (B,L,E)
        x_mask: (B,L,E) (bool) 
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        '''
        b, l, d = q.shape
        q = torch.cat([q, self.pos_encoder(q, offset=offset)], dim=-1)
        q = self.input_encoder(q)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)
        if dynamics:
            forward_mask = self.generate_square_subsequent_mask(x.shape[0])
        else:
            forward_mask = self.forward_mask[:l, :l]
        feat = self.transformer_decoder(q, v, tgt_mask=forward_mask, memory_mask=forward_mask,
                                        memory_key_padding_mask=padding_mask, tgt_key_padding_mask=padding_mask)
        feat = feat.transpose(0, 1)
        return feat


class TransformerModel(nn.Module):
    '''
    X[1:T] -> Y[1:T]
    X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]
    '''

    def __init__(self, n_outputs, n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3,
                 max_len=512, dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.nhid = nhid
        self.max_len = max_len
        self.encoder = TransformerCausalEncoder(n_input, n_hidden, nhead, nhid, nlayers, max_len=max_len,
                                                dropout=dropout)
        self.decoder = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        '''
        X: (B,L,E)
        x_mask: (B,L,E) (bool) 
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        '''

        feat = self.encoder(x, padding_mask=padding_mask)
        output = self.decoder(feat)
        return output


class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size,
                 output_activation=torch.nn.Identity, activation=torch.nn.ELU):
        super(MLP, self).__init__()

        sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        return self.mlp(x)


class CNNCausalBlock(nn.Module):
    def __init__(self, input_size, output_size, activation=torch.nn.ELU):
        super(CNNCausalBlock, self).__init__()
        self.cnn = torch.nn.Conv1d(input_size, output_size, kernel_size=3, padding=2)
        self.act = activation()

    def forward(self, x):
        x = self.cnn(x)
        x = self.act(x)
        x = x[..., :-2]
        return x


class CNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size,
                 output_activation=torch.nn.Identity, activation=torch.nn.ELU):
        super(CNN, self).__init__()

        sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [CNNCausalBlock(sizes[i], sizes[i + 1], act)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        x = x.transpose(1, 2)
        y = self.layers(x)
        y = y.transpose(1, 2)
        return y


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class CausalMultiHeadAttn(nn.Module):
    '''
    p(y|x) model

    X[1:T] -> Y[1:T]

    X[1:t]+Y[1:t]+Y[t+1:T] -> Y[t+1:T]

    norm is placed behind the active layer https://www.zhihu.com/question/283715823
    '''

    def __init__(self, n_input, n_hidden, nhead, max_len: int = 512, dropout: float = 0.0):
        super(CausalMultiHeadAttn, self).__init__()
        self.model_type = 'Transformer'
        self.n_hidden = n_hidden
        self.max_len = max_len
        self.forward_mask = nn.Parameter(self.generate_square_subsequent_mask(max_len * 2), requires_grad=False)
        self.pos_encoder = PositionalEncoding(n_hidden)

        self.input_encoder = nn.Sequential(
            nn.Linear(n_input + n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
        )
        self.transformer_encoder = nn.MultiheadAttention(embed_dim=n_input, num_heads=nhead, batch_first=True)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        # src_mask: If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, q: torch.Tensor, v: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                dynamics: bool = False):
        '''
        X: (B,L,E)
        x_mask: (B,L,E) (bool) 
            - imputation and padding
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        NOTE:
        - transformer_encoder input
            - src (S,N,E)
            - src_mask (S,S)
            - src_key_padding_mask (N,S)
        '''
        b, l, d = v.shape
        v = torch.cat([v, self.pos_encoder(v)], dim=-1)
        v = self.input_encoder(v)
        if dynamics:
            forward_mask = self.generate_square_subsequent_mask(l)
        else:
            forward_mask = self.forward_mask[:l, :l]
        feat, attn_output_weights = self.transformer_encoder(query=q, key=v, value=v, key_padding_mask=padding_mask,
                                                             attn_mask=forward_mask)
        return feat


if __name__ == '__main__':
    n_input = 147
    n_hidden = 256
    nhead = 8
    nhid = 512
    nlayers = 3
    n_outputs = 4
    max_len = 128
    model = LSTM(n_outputs=n_outputs, n_input=n_input, max_len=max_len)

    batch_size = 3

    x = torch.rand(batch_size, max_len, n_input)
    y = model(x)
    print(y.shape)

    model_scripted = torch.jit.script(model)
