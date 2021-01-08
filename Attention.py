import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from matplotlib import pyplot as plt

# emb+pos_emb = IncredableAI ref
class Embedding(nn.Module):
    def __init__(self, voca_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(voca_size, emb_dim)
    def forward(self,x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_seq_len=200, pos_dropout=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, emb_dim)
        # CODE IMPLEMENTATION
        # for pos in range(max_seq_len):
        #     for i in range(0, emb_dim, 2):
        #         pe[pos, i] = math.sin(pos/(10000**((2*i)/emb_dim)))
        #         pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/emb_dim)))
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,emb_dim, 2).float()*(-math.log(10000.0)/emb_dim))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        # pe 와 forward 수정
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(pos_dropout) # dropout을 해야하는가 ->네
    def forward(self, x): # in evel, x = emb_dec(64,9,512)->b=64, 9?,emb=512
        # pos = self.pe[:, :x.size(1)]  # self.pe(1,200,512)
        # pos_out = x + pos.detach()
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# <Positional encoding Graph>
# max_seq_len = 200
# embed_dim = 512 # 논문에서 d_model
#
# pe = np.zeros([max_seq_len, embed_dim])
# for pos in range(max_seq_len):
#     for i in range(0, embed_dim, 2):
#         pe[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
#         pe[pos, i + 1] = np.cos(pos / (10000 ** (i / embed_dim)))
#
# fig, ax = plt.subplots(figsize=(15, 9))
# cax = ax.matshow(pe, cmap=plt.cm.YlOrRd)
# fig.colorbar(cax)
# ax.set_title('Positional Emcoder Matrix', fontsize=18)
# ax.set_xlabel('Embedding Dimension', fontsize=14)
# ax.set_ylabel('Sequence Length', fontsize=14)
# plt.show()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout_attn):
        super().__init__() # 자기 parent class인 nn.Module의 함수()를 쓰겠다는 것/ super()=nn.Module 이라는 뜻
        self.sqrt_dk = d_k ** 0.5
        self.dropout = nn.Dropout(dropout_attn) # for regularization -> 노드를 랜덤으로 없애는거가 아니구 매트릭스 값을 랜덤으로 없앰
        self.softmax = nn.Softmax(dim=-1) # 왜 dim이 -1인지 보기

    def forward(self,Q,K,V, mask = None):
        score_qk = torch.matmul(Q,K.transpose(-2, -1)) / self.sqrt_dk
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = score_qk.masked_fill(mask, -1e9) # masked_fill:
        else:
            attn = score_qk
        attn_prob = self.dropout(self.softmax(attn)) # softmax에 dim=-1 왜?
        context = torch.matmul(attn_prob, V)
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, dropout_attn, dropout_multi):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.d_k = embed_dim // n_head
        self.d_v = embed_dim // n_head

        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.final_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k, dropout_attn)

        self.dropout = nn.Dropout(dropout_multi)

    def forward(self,Q,K,V, mask):
        batch_size = Q.size(0)
        n_head = self.n_head
        d_k = self.d_k
        d_v = self.d_v

        q = self.linear_q(Q).view(batch_size, -1, n_head, d_k) # (128, 27, 8, 64)
        k = self.linear_k(K).view(batch_size, -1, n_head, d_k)
        v = self.linear_v(V).view(batch_size, -1, n_head, d_v)

        q = q.transpose(1, 2) # 왜지ㅠㅜ 사이즈맞추려구..?->(128, 8, 27, 64)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        context, attn_prob = self.attention(q,k,v, mask) # size 변경해야함
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) # contiguous():
        context = self.final_linear(context)
        total_context = self.dropout(context)
        return total_context, attn_prob

class PositionalFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout_ff):
        super().__init__()
        self.weight_1 = nn.Linear(embed_dim, d_ff, bias=True)
        self.weight_2 = nn.Linear(d_ff, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout_ff)

    def forward(self, x):
        residual = x
        x = F.relu(self.weight_1(x)) # nn.ReLU?
        x = self.dropout(x)
        x = self.weight_2(x)
        return x + residual
