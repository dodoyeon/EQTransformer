import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from matplotlib import pyplot as plt

class Embedding(nn.Module):
    def __init__(self, voca_size:int, emb_dim:int):
        super().__init__()
        self.embed_dim = emb_dim
        self.embedding = nn.Embedding(voca_size, emb_dim)
    def forward(self,x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=400, emb_dim=512, pos_dropout = 0.1):
        super().__init__()
        self.pe = np.zeros([max_seq_len,emb_dim])
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                self.pe[pos, i] = np.sin(pos/(10000**(i/emb_dim)))
                self.pe[pos, i+1] = np.cos(pos/(10000**(i/emb_dim)))
        self.dropout = nn.Dropout(pos_dropout) # dropout을 해야하는가 ->네
    def forward(self, x):
        pos_out = x + self.pe[:,:x.size(1)]
        return self.dropout(pos_out)

# max_seq_len = 400
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
    def __init__(self, d_k, dropout_attn=0.1):
        super().__init__() # 자기 parent class인 nn.Module의 함수()를 쓰겠다는 것/ super()=nn.Module 이라는 뜻
        self.sqrt_dk = d_k ** 0.5
        self.dropout = nn.Dropout(dropout_attn) # for regularization -> 노드를 랜덤으로 없애는거가 아니구 매트릭스 값을 랜덤으로 없앰
        self.softmax = nn.Softmax(dim=-1) # 왜 dim이 -1인지 보기

    def forward(self,Q,K,V, mask = None):
        score_qk = torch.matmul(Q,K.transpose()) / self.sqrt_dk
        if mask is not None:
            attn = score_qk.masked_fill(mask, -1e9)
        attn_prob = self.dropout(self.sotfmax(attn)) # mask 필요
        context = nn.matmul(attn_prob,V)
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 512, n_head = 8, dropout_attn = 0.1, dropout_multi = 0.1):
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

        q = self.linear_q(Q).view(batch_size, -1, n_head, d_k)
        k = self.linear_k(K).view(batch_size, -1, n_head, d_k)
        v = self.linear_v(V).view(batch_size, -1, n_head, d_v)

        context, attn_prob = self.attention(q,k,v, mask) # size 변경해야함
        context = context.view(batch_size, -1, self.embed_dim)
        context = self.final_linear(context)
        total_context = self.dropout(context)
        return total_context, attn_prob

class PositionalFeedForward(nn.Module):
    def __init__(self, embed_dim= 512, d_ff= 2048, dropout_ff= 0.1):
        super().__init__()
        self.weight_1 = nn.Linear(embed_dim, d_ff, bias=True)
        self.weight_2 = nn.Linear(d_ff, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout_ff)

    def forward(self, x):
        residual = x
        x = F.relu(self.weight_1(x))
        x = self.weight_2(x)
        x = self.dropout(x)
        return x + residual
