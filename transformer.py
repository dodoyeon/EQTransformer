import torch
import torch.nn as nn
from Layer import Encoder, Decoder
# from mask import create_padding_mask, create_attn_decoder_mask

class Transformer(nn.Module): # encoder와 decoder의 emb_dim등은 다르려나?
    def __init__(self, src_voca_size, trg_voca_size, emb_dim=512, max_seq_len=200, pos_dropout = 0.1,dropout_attn=0.1,
                 dropout_multi = 0.1, d_ff= 2048, dropout_ff= 0.1, n_layers = 8, n_head = 8, pad_idx = 0, layernorm_epsilon = 1e-12):
        super().__init__()
        self.encoder = Encoder(src_voca_size, emb_dim, max_seq_len, pos_dropout, dropout_attn,
                               dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon)
        self.decoder = Decoder(trg_voca_size, emb_dim, max_seq_len, pos_dropout, dropout_attn,
                               dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon)
        self.out = nn.Linear(emb_dim, trg_voca_size)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input, dec_input):
        # print("encoding")
        enc_output, self_enc_attn_prob = self.encoder(enc_input)
        # print("decoding")
        dec_output, self_dec_attn_prob, dec_enc_attn_prob = self.decoder(enc_input, dec_input, enc_output) #JWP
        output = self.out(dec_output)
        # output = self.softmax(out)
    
        return output, self_enc_attn_prob, self_dec_attn_prob, dec_enc_attn_prob
