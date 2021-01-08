from exist_Trmodel import Trmodel
import time
from torchtext.data.utils import get_tokenizer
import torchtext
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Dataset import *
from Attention import PositionalEncoding

# TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
#                             init_token='<sos>',
#                             eos_token='<eos>',
#                             lower=True)
# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
# TEXT.build_vocab(train_txt)

# def batchify(data, bsz):
#     data = TEXT.numericalize([data.examples[0].text])
#     # 데이터셋을 bsz 파트들로 나눕니다.
#     nbatch = data.size(0) // bsz
#     # 깔끔하게 나누어 떨어지지 않는 추가적인 부분(나머지들) 은 잘라냅니다.
#     data = data.narrow(0, 0, nbatch * bsz)
#     # 데이터에 대하여 bsz 배치들로 동등하게 나눕니다.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)

# batch_size = 20
# eval_batch_size = 10
# train_data = batchify(train_txt, batch_size)
# val_data = batchify(val_txt, eval_batch_size)
# test_data = batchify(test_txt, eval_batch_size)

# bptt = 35
# def get_batch(source, i):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

# ntokens = len(TEXT.vocab.stoi)  # 단어 사전의 크기
# emsize = 200  # 임베딩 차원
# nhid = 200  # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
# nlayers = 2  # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
# nhead = 2  # 멀티헤드 어텐션(multi-head attention) 모델의 헤드 개수
# dropout = 0.2  # 드랍아웃(dropout) 값

class TransformerModel(nn.Module):
    def __init__(self, src_voca_size, trg_voca_size, d_model=512, max_seq_len=200, pos_dropout=0.1, dropout_attn=0.1,
                 dropout_multi=0.1, d_ff=2048, dropout_ff=0.1, n_layers=8, nhead=8, pad_idx=0, layernorm_epsilon=1e-12):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Embedding(num_embeddings=src_voca_size, embedding_dim=d_model, padding_idx=1)
        self.output_embedding = nn.Embedding(num_embeddings=trg_voca_size, embedding_dim=d_model, padding_idx=1)
        self.input_pe = PositionalEncoding(d_model)
        self.output_pe = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.ff = nn.Linear(d_model, trg_voca_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input, dec_input):
        # print("encoding")
        src = self.input_embedding(enc_input)
        src = self.input_pe(src)
        trg = self.output_embedding(dec_input)
        trg = self.output_pe(trg)
        # print("decoding")
        output = self.transformer(src, trg)
        output = self.ff(output)
        output = self.softmax(output)

        return output

def train(model, train_iterator, optimizer, criterion, epochs):
    model.train()
    for p in model.parameters():
        if p.dim() > 1:  # why?
            nn.init.xavier_uniform_(p)
    pad_idx = 1
    total_loss = 0
    start_time = time.time()
    train_len = len(train_iterator)  # 227

    for epoch in range(epochs):
        for i, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg

            trg_input = trg.clone().detach()
            # if trg_input.data = 3 ->convert 1(<pad>)
            trg_input[trg_input == 3] = 1
            # trg_input = trg_input[:, :-1]
            trg_input = trg_input[:-1, :]
            batch_size = trg.size(0)
            # trg_input = torch.empty((trg.size(0),trg.size(1)-1))
            # for j in range(trg.size(0)): # row
            #     trg_row = trg[j,:]
            #     for k in range(trg.size(1)): # col
            #         if trg_row[k] == 3: # <eos>=3
            #             trg_k = torch.cat((trg_row[:k],trg_row[k+1:]))
            #             trg_input.cat(trg_k, dim=1)

            # ys = trg[:, 1:].contiguous().view(-1)
            ys = trg[1:, :].contiguous().reshape(-1)
            # view():torch, same data but different shape/ -1:make 1 size row

            # Masking

            # enc_pad_mask = create_padding_mask(src, src, pad_idx).to(device)
            # self_pad_mask = create_padding_mask(trg_input, trg_input, pad_idx).to(device)
            # self_attn_mask = create_attn_decoder_mask(trg_input).to(device)
            # self_dec_mask = torch.gt((self_pad_mask + self_attn_mask),0).to(device) # 첫번째 인풋에 broadcastable한 두번째 아규먼트사용, input>2nd이면 true
            # enc_dec_pad_mask = create_padding_mask(trg_input, src, pad_idx).to(device)

            optimizer.zero_grad()
            pred = model(src, trg_input)

            # ys_one = torch.zeros_like(pred.reshape(-1, pred.size(-1)), dtype=torch.int64)
            # ys_one[torch.arange(ys.shape[0]), ys] = 1

            loss = criterion(pred.reshape(-1, pred.size(-1)), ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            interval = 10

            if i % interval == 0 and i > 0:
                p = int(100*(i+1)/train_len)
                avg_loss = total_loss / interval
                ppl = math.exp(avg_loss)

                print("time= %dm: epoch %d iter %d [%s%s]  %d%%  loss = %.3f | ppl = %.3f" %
                    ((time.time() - start_time) // 60, epoch + 1, i + 1, "".join('#' * (p // 5)),
                    "".join(' ' * (20 - (p // 5))),
                    p, avg_loss, ppl), end='\r')
                total_loss = 0

        torch.save({'epoch': epoch,
                    'model state_dict': model.state_dict(),
                    'optimizer state_dict': optimizer.state_dict(),
                    'loss': total_loss}, 'weight/train_weight.pth')  # new file'train_weight.pth' if not exists

        # print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f | ppl = %.03f" % (
        #     (time.time() - start_time) // 60, epoch + 1, "".join('#' *
        #                                                          (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss,
        #     epoch + 1, total_loss/10, math.exp(total_loss/10)))


def evaluate(model, test_iterator, max_seq_len):
    checkpoint = torch.load('weight/train_weight.pth')
    model.load_state_dict(checkpoint['model state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer state_dict'])#->optimizer aren't used in test?

    model.eval()
    # total_loss = 0
    # test_len=8(-> the number of batch?/not change) ->?? but it seems 32(batch_size changed)
    test_len = len(test_iterator)
    # start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            total_loss = 0

            src = batch.src.transpose(0, 1)  # (128,12) -> batch_size = 128
            trg = batch.trg.transpose(0, 1)  # (128,11)

            # (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            trg_mask = torch.ones(
                trg.size(0), 201 - trg.size(1)).long().to(device)
            trg_test = torch.cat((trg, trg_mask), dim=1)
            ys = trg_test[:, 1:].contiguous().view(-1)  # size:(128x10=1280)
            # target_mask = nn.ConstantPad2d((0,200),1) # (maybe)PAD=1..
            dec_input = 2 * torch.ones(src.size(0),
                                       1).long().to(device)  # (batch=32,1)
            # test_input = torch.ones(src.size(0), max_seq_len) # (1,11)
            # test_input[:,0] = 2 # START_TOKEN = 2 / (maybe)PAD=1..

            # for j in range(src.size(0)):
            for k in range(max_seq_len):
                # https://github.com/eagle705/pytorch-transformer-chatbot/blob/master/inference.py 참조
                # we have to shape pred(32,1,512)->but no..
                pred, _, _, _ = model(src, dec_input)
                prediction = pred[:, -1, :].unsqueeze(1)
                # dim=-1(=512): [1]=index (32,1)->argmax?
                pred_ids = prediction.max(dim=-1)[1]
                # if (pred_ids[i,-1] == 3 for i in pred.size(0)).to(torch.device('cpu')).numpy():# why cpu? vocab.END_TOKEN=3
                #     # decoding_from_result(enc_input=enc_input, pred=pred, tokenizer=tokenizer)
                #     break

                # dec_input.unsqueeze(1),/pred_ids[0,-1].unsqueeze(0).unsqueeze(0)
                dec_input = torch.cat([dec_input, pred_ids.long()], dim=1)
                # if i == max_seq_len-1:
                #     # decoding_from_result(enc_input= enc_input, pred=pred, tokenizer=tokenizer)
                #     break
            # print("%d th batch is over"%(i))
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), ys)
            total_loss += loss.item()
            avg_loss = total_loss / trg.size(0)
            ppl = math.exp(avg_loss)
            print("loss = %.3f  perplexity = %.3f" % (avg_loss, ppl))
    # avg_loss = total_loss / test_len
    # ppl = math.exp(avg_loss)
    #
    # print("loss = %.3f  perplexity = %.3f" %(avg_loss, ppl))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_src_tokens = len(SRC.vocab.stoi)  # vocabulary dictionary size
    n_trg_tokens = len(TRG.vocab.stoi)
    epochs = 18
    emb_size = 512  # emb_dim
    n_hid = 200  # encoder의 positional ff층 차원수
    n_layers = 2  # transformer encoder decoder layer 개수
    n_head = 8  # multi-head attention head 개수
    d_model = 512
    max_seq_len = 200
    lr = 0.0001

    # model = Transformer(src_voca_size=n_src_tokens, trg_voca_size=n_trg_tokens, emb_dim=emb_size, d_ff=n_hid,
    #                     n_layers=n_layers, n_head=n_head).to(device)
    # model = nn.Sequential(
    #     nn.Embedding(num_embeddings=n_src_tokens, embedding_dim=d_model, padding_idx=1),
    #     nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=n_layers)
    # ).to(device)
    # model = TransformerModel(n_src_tokens, n_trg_tokens, d_model=d_model, nhead=n_head).to(device)
    model = Trmodel(src_vocab=SRC.vocab, trg_vocab=TRG.vocab).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=1)  # optimizer,loss->to(device) x
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

    # print("model's state_dict:")  # 모델의 학습가능한 parameter는 model.parameters()로 접근한다
    # # state_dict: 각 레이어를 파라미터 tensor와 매핑하는 PYTHON dict objects/ cnn, linear 등이나 registered buffer(batchnorm)등 저장
    # # optimizer도 state와 hyper parameter의 state_dict 가짐
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #
    # print("optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    # train
    print("start training..")
    train(model, train_iterator, optimizer, criterion, epochs)
    # test
    # print("start testing..")
    # evaluate(model, test_iterator, max_seq_len)


if __name__ == '__main__':
    main()
