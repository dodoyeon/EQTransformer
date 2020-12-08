# import os
import time
import torch
import torch.nn as nn
from transformer import Transformer
from Dataset import *
# from process_Samlynn import batch_size_fn, read_data, create_dataset, create_fields

def train_model(model, train_data, epochs, criterion, optimizer, train_len, src_pad, trg_pad):
    print("training model...")
    model.train()
    start = time.time()
    # temp = start

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_data):
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            trg_input = trg[:, :-1]
            preds, _, _, _ = model(src, trg_input)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() # total_loss += loss.data[0]

            if (i + 1) % 100 == 0:
                p = int(100 * (i + 1) / train_len)
                avg_loss = total_loss / 100
                print("time= %dm: epoch %d iter %d [%s%s]  %d%%  loss = %.3f" %
                      ((time.time() - start) // 60, epoch + 1, i + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))),
                       p, avg_loss), end='\r')
                total_loss = 0
                # temp = time.time()
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss, epoch + 1, avg_loss))

# def test_model(model, test_data, epochs, criterion, optimizer, test_len, src_pad, trg_pad):
#     print("training model...")
#     model.eval()
#     start = time.time()
#     total_loss = 0
#     with torch.no_grad():
#         for
#             pred = model(test_data)
#             loss = criterion(pred, ys)
#             total_loss += loss.item()

def main():
    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyper-parameters
    epochs = 2
    emb_size = 512  # emb_dim
    n_hid = 200  # encoder의 positional ff층 차원수
    n_layers = 2  # transformer encoder decoder layer 개수
    n_head = 8  # multi-head attention head 개수
    d_model = 512
    max_seq_len = 200
    batch_size = 1500
    lr = 0.0001

    # # Loading Data on variables
    # src_data = os.path.abspath("data/Samlynn_data/english.txt")
    # trg_data = os.path.abspath("data/Samlynn_data/french.txt")
    # src_data, trg_data = read_data(src_data, trg_data)
    # SRC, TRG = create_fields(src_lang='en', trg_lang='fr')
    #
    # train_data, train_len, src_pad, trg_pad = create_dataset(src_data, trg_data, batch_size, device, max_seq_len, SRC,
    #                             TRG)  # = train_iter(in Process.py)
    # src_n_tokens = len(SRC.vocab)
    # trg_n_tokens = len(TRG.vocab)

    # Define model
    model = Transformer(src_voca_size=src_n_tokens, trg_voca_size=trg_n_tokens, emb_dim=emb_size, d_ff=n_hid,
                        n_layers=n_layers, n_head=n_head).to(device)

    # Define Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

    # get_model= (just define)return model contained src_vocab, trg_vocab, opt.d_model, opt.n_layers..

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # if opt.SGDR == True:  # in first, all 3 is not satisfied
    #     opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    # if opt.checkpoint > 0:
    #     print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    # if opt.load_weights is not None and opt.floyd is not None:
    #     os.mkdir('weights')
    #     pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
    #     pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    # Train model
    train_model(model, train_data, epochs, criterion, optimizer, train_len, src_pad, trg_pad)

if __name__ == '__main__':
    main()