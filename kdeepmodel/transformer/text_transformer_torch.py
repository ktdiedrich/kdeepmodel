#!/usr/bin/env python

"""Transformer sequence to sequence Pyttorch
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
with logging https://tensorboardx.readthedocs.io/en/latest/tutorial.html
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time
from tensorboardX import SummaryWriter
import os


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, nToken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self._model_type = 'Transformer'
        self._pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self._transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self._encoder = nn.Embedding(nToken, ninp)
        self._ninp = ninp
        self._decoder = nn.Linear(ninp, nToken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self._encoder.weight.data.uniform_(-initrange, initrange)
        self._decoder.bias.data.zero_()
        self._decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src_encoded = self._encoder(src)
        src = src_encoded * math.sqrt(self._ninp)
        src = self._pos_encoder(src)
        output = self._transformer_encoder(src, src_mask)
        output = self._decoder(output)
        return output


class TransformProcess:
    def __init__(self, params, model):
        self._device = params['device']
        self._bptt = params['bptt']
        self._ntokens = params['ntokens']
        self._params = params
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 1.0, gamma=0.95)
        self._best_model = None
        self._output_dir = params['output_dir']
        os.makedirs(self._output_dir, exist_ok=True)
        self._log_writer = SummaryWriter(os.path.join(self._output_dir, 'logs'))

    def data_process(self, raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
            for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, batches):
        nbatch = data.size(0) // batches
        data = data.narrow(0, 0, nbatch * batches)
        data = data.view(batches, -1).t().contiguous()
        return data.to(self._device)

    def get_batch(self, source, i):
        seq_len = min(self._bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def train_epochs(self, model, train_data, val_data):
        best_val_loss = float("inf")

        for epoch in range(1, params['epochs']+1):
            epoch_start_time = time.time()
            self.train(model, train_data, epoch)
            val_loss = self.evaluate(model, val_data)
            self._log_writer.add_scalar('val_loss', val_loss, epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._best_model = model

    def evaluate(self, eval_model, data_source, ):
        eval_model.eval()
        total_loss = 0.0
        src_mask = model.generate_square_subsequent_mask(self._bptt).to(self._device)
        with torch.no_grad():
            for i in range(0, data_source.size(0)-1, self._bptt):
                data, targets = self.get_batch(data_source, i)
                if data.size(0) != self._bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(self._device)
                output = eval_model(data, src_mask)
                output_flat = output.view(-1, self._ntokens)
                total_loss += len(data) * self._criterion(output_flat, targets).item()
        return total_loss/(len(data_source)-1)

    def train(self, model, train_data, epoch):

        model.train()
        total_loss = 0
        start_time = time.time()
        src_mask = model.generate_square_subsequent_mask(self._bptt).to(self._device)
        for batch, i in enumerate(range(0, train_data.size(0)-1, self._bptt)):
            data, targets = self.get_batch(train_data, i)
            self._optimizer.zero_grad()
            if data.size(0) != self._bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(self._device)
            output = model(data.to(self._device), src_mask)
            loss = self._criterion(output.view(-1, self._ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self._optimizer.step()
            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // self._bptt, self._scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
        self._log_writer.add_scalar('train_loss', cur_loss, epoch)
        print("fin train epoch")

    @property
    def best_model(self):
        return self._best_model


if __name__ == '__main__':
    import argparse
    params = {}
    params['batch_size'] = 20
    params['eval_batch_size'] = 10
    params['bptt'] = 35
      # the size of vocabulary
    params['emsize'] = 200  # embedding dimension
    params['nhid'] = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    params['nlayers'] = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    params['nhead'] = 2  # the number of heads in the multiheadattention models
    params['dropout'] = 0.2  # the dropout value
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['lr'] = 5.0
    params['epochs'] = 400
    params['output_dir'] = '/home/ktdiedrich/output/text_transformer_torch'

    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))
    params['ntokens'] = len(vocab.stoi)
    model = TransformerModel(params['ntokens'], params['emsize'], params['nhead'],
                             params['nhid'], params['nlayers'],
                             params['dropout']).to(params['device'])
    process = TransformProcess(params=params, model=model)
    train_data = process.data_process(iter(io.open(train_filepath, encoding='utf8')))
    val_data = process.data_process(iter(io.open(valid_filepath, encoding='utf8')))
    test_data = process.data_process(iter(io.open(test_filepath, encoding='utf8')))

    train_data = process.batchify(train_data, params['batch_size'])
    val_data = process.batchify(val_data, params['eval_batch_size'])
    test_data = process.batchify(test_data, params['eval_batch_size'])

    process.train_epochs(model, train_data, val_data)
    test_loss = process.evaluate(process.best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    print("fin")
