from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import gluonnlp as nlp
from pytorch_pretrained_bert import BertModel, BertConfig
from torchcrf import CRF
from typing import List, Callable, Union, Dict
from utils import download as _download

import os
import sys
import requests
import hashlib
import json
import torch.nn.functional as F
import torch.nn as nn

        
def get_bert_multi_model(ctx="cuda"):

    with open('bert_config.json', mode='r') as io:
            bert_config = json.loads(io.read())
    #bertmodel = BertModel(config=BertConfig.from_dict(bert_config))
    bertmodel = BertModel.from_pretrained('bert-base-multilingual-cased')
    #bertmodel.load_state_dict(torch.load(model_path + 'bert_model.ckpt.data-00000-of-00001'))
    device = torch.device(ctx)
    bertmodel.to(device)
    #bertmodel.eval()
    return bertmodel


class Config:
    def __init__(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class PadSequence:
    """PadSequence class"""

    def __init__(self, length: int, pad_val: int = 0, clip: bool = True) -> None:
        """Instantiating PadSequence class
        Args:
            length (int): the maximum length to pad/clip the sequence
            pad_val (int): the pad value
            clip (bool): whether to clip the length, if sample length is longer than maximum length
        """
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[: self._length]
            else:
                return sample
        else:
            return sample + [self._pad_val for _ in range(self._length - sample_length)]
        
class BertMulti_CRF(nn.Module):
    """ KoBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(BertMulti_CRF, self).__init__()
        self.name = self.__class__.__name__
        
        self.bert = get_bert_multi_model()
        self.vocab = vocab
    
        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        #print(input_ids) batch_size*max_len 二维tensor
        #self.vocab.token_to_idx[self.vocab.padding_token]的值为1
        #print(attention_mask) 没有被padding的部分为1，padding的部分为0
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        #print(len(all_encoder_layers))
        #last_encoder_layer = all_encoder_layers[-1]
        last_encoder_layer = all_encoder_layers[-4] + all_encoder_layers[-3] + all_encoder_layers[-2] + all_encoder_layers[-1]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            #print(len(sequence_of_tags)==128) #True
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)  
            return sequence_of_tags
        
class KobertBiLSTMCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertBiLSTMCRF, self).__init__()
        self.name = self.__class__.__name__
        self.config = config
        
        self.bert = get_bert_multi_model()
        self.vocab = vocab
        
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]
        self.num_layers = 1
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.bilstm = nn.LSTM(self.hidden_size, self.hidden_size // 2, num_layers=1, dropout=config.dropout, batch_first=True, bidirectional=True)
        self.position_wise_ff = nn.Linear(self.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=True):

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        attention_mask = input_ids.ne(self._pad_id).float()
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = all_encoder_layers[-1]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            #self.hidden = self.init_hidden(self.config.batch_size)
            pack_padded_last_encoder_layer = pack_padded_sequence(last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, self.hidden = self.bilstm(pack_padded_last_encoder_layer, self.hidden)
            outputs = pad_packed_sequence(outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bilstm(last_encoder_layer)
        
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        #outputs=self.dropout(output_combine)
        emissions = self.position_wise_ff(outputs)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags
        
    def init_hidden(self, batchsize=100):

        hidden_a = torch.empty(self.num_layers*2, batchsize,  self.hidden_size // 2)
        hidden_a = nn.init.xavier_uniform_(hidden_a)
        hidden_b = torch.empty(self.num_layers*2, batchsize,  self.hidden_size // 2)
        hidden_b = nn.init.xavier_uniform_(hidden_b)
       
        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

class BertMulti_Only(nn.Module):
    """ koBERT alone """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(BertMulti_Only, self).__init__()
        self.name = self.__class__.__name__
        
        self.bert = get_bert_multi_model()
        self.vocab = vocab
        
        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #last_encoder_layer = all_encoder_layers[-1]
        last_encoder_layer = all_encoder_layers[-4] + all_encoder_layers[-3] + all_encoder_layers[-2] + all_encoder_layers[-1]
        last_encoder_layer = self.dropout(last_encoder_layer)
        tag_out = self.position_wise_ff(last_encoder_layer)

        return tag_out
    
class BiLSTM(nn.Module):
    
    def __init__(self, config, num_classes, vocab=None):
        super().__init__()
        self.name = self.__class__.__name__
        
        self.vocab = vocab
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]

        self.hidden_size = 500
        self.num_layers = 3
        self.embed_dim = 100
        self.rnn=torch.nn.LSTM(
            input_size= self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )  
        self.out=torch.nn.Linear(in_features=self.hidden_size,out_features=num_classes)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

    def forward(self, input_ids, token_type_ids=None, tags=None, evaluate_mode=False):
        
        if evaluate_mode == False:
            self.hidden = self.init_hidden(input_ids.size(0))
        else:
            self.hidden = self.init_hidden(1)
         
        embeds = self.word_embeddings(input_ids)       

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        #print(input_ids, seq_length, '\n')
        pack = pack_padded_sequence(embeds, seq_length, batch_first=True, enforce_sorted=False)
        #self.rnn.flatten_parameters()
        output, self.hidden = self.rnn(pack, self.hidden) #或者embeds.view(1, -1, cfg.embed_dim)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=self._pad_id)
        unpacked = unpacked[:, :, :self.hidden_size] + unpacked[:, : ,self.hidden_size:]
        
        output_dropout = F.dropout(unpacked)
        tag_out = self.out(output_dropout)

        return tag_out
    
    def init_hidden(self, batchsize=100):
        
        #hidden_a = torch.randn(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_a = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_a = nn.init.xavier_uniform_(hidden_a)
        
        #hidden_b = torch.randn(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_b = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_b = nn.init.xavier_uniform_(hidden_b)
       
        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

class BiLSTM_CRF(nn.Module):
    
    def __init__(self, config, num_classes):
        super().__init__()
        self.name = self.__class__.__name__
        
        _, self.vocab = get_kobert_model()
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]

        self.hidden_size = 500
        self.num_layers = 3
        self.embed_dim = 100
        self.rnn=torch.nn.LSTM(
            input_size= self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )  
        self.out=torch.nn.Linear(in_features=self.hidden_size,out_features=num_classes)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, evaluate_mode=False):
        
        if evaluate_mode == False:
            self.hidden = self.init_hidden(input_ids.size(0))
        else:
            self.hidden = self.init_hidden(1)
         
        embeds = self.word_embeddings(input_ids)       

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        #print(input_ids, seq_length, '\n')
        pack = pack_padded_sequence(embeds, seq_length, batch_first=True, enforce_sorted=False)
        #self.rnn.flatten_parameters()
        output, self.hidden = self.rnn(pack, self.hidden) #或者embeds.view(1, -1, cfg.embed_dim)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=self._pad_id)
        unpacked = unpacked[:, :, :self.hidden_size] + unpacked[:, : ,self.hidden_size:]
        
        output_dropout = F.dropout(unpacked)
        emissions = self.out(output_dropout)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags
    
    def init_hidden(self, batchsize=100):
        
        #hidden_a = torch.randn(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_a = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_a = nn.init.xavier_uniform_(hidden_a)
        
        #hidden_b = torch.randn(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_b = torch.empty(self.num_layers*2, batchsize,  self.hidden_size)
        hidden_b = nn.init.xavier_uniform_(hidden_b)
       
        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)