import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel, BertConfig

class ModelOutput:
  def __init__(self, logits, labels, loss=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss

class BertNer(nn.Module):
  def __init__(self, args):
    super(BertNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0]  # [batchsize, max_len, 768]
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output
