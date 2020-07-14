import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        """ 768: bert we use have 768 features | 1: binary classification
        if we use 2, we need to change the loss function"""

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        """ We have 2 outputs from a BERT model
         o1(last hidden state): is the sequence of hidden states. eg. if we have 512 tokens (MAX_LEN), 
         we have 512 vectors of size 768 for each batch. We can use out1 to max pooling or averge pooling
         o2(pooler output from bert pooler layer): we get vector of size 768 for each sample in batch"""
        bo = self.bert_drop(o2)                                 # drop-out
        output = self.out(bo)                                   # linear-layer
        return output
