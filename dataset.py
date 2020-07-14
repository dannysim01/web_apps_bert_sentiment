import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review                     # the review text, a list
        self.target = target                     # 0 or 1, a list
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):                           # returns the total length of data set
        return len(self.review)

    def __getitem__(self, item):                 # takes an 'item' and returns tokenizer of that item from data set
        review = str(self.review[item])          # converts everything to str incase there exists numbers etc.
        review = " ".join(review.split())        # removes all unnecessary space

        inputs = self.tokenizer.encode_plus(     # encode_plus can encode 2 strings at a time
            review,
            None,                                # since we use only 1 string at a time
            add_special_tokens=True,             # adds cld, sep tokens
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"] # since only 1 string token_type_ids are same and unnecessary

        padding_length = self.max_len - len(ids)  # for bert we pad on the right side
        ids = ids + ([0] * padding_length)        # zero times the padding length
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.target[item], dtype=torch.float)
        }
    """ if we have 2 target outputs then set to torch.long,
    depends on loss function also, from cross-entropy we should use torch.long"""



