import re
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, file, text_vocab, max_length=1024, pad_token="<PAD>", unk_token="<UNK>", label2index=None):
        self.text, self.label = self.load(file)

        assert len(self.text) == len(self.label), print("text: {}, label: {}".format(len(self.text), len(self.label)))

        if label2index is None:
            self.label2index = dict(zip(sorted(set(self.label)), range(len(set(self.label)))))
        else:
            self.label2index = label2index
        self.convert_label2index()
        assert len(self.text) == len(self.label), print("text: {}, label: {}".format(len(self.text), len(self.label)))

        self.text_vocab = text_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_length = max_length
        self.text = self.word2index(self.text)
        assert len(self.text) == len(self.label), print("text: {}, label: {}".format(len(self.text), len(self.label)))

        self.pad()
        assert len(self.text) == len(self.label), print("text: {}, label: {}".format(len(self.text), len(self.label)))
    
    def convert_label2index(self):
        self.label = [self.label2index[_label] for _label in self.label]

    def word2index(self, text):
        #############################
        _text = []
        for i in text:
            temp = []
            for j in i:
                temp.append(self.text_vocab.get(j, self.text_vocab[self.unk_token]))
            _text.append(temp)
        ###########################
        return _text
    
    def load(self, file):
        text, label = [], []
        f = open(file, "r+", encoding="utf-8")
        for i in f.readlines():
            #####################
            content = i.strip().split("\t")
            _label = content[0].strip()
            _text = re.sub("\s+", "", content[1])
            label.append(_label)
            temp = []
            for char in _text:
                temp.append(char)
            text.append(temp)
            #####################
        return text, label
    
    def pad(self):
        pad_text = []
        for _text in self.text:
            ################
            pad_text.append((_text + [self.text_vocab[self.pad_token]] * max(0, self.max_length - len(_text)))[:self.max_length])
            ################
        self.text = torch.tensor(pad_text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.label[item]