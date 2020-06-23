import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *
import math

from scraper import get_wiki_para

def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r'http\S+', ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def embeddings(datafile, tokenizer, token_length):
    hidden_features = []
    targets = []
    input_ids = []
    with open(datafile,'r') as f:
        lines = f.readlines()
    
    for ii in range(len(lines)):
        wiki_para = get_wiki_para(lines[ii])
        text = preprocess_text(wiki_para)
        tokens = tokenizer.tokenize(text)
        input_ids.append(tokenizer.encode(tokens, add_special_tokens=True, max_length=token_length, pad_to_max_length=True))
        
        if (cnt < 3):
            print(input_ids[-1])

        targets.append(1)
        cnt += 1

    print('converted all wiki paragraphs to input ids!')    
    return input_ids, targets


class MyMapDataset(Dataset):
    def __init__(self, dataset_type, datafile, tokenizer, token_length, DEVICE):
        
        input_ids, targets = get_lm_token_ids(datafile, tokenizer, token_length)
        
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        targets = torch.from_numpy(np.array(targets)).long().to(DEVICE)

        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.targets[idx])
