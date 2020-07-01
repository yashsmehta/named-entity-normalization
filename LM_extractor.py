import numpy as np
import pandas as pd
import csv
import pickle
import re
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *

# This file passes the 'description' paragraph of the passed item through a language model (used ALBERT) to get a 768 dim vector representation of the 
# paragraph. The point of this is so that we can create a heirarchial category graph by evaluating how 'similar' (e.g. cosine similarity) the vectors of 2 
# descriptions are. I make use of the HuggingFace implementation of ALBERT.

start = time.time()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print('GPU found (', torch.cuda.get_device_name(torch.cuda.current_device()), ')')
    torch.cuda.set_device(torch.cuda.current_device())
    print('num device avail: ', torch.cuda.device_count())

else:
    DEVICE = torch.device('cpu')
    print('running on cpu')

# using ALBERT-small model
n_hl = 12
hidden_dim = 768
datafile, max_token_length, batch_size, op_dir = utils.parse_args_extractor()

entity_class = 'countries'
entity_class_id = 1

model_class, tokenizer_class, pretrained_weights = AlbertModel, AlbertTokenizer, 'albert-base-v2'

model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)  # output_attentions=False
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

map_dataset = MyMapDataset(datafile, tokenizer, max_token_length, DEVICE, entity_class_id)

#class to load information in the form of a pyTorch dataset; 
data_loader = DataLoader(dataset=map_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         )

if (DEVICE == torch.device("cuda")):
    model = model.cuda()
    # model.parameters() returns a generator obj
    print('\ngpu mem alloc: ', round(torch.cuda.memory_allocated() * 1e-9, 2), ' GB')

print('starting to extract LM embeddings...')

# for the initial training phase, we need to have 'targets', which can be either manually assigned or synthetically created. 
hidden_features = []
all_targets = []

for input_ids, targets in data_loader:
    with torch.no_grad():
        all_targets.append(targets.cpu().numpy())
        tmp = []
        bert_output = model(input_ids)
    
        for ii in range(n_hl):
            tmp.append((bert_output[2][ii + 1].cpu().numpy()).mean(axis=1))
            
        hidden_features.append(np.array(tmp))
    
# Save the LM embeddings in a file to save time/computation. So that for every training session, we are not going online to parse information and are saving the 
# description paragraph embedding in a file.

file = open(op_dir + '-' + entity_class + 'embedings.pkl', 'wb')
pickle.dump(zip(hidden_features, all_targets), file)
file.close()

print(timedelta(seconds=int(time.time() - start)), end=' ')
print('embeddings extraction for {} - DONE!'.format(entity_class))                