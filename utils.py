from datasets import load_dataset, Dataset
#from transformers import MarianTokenizer
import sentencepiece as spm 
import re
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence # For dynamic padding
from torch.utils.data import random_split
import torch.nn as nn
import argparse
import yaml


tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sv')

def collate_fn2(batch): # This is a dynamic padding function
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor([1] * len(item['input_ids'])) for item in batch], batch_first=True, padding_value= 0)
    #Important ! 
    # 5643 is the id of the <pad> token in the tokenizer. So I am not it should be 56433 or 0
    return input_ids, labels, attention_mask

def collate_fn(batch): #In order to run this first change padding = 'max_length' in preprocess_function
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    return input_ids, labels, attention_mask

def set_seed(seed_value=42):
    return torch.manual_seed(seed_value)  # For reproducibility it works as random_seed in numpy
    
def decode_tokens(token_ids): # Decoder if you want to print the tokens not the numbers
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text

def init_weights(m):
    for name, param in m.named_parameters():
        return nn.init.uniform_(param.data, -0.08, 0.08)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()

def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings
