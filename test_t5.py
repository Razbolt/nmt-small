from datasets import Dataset
import re
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import T5ForConditionalGeneration,MarianTokenizer, T5Tokenizer, AdamW
from utils import collate_fn2, set_seed
import sacrebleu


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device set to {device}')


def test_model(model, test_loader, tokenizer, device):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, _ = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model.generate(input_ids=input_ids, max_length=20, num_beams=2, early_stopping=True)
            predicted_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_sentences = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(predicted_sentences)
            references.extend(label_sentences)
    
    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU score: {bleu.score}")
    
    # Print some sample translations
    for pred, ref in zip(predictions[:10], references[:10]):
        print(f"Predicted: {pred}")
        print(f"Reference: {ref}")
        print("---")

    return predictions, references


if __name__ == '__main__':

    #tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sv')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load('models/t5-model.pth'))
    model.to(device)

    test_data = torch.load('test_data_t5.pth')
    test_loader = DataLoader(test_data, batch_size=16,collate_fn=collate_fn2, shuffle=False)
    predictions, references = test_model(model, test_loader, tokenizer, device)