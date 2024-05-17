from datasets import Dataset
import re
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW,MarianTokenizer
from utils import collate_fn2, set_seed

# Setting the device 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device set to {device}')

# Initialize T5 tokenizer and model
#tokenizer = T5Tokenizer.from_pretrained('t5-small')

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sv')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.resize_token_embeddings(len(tokenizer))  # Adjust the model's embeddings if tokenizer has new tokens
model.to(device)

def clean_text(text):
    text = text.lower()
    return text

def preprocess_function(examples):
    examples['src'] = [clean_text(text) for text in examples['src']]
    examples['tgt'] = [clean_text(text) for text in examples['tgt']]

    model_inputs = tokenizer(examples['src'], max_length=20, truncation=True, padding="max_length")
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['tgt'], max_length=20, truncation=True, padding="max_length")
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels, _ = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
            epoch_loss += loss.item()
            
            predicted_ids = logits.argmax(dim=-1)
            predicted_sentences = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            label_sentences = tokenizer.batch_decode(labels, skip_special_tokens=True)

            #for pred, true in zip(predicted_sentences, label_sentences):
            #    print("Predicted Sentence:", pred)
            #    print("True Sentence:", true)

    return epoch_loss / len(data_loader)

if __name__ == '__main__':
    # Load your data
    with open('english_50k_clean.txt', 'r') as f:
        english_sentences = [line.strip() for line in f.readlines()]

    with open('swedish_50k_clean.txt', 'r') as f:
        swedish_sentences = [line.strip() for line in f.readlines()]

    print(f"Number of sentences: {len(english_sentences), len(swedish_sentences)}")
    assert len(english_sentences) == len(swedish_sentences)

    data = {'src': english_sentences, 'tgt': swedish_sentences}
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Split the dataset
    train_size = int(0.8 * len(tokenized_dataset))
    val_size = int(0.1 * len(tokenized_dataset))
    test_size = len(tokenized_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(tokenized_dataset, [train_size, val_size, test_size])

    test_data = [tokenized_dataset[i] for i in test_dataset.indices]
    torch.save(test_data, 'test_data_t5.pth')# Save the extracted data

    # Set random seed for reproducibility
    set_seed(42)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn2, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=0.0005)

    number_of_epochs = 4

    # Training loop
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, labels, _ = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx} Loss: {loss.item()}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {average_loss}")

        validation_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch: {epoch+1}, Validation Loss: {validation_loss:.4f}')

    torch.save(model.state_dict(), 'models/t5-model.pth')
