from datasets import load_dataset, Dataset
from transformers import MarianTokenizer
import sentencepiece as spm 
import re
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence # For dynamic padding
from torch.utils.data import random_split
import torch.nn as nn

from utils import collate_fn2, collate_fn, set_seed, decode_tokens, init_weights, parse_arguments, read_settings
from model import Encoder, Decoder, Seq2Seq
from logger import Logger


#Setting the device 
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print('Device set to {0}'.format(device))

tokenizer = MarianTokenizer.from_pretrained('./opus-mt-en-sv')

num_token_id = tokenizer.convert_tokens_to_ids('<num>')
if num_token_id == tokenizer.unk_token_id:  # Checking if <num> is recognized , they should be equal id since above convertion makes <num> as unk
    tokenizer.add_tokens(['<num>'])
    #print("Added <num> with ID:", tokenizer.convert_tokens_to_ids('<num>'))

#Inlcude bos id 
if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    #print("Added <bos> with ID:", tokenizer.convert_tokens_to_ids('<bos>'))

print('Showing the special tokens of the tokenizer:',tokenizer.special_tokens_map)
print(f'eos_id:',tokenizer.eos_token_id,'bos_id:',tokenizer.bos_token_id,'pad_id:', tokenizer.pad_token_id, 'unk_id:',tokenizer.unk_token_id)


print('Tokenizer vocab size:',tokenizer.vocab_size)
vocab_size = len(tokenizer.get_vocab())
print("Updated tokenizer vocab size:", vocab_size) #This one should be used


def clean_text(text):
    #text = re.sub(r'\d+', '<num>', text) #Replacing all numbers with <num>
    #text = re.sub(r'[^\w\s]', '', text) #Remove punctuation
    #text = re.sub(r'\s+', ' ', text) #Remove extra spaces
    text = text.lower()
   
    return text

def preprocess_function(examples):

    examples['src'] = [clean_text(text) for text in examples['src']]
    examples['tgt'] = [clean_text(text) for text in examples['tgt']]

    examples['src'] = [tokenizer.bos_token + ' ' + text for text in examples['src']]
    examples['tgt'] = [tokenizer.bos_token + ' ' + text for text in examples['tgt']]



    model_inputs = tokenizer(examples['src'], max_length=20, truncation=True) # ,padding = 'max_length'
    # Setup the tokenizer to also prepare the target so that it can be used in the model
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['tgt'], max_length=20, truncation=True,) #,padding = 'max_length'
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

def evaluate_model(model,data_loader,criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids,labels,_ = batch
            src = input_ids.to(device)
            trg = labels.to(device)

            #src_sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            #print("Source sentences:", src_sentences)
            output = model(src, trg, 0) #turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #output_sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            #print("Output sentences:", output_sentences)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
    

def main():

    args = parse_arguments()

    # Read the settings from the YAML file
    settings = read_settings(args.config)

    paths = settings['sentence_paths']

    config = settings['model_settings']
    number_of_epochs = config['num_epochs']
    # Initialize logger
    logger_settings = settings['logger']
    experiment_name = logger_settings['experiment_name']
    project = logger_settings['project']
    entity = logger_settings['entity']
    

    my_logger = Logger(experiment_name, project, entity)
    #my_logger.login()
    my_logger.start(settings)

    with open(paths['file_en'], 'r') as f:
        english_sentences = [line for line in f.readlines() ]

    with open(paths['file_sv'], 'r') as f:
        swedish_sentences = [line for line in f.readlines() ]


    print(f"Number of sentences: {len(english_sentences), len(swedish_sentences)}")
    assert len(english_sentences) == len(swedish_sentences)
    print(f"Number of sentences: {len(english_sentences), len(swedish_sentences)}")
    


    data = {'src': english_sentences, 'tgt': swedish_sentences}
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    train_size = int(0.8 * len(tokenized_dataset))  # 80% for training
    val_size = int(0.1 * len(tokenized_dataset))  # 10% for validation
    test_size = len(tokenized_dataset) - train_size - val_size  # Remaining for testing

    # Split the dataset
    train_dataset,val_dataset, test_dataset = random_split(tokenized_dataset, [train_size,val_size, test_size])

    test_data = [tokenized_dataset[i] for i in test_dataset.indices]
    torch.save(test_data, 'test_data.pth')# Save the extracted data

    print(tokenized_dataset['src'][114], '---',tokenized_dataset['tgt'][114])
    print(tokenized_dataset['input_ids'][114], '---',tokenized_dataset['labels'][114])


    set_seed(42)
    train_loader = DataLoader(train_dataset, config['batch_size'], collate_fn=collate_fn2, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], collate_fn=collate_fn2, shuffle=False)

    test_loader = DataLoader(test_dataset, config['batch_size'], collate_fn=collate_fn2, shuffle=False)

    encoder = Encoder(vocab_size,300,1024,2,0.5)
    decoder = Decoder(vocab_size,300,1024,2,0.5)
    model = Seq2Seq(encoder, decoder, device)

    model.apply(init_weights)
    model.to(device)

    clip = 1.0
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index= tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epochs in range(number_of_epochs): # Number of epochs

        
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad() # Zero the gradient after each batch
            input_ids, labels, attention_mask = batch
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

            output = model(input_ids, labels, teacher_forcing_ratio=0.5)
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            
            total_loss += loss.item()
            print(f"Epoch {epochs+1}, Batch Loss: {loss.item()}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epochs+1} Average Loss: {average_loss}")
        validation_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch: {epochs+1}, Validation Loss: {validation_loss:.4f}')
        # Logging
        my_logger.log({
            'epoch': epochs +1,
            'train_loss': average_loss,
            'val_loss': validation_loss,
        })   

        
    torch.save(model.state_dict(), 'models/third-model.pth')
    print("Output shape:", output.shape)  # Expected shape: [batch_size, trg_len, TRG_VOCAB_SIZE]

if __name__ == '__main__':
    main()      
        
