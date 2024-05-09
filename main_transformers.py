from datasets import load_dataset, Dataset
from transformers import MarianTokenizer
import sentencepiece as spm 
import re
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence # For dynamic padding
from torch.utils.data import random_split
import torch.nn as nn

from utils import collate_fn2, collate_fn, set_seed, decode_tokens, init_weights
from model import Encoder, Decoder, Seq2Seq
from model_transformers import Transformers

#Setting the device 
#TODO: Change the device to cuda whenever you set it to Hyperion mode !

device = torch.device('cpu' if torch.backends.mps.is_available() else 'cuda')
print('Device set to {0}'.format(device))

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sv')

num_token_id = tokenizer.convert_tokens_to_ids('<num>')
if num_token_id == tokenizer.unk_token_id:  # Checking if <num> is recognized , they should be equal id since above convertion makes <num> as unk
    tokenizer.add_tokens(['<num>'])
    #print("Added <num> with ID:", tokenizer.convert_tokens_to_ids('<num>'))

#Inlcude bos id 
if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    #print("Added <bos> with ID:", tokenizer.convert_tokens_to_ids('<bos>'))

#print('Showing the special tokens of the tokenizer:',tokenizer.special_tokens_map)
#print(f'eos_id:',tokenizer.eos_token_id,'bos_id:',tokenizer.bos_token_id,'pad_id:', tokenizer.pad_token_id, 'unk_id:',tokenizer.unk_token_id)
#print('Tokenizer vocab size:',tokenizer.vocab_size)

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



    model_inputs = tokenizer(examples['src'], max_length=20, truncation=True) # 
    # Setup the tokenizer to also prepare the target so that it can be used in the model
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['tgt'], max_length=20, truncation=True,) # padding = 'max_length'
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

def evaluate_model(model,data_loader,criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids,labels,_ = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(input_ids,labels[:,:-1])
            argmax = output.argmax(dim=-1)
            #print("Argmax:", argmax[0])

            predicted_sentences = tokenizer.batch_decode(argmax, skip_special_tokens=True)
            label_sentences = [tokenizer.batch_decode(labels[i, 1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in range(labels.size(0))]


            for pred, true in zip(predicted_sentences, label_sentences):
                print("Predicted Sentence:", pred)
                print("True Sentence:", true)
            


            output = output.reshape(-1, output.shape[2])
            labels = labels[:, 1:].reshape(-1)

            loss = criterion(output, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
    

if __name__ == '__main__':

    with open('english_50k_clean.txt', 'r') as f:
        english_sentences = [line for line in f.readlines() ]

    with open('swedish_50k_clean.txt', 'r') as f:
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

    #print(tokenized_dataset['src'][114], '---',tokenized_dataset['tgt'][114])
    #print(tokenized_dataset['input_ids'][114], '---',tokenized_dataset['labels'][114])


    set_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn2, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn2, shuffle=False)

    print('Train Loader:',len(train_loader))


    #encoder = Encoder(vocab_size,300,1024,2,0.5)
    #decoder = Decoder(vocab_size,300,1024,2,0.5)
    #model = Seq2Seq(encoder, decoder, device)
    #model.apply(init_weights)

    #TODO: 1) Convert these transformers parameters to config file
    #TODO 2) Change the parameters and try in different way. Try to understand the head concept
    #TODO 3) Some parameters are not used in the model or, some of them are the same such as vocab_size and pad_idx. Change it
    #TODO 4) Re-run the model with different learning rates, epochs and batch sizes

    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    forward_expansion = 4
    dropout = 0.4
    max_length = 20
    src_pad_idx = tokenizer.pad_token_id
    trg_pad_idx = tokenizer.pad_token_id

    model = Transformers(vocab_size,vocab_size,src_pad_idx,trg_pad_idx,embedding_size,num_encoder_layers,forward_expansion,num_heads,dropout,device,max_length)
    model.to(device)

    clip = 1.0
    number_of_epochs = 5
    learning_rate = 3e-4
    criterion = torch.nn.CrossEntropyLoss(ignore_index= tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epochs in range(number_of_epochs): # Number of epochs

        
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad() # Zero the gradient after each batch
            input_ids, labels, attention_mask = batch
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

            #print(input_ids.shape,'---' ,labels.shape)
            #forward propagation
            output = model(input_ids,labels[:,:-1])
            #print(f"Output shape: {output.shape}")
            output = output.reshape(-1, output.shape[2])
            labels = labels[:, 1:].reshape(-1)
            #print(f"After reshape Output shape: {output.shape}")
            #print(f"After reshape Labels shape: {labels.shape}")

            #print("Output device and format:", output.device, output.is_contiguous(memory_format=torch.channels_last))
            #print("Labels device and format:", labels.device)
            
            optimizer.zero_grad()
            output = output.contiguous()
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # Clip the gradients to prevent exploding gradients
            optimizer.step()

            #For seq2seq model 
            #output = model(input_ids, labels, teacher_forcing_ratio=0.5)
            #loss = criterion(output.view(-1, vocab_size), labels.view(-1))
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            #optimizer.step()

            
            total_loss += loss.item()
            print(f"Epoch {epochs+1}, Batch:{batch_idx} Loss: {loss.item()}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epochs+1} Average Loss: {average_loss}")
        validation_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch: {epochs+1}, Validation Loss: {validation_loss:.4f}')
        
       

        
    torch.save(model.state_dict(), 'models/transformers-model.pth')
    print("Output shape:", output.shape)  # Expected shape: [batch_size, trg_len, TRG_VOCAB_SIZE]

        
        
