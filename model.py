# models.py

import torch
import torch.nn as nn

#Setting the device 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
print('Device set to {0}'.format(device))

'''
This architecture influenced by the following tutorial: https://www.youtube.com/watch?v=EoGUlvhRYpk&t=663s
The changes applied in seq2seq model others encoder and decoder are straightforward as in all examples
Minor changes are applied to the model to not include <bos> token in the decoder input.
'''

SRC_VOCAB_SIZE = 1000  # source vocabulary size
TRG_VOCAB_SIZE = 1000  # target vocabulary size
EMB_DIM = 256          # embedding dimension
HID_DIM = 512          # hidden dimension of LSTM
N_LAYERS = 4           # number of LSTM layers
DROPOUT = 0.5          # dropout rate
MAX_LENGTH = 25

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True,bidirectional=False)
        
        
    def forward(self, x):
        
        embedded = self.dropout(self.embedding(x))
        output, (hidden,cell) = self.rnn(embedded)

        return  (hidden,cell)


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden,cell):
        #print('----Decoder----')

        x = x.unsqueeze(1) #Shape is changed to [batch_size, 1, embedding_size]
        #print("Input shape:", x.shape)

        embedded = self.dropout(self.embedding(x))
        #print("Embedded shape:", embedded.shape)


        output, (hidden,cell) = self.rnn(embedded, (hidden,cell))
        output = output.squeeze(1)
        #print("Output shape from LSTM:", output.shape)

        predictions = self.fc(output)

        return predictions, hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target=None, teacher_forcing_ratio=0.5):
        #print(f"Source shape: {source.shape}, Target shape: {target.shape}")
        batch_size = target.shape[0] # This line is changed
        target_len = target.shape[1] if target is not None else MAX_LENGTH
        target_vocab_size = self.decoder.fc.out_features
        #print(f"Target Vocabulary Size: {target_vocab_size}")
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        hidden,cell = self.encoder(source)

        #<sos> token is not used to replaced to target[:,0]
        decoder_input = target[:,0]         #torch.zeros(batch_size, dtype=torch.long).to(self.device) maybe i should include it check tomorrow


        for t in range(1, target_len):
            decoder_output, hidden,cell = self.decoder(decoder_input, hidden,cell)

            outputs[:,t,:] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if (teacher_force and target is not None) else top1
            #print(f"Next decoder input shape: {decoder_input.shape}")
        
        return outputs


def main():


    encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)


    # Create a batch of source and target sentences
    batch_size = 16
    src_len = 10  # length of source sentences
    trg_len = 12  # length of target sentences

    src = torch.randint(0, SRC_VOCAB_SIZE, (batch_size, src_len)).to(device)  # source sequences
    trg = torch.randint(0, TRG_VOCAB_SIZE, (batch_size, trg_len)).to(device)  # target sequences

    model = Seq2Seq(encoder, decoder, device).to(device)

    output = model(src, trg, teacher_forcing_ratio=0.5)
    print("Output shape:", output.shape)  # Expected shape: [batch_size, trg_len, TRG_VOCAB_SIZE]

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, TRG_VOCAB_SIZE), trg.view(-1))
    print("Output shape:", output.shape)
    print("Mock loss:", loss.item())
    


if __name__ == '__main__':
    main()