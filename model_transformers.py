import torch 
import torch.nn as nn
import torch.optim as optim
'''
code is inspired from here : https://www.youtube.com/watch?v=U0s0f995w14
'''

class SelfAttention(nn.Module):
    def __init__(self,embedding_size,heads): # heads splitting the embedding size
        super(SelfAttention,self).__init__()
        self.emebedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads #integer division

        assert (self.head_dim * heads == embedding_size), "Check the embedding size and heads"

        self.values = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.keys = (nn.Linear(self.head_dim,self.head_dim,bias = False))
        self.queries = (nn.Linear(self.head_dim,self.head_dim,bias = False))
        self.fc_out = nn.Linear(self.heads*self.head_dim,embedding_size)
        
    def forward(self,values,keys,queries,mask):
        n = queries.shape[0] #  taking the batch size 
        len_value, len_keys, len_queries = values.shape[1], keys.shape[1], queries.shape[1] # sequence length of values, keys and queries
        
        # Reshape the values, keys and queries
        values = values.reshape(n,len_value,self.heads,self.head_dim)
        keys = keys.reshape(n,len_keys,self.heads,self.head_dim)
        queries = queries.reshape(n,len_queries,self.heads,self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys]) # (n,heads,len_queries, len_keys)
                                                                    #energy represent the attention score between each pair of tokens in the input  
        #shape of query is (n,len_queries,heads,head_dim)
        # shape of keys is (n,len_keys,heads,head_dim)
        # shape of energy is (n,heads,len_queries,len_keys) 
        scale_factor = self.head_dim ** 0.5
        if mask != None:
            energy = energy.masked_fill(mask == 0,float("-1e19")
                                            )
        
        attention = torch.softmax(energy / (scale_factor),dim = 3) # (n,heads,len_queries,len_keys)
                                                                                # dim 3 takes the len_keys and apply softmax on it
        #attention shape is (n,heads,len_queries,len_keys)
        #value shape is (n,len_value,heads,head_dim)
        output = torch.einsum("nhql, nlhd->nqhd",[attention,values])
        #after einsum output shape is (n,heads,len_queries,head_dim) # Not sure yet !
        output = output.reshape(n, len_queries, self.heads* self.head_dim)
        output = self.fc_out(output)

        return output
            
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embedding_size,heads)
        self.norm1 = nn.LayerNorm(embedding_size) # similar to batch normalization but it normalizes in every layer
        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential( 
            nn.Linear(embedding_size, forward_expansion*embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embedding_size,embedding_size)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query,mask):
        attention = self.attention(value,key,query,mask)

        x = self.dropout(self.norm1(attention + query)) # query is input to norm1 and used as residual connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # x is input to norm2 and used as residual connection
        return out 

class Encoder(nn.Module):
    def __init__(self,src_vocab_size,embedding_size, num_layers, heads,device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embedding_size)
        self.position_embedding = nn.Embedding(max_length,embedding_size)

        self.layers = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out,out,out,mask)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, forward_expansion,dropout, device):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embedding_size,heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(embedding_size,heads,dropout,forward_expansion) 
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,value,key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention +x))
        out   = self.transformer_block(value,key,query,src_mask)
        return out 
    

class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embedding_size,num_layers,heads,device,forward_expansion,dropout,max_length):
        super(Decoder,self).__init__()
        self.words_embedding = nn.Embedding(trg_vocab_size,embedding_size)
        self.position_embedding = nn.Embedding(max_length,embedding_size)
        self.device = device
        self.layers = nn.ModuleList([DecoderBlock(embedding_size,heads,forward_expansion,dropout,device) for _ in range(num_layers)])

        self.fc_out= nn.Linear(embedding_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x,enc_out,src_mask,trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout(self.words_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x= layer(x,enc_out,enc_out,src_mask,trg_mask)

        out = self.fc_out(x)
        return out 


class Transformers(nn.Module):
    def __init__(self, src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,embedding_size = 256,num_layers= 6,forward_expansion = 4,
                 heads= 8, dropout = 0, device = "cpu", max_length = 40):
        super(Transformers,self).__init__()

        self.encoder = Encoder(src_vocab_size,embedding_size,num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size,embedding_size,num_layers,heads,device,forward_expansion,dropout,max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask shape is (N,1,1,src_len)
        src_mask.to(self.device)
        return src_mask
    
    def make_trg_mask(self,trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)

        return trg_mask.to(self.device)

    def forward(self,src,trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg,enc_src,src_mask,trg_mask)
        
        return out
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    x = torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
    trg = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)
    print(x.shape,'---',trg.shape)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformers(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,device = device).to(device)

    output = model(x,trg[:,:-1])
    print('Initial Output:',output.shape)
    argmax = output.argmax(-1)
    print('Output after argmax:',argmax.shape)
    print('Sample prediction:',argmax[0])


    output = output.reshape(-1, output.shape[2])
    labels = trg[:, 1:].reshape(-1)

    print('Output shape after reshape:',output.shape)
    print('Labels shape after reshape:',labels.shape)
    