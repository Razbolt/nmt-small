import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
'''
code is inspired from here : https://www.youtube.com/watch?v=U0s0f995w14
'''

#CHANGES
# 1) Bias included in SelfAttention
# 2) FeedForward in TransformersBlock is changed to GLU (Gated Linear Unit)

class SelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_per_head = emb_size // num_heads  # Dimension per attention head

        # Ensure embedding size is divisible by number of heads
        assert (self.dim_per_head * num_heads == emb_size), "Embedding size must be divisible by number of heads"

        # Initialize linear transformations for each component
        self.value_proj = nn.Linear(self.dim_per_head, self.dim_per_head, bias=True)
        self.key_proj = nn.Linear(self.dim_per_head, self.dim_per_head, bias=True)
        self.query_proj = nn.Linear(self.dim_per_head, self.dim_per_head, bias=True)
        self.final_proj = nn.Linear(num_heads * self.dim_per_head, emb_size)
        
    def forward(self, value_input, key_input, query_input, mask):
        batch_size = query_input.shape[0]  # Batch size
        value_len, key_len, query_len = value_input.shape[1], key_input.shape[1], query_input.shape[1]  # Sequence lengths

        # Reshape inputs for multi-head attention
        value_input = value_input.reshape(batch_size, value_len, self.num_heads, self.dim_per_head)
        key_input = key_input.reshape(batch_size, key_len, self.num_heads, self.dim_per_head)
        query_input = query_input.reshape(batch_size, query_len, self.num_heads, self.dim_per_head)

        # Apply linear transformations
        value_input = self.value_proj(value_input)
        key_input = self.key_proj(key_input)
        query_input = self.query_proj(query_input)

        # Calculate attention scores using einsum for efficient computation
        scores = torch.einsum("bqhd, bkhd -> bhqk", [query_input, key_input])

        # Scale the scores
        scale = self.dim_per_head ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e19"))

        # Softmax normalization on scores
        attention_weights = torch.softmax(scores / scale, dim=3)

        # Multiply attention weights by values
        weighted_values = torch.einsum("bhql, blhd -> bqhd", [attention_weights, value_input])
        
        # Reshape and project to the original embedding size
        weighted_values = weighted_values.reshape(batch_size, query_len, self.num_heads * self.dim_per_head)
        output = self.final_proj(weighted_values)

        return output

            
class AttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, drop_rate, exp_factor):
        super(AttentionBlock, self).__init__()
        self.self_attention = SelfAttention(emb_dim, num_heads)  # Use paraphrased AttentionModule from previous response
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        # Feed-forward network with GLU activation
        self.feedforward_net = nn.Sequential(
            nn.Linear(emb_dim, exp_factor * emb_dim * 2),
            nn.GLU(dim=-1),  # Using GLU activation for intermediate transformation
            nn.Linear(exp_factor * emb_dim, emb_dim)  # Reduce dimension back to original after expansion
        )
        self.dropout_layer = nn.Dropout(drop_rate)

    def forward(self, value_input, key_input, query_input, mask):
        # Apply self-attention
        attn_output = self.self_attention(value_input, key_input, query_input, mask)

        # Add and normalize, use dropout
        add_norm1 = self.dropout_layer(self.layer_norm1(attn_output + query_input))

        # Ensure input dimension to GLU is even
        if add_norm1.size(-1) % 2 != 0:
            add_norm1 = F.pad(add_norm1, (0, 0, 0, 1))
        
        # Pass through feed-forward network
        ff_output = self.feedforward_net(add_norm1)
        # Final add and normalize step with dropout
        final_output = self.dropout_layer(self.layer_norm2(ff_output + add_norm1))

        return final_output

class Encoder(nn.Module):
    def __init__(self,src_vocab_size,embedding_size, num_layers, heads,device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embedding_size)
        self.position_embedding = nn.Embedding(max_length,embedding_size)

        self.layers = nn.ModuleList([AttentionBlock(embedding_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
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
        self.device = device
        self.attention = SelfAttention(embedding_size,heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = AttentionBlock(embedding_size,heads,dropout,forward_expansion) 
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


class TransformerImproved(nn.Module):
    def __init__(self, src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,embedding_size = 300,num_layers= 6,exp_factor = 4,
                 heads= 6, dropout = 0.2, device = "cpu", max_length = 40):
        super(TransformerImproved,self).__init__()

        self.encoder = Encoder(src_vocab_size,embedding_size,num_layers, heads, device, exp_factor, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size,embedding_size,num_layers,heads,device,exp_factor,dropout,max_length)

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
    model = TransformerImproved(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,device = device).to(device)

    output = model(x,trg[:,:-1])
    print('Initial Output:',output.shape)
    argmax = output.argmax(-1)
    print('Output after argmax:',argmax.shape)
    print('Sample prediction:',argmax[0])


    output = output.reshape(-1, output.shape[2])
    labels = trg[:, 1:].reshape(-1)

    print('Output shape after reshape:',output.shape)
    print('Labels shape after reshape:',labels.shape)
    