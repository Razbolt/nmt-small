import torch 
import torch.nn as nn

'''
code is taking from here: https://www.youtube.com/watch?v=U0s0f995w14
'''

class SelfAttention(nn.Module):
    def __init__(self,embedding_size,heads): # heads splitting the embedding size
        super(SelfAttention,self).__init__()
        self.emebedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads #integer division

        assert (self.head_dim * heads == embedding_size), "Check the embedding size and heads"

        self.values = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.keys(nn.Linear(self.head_dim,self.head_dim,bias = False))
        self.queries(nn.Linear(self.head_dim,self.head_dim,bias = False))
        self.fc_out = nn.Linear(self.heads*self.head_dim,embedding_size)

        def forward(values,keys,queries,mask):
            n = queries.shape[0] #  taking the batch size 
            len_value, len_keys, len_queries = values.shape[1], keys.shape[1], queries.shape[1] # sequence length of values, keys and queries

            # Reshape the values, keys and queries
            values = values.reshape(n,len_value,self.heads,self.head_dim)
            keys = keys.reshape(n,len_keys,self.heads,self.head_dim)
            queries = queries.reshape(n,len_queries,self.heads,self.head_dim)

            energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys]) # (n,heads,len_queries, len_keys)
            if mask != None:
                energy = energy.masked_fill(mask == 0,float("-1e20"))



