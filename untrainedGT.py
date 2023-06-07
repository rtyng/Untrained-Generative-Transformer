# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import time 


class PositionalEncoding(nn.Module):
    """
    Adds information about the absolute or relative positions of tokens in a sequence
    
    Transformer models do not have an inherent knowledge of token positions like an RNN, so we need to provide that explicitly
    Args:
        nn (Pytorch .nn module): Activates our positional encoding class 
    """
    def __init__(self, d_model, max_len=1000):
        """
        Initialization of parent class 
        
        Args:
            d_model (int): model dimension or # of features in input (typically called the embedding size)
            max_len (int, optional): max length of sequence that model can handle. Defaults to 1000.
        """
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.positional_encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):
        """
        Args:
            d_model (int): embedding size or # of input features
            max_len (int): max length of sequence model handles

        Returns:
            encoding : a tensor representing positional information
        """
        # initialize an encoding tensor, a position tnesor, and a scaling tensor
        
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions to generate positional encoding values for scaling the positional encoding
    
        encoding[:, 0::2] = torch.sin(position * div_term) 
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape and Transpose encoding to match expected shape for positional encodings
        
        encoding = encoding.unsqueeze(0).transpose(0, 1)
        return encoding

    def forward(self, x):
        """
        Takes input tensor x and applies positional encoding 

        Args:
            x (tensor): input tensor

        Returns:
            self.dropout(x) : randomly "zeroed-out" (element-wise) modified input tensor x
        """
        x = x + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        initialization

        Args:
            d_model (int): # of input features
            num_heads (int): # of attention heads
        """
        
        # initialize parent class, attributes
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Define Linear layers for query, key, value tensors
        # they are used to project input into desired dimensons for q,k, and v
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Scaling attention scores DURING computation to avoid vanishing or exploding gradients
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        """
        Forward pass through the self-attention mechanism

        Args:
            x (tensor): input tensor

        Returns:
            attn_output: multi-head self-attention outputs
        """
        
        batch_size = x.size(0)

        # Project inputs to query, key, and value using previously defined Linear Layers
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape tensors to enable multi-head attention
        q = q.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)
        k = k.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)
        v = v.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(1, 2))
        scores = scores * self.scale

        # Apply softmax to obtain attention probabilities
        attn_probs = F.softmax(scores, dim=-1)

        # Apply attention weights to the value vectors
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and concatenate multi-head outputs
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        

        Args:
            d_model (int): # of input features
            d_ff (int): size of feed-forward layer
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerBlock, self).__init__()
        self.self_attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.self_attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super(Transformer, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x

#start timer here
start_time = time.time()

# Generate random values for batch_size, seq_length, and d_model
batch_size = 9
seq_length = 25
d_model = 256

# Create an instance of your Transformer model
t_sne_visual = Transformer(d_model, num_heads=4, d_ff=256, num_layers=3)

# Generate a random input tensor
input_tensor = np.random.rand(batch_size, seq_length, d_model)

# Convert the input tensor to a PyTorch tensor
input_tensor = torch.Tensor(input_tensor)

# Forward pass through the Transformer model
output_tensor = t_sne_visual(input_tensor)

# Initialize a list to store the intermediate activations
activations = []

# Define the hooks to capture the activations
def hook(module, input, output):
    activations.append(output.detach().numpy())

# Register the hook for each transformer block
for transformer_block in t_sne_visual.transformer_blocks:
    transformer_block.register_forward_hook(hook)

# Forward pass through the Transformer model again to capture the activations
output_tensor = t_sne_visual(input_tensor)
_ = output_tensor  # Optional, to discard the output if not needed for further calculations

# Concatenate the activations along the sequence length axis
activations = np.concatenate(activations, axis=1)  # Assuming activations are of shape (batch_size, seq_length, d_model)

# Reshape the activations to match the desired shape
reshaped_activations = activations.reshape(-1, d_model)  # Reshape to (batch_size * seq_length, d_model)

# Flatten the activations
flattened_activations = activations.reshape(-1, activations.shape[-1])

# Apply t-SNE to obtain the low-dimensional embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings = tsne.fit_transform(flattened_activations)

# Visualize the t-SNE embeddings
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.title("t-SNE Visualization of Transformer Model Activations")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

#stop timer
end_time = time.time()
plt.show()

elapsed_time = end_time - start_time
print("Time of model input to output and to graphing:\n", elapsed_time)