import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledTanh(nn.Module):
    """
    Applies a scaled tanh activation that maps outputs from (-1,1) to (0,1).

    Forward Input:
        x: torch.Tensor (any shape)
    """
    def forward(self, x):
        # Rescale tanh output from (-1,1) to (0,1)
        return 0.5 * (torch.tanh(x) + 1)
    
class ClampedActivation(nn.Module):
    """
    Clamps input tensor values to the range [0,1].

    Forward Input:
        x: torch.Tensor (any shape)
    """
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)
    
class ResidualEmbedding(nn.Module):
    """
    Applies a linear projection to stacked input features to form a residual embedding.

    Forward Input:
        x_stacked: torch.Tensor of shape (batch_size, 2*time_steps, N)
    """
    def __init__(self, time_steps):
        super(ResidualEmbedding, self).__init__()
        # Linear projection: input dimension 2*time_steps → output dimension time_steps.
        self.fc = nn.Linear(time_steps * 2, time_steps)

    def forward(self, x_stacked):
        # Permute dimensions so that linear projection is applied along the time dimension.
        return self.fc(x_stacked.permute(0, 2, 1)).permute(0, 2, 1)

class D_GCN(nn.Module):
    """
    Diffusion Graph Convolution layer that captures spatial dependencies via a diffusion process.

    Forward Inputs:
        X: Node features, shape (batch_size, num_nodes, input_size)
        A_q: Adjacency matrix (query), shape (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        A_h: Adjacency matrix (hidden), shape (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
    """
    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        # Total number of matrices to aggregate: 2*orders + 1.
        self.num_matrices = 2 * self.orders + 1
        # Learnable weight parameter.
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        # Learnable bias.
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters using uniform distribution.
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        # Unsqueeze x_ and concatenate along a new dimension.
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        batch_size, num_nodes, input_size = X.shape
        supports = [A_q, A_h]

        x0 = X  # Initial input.
        x = torch.unsqueeze(x0, 0)  # Shape: (1, batch_size, num_nodes, input_size)
        # Process each support (adjacency matrix).
        for support in supports:
            # Expand support to have a batch dimension if needed.
            if support.dim() == 2:
                support = support.unsqueeze(0).expand(batch_size, -1, -1)
            x1 = torch.bmm(support, x0)  # First diffusion order.
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                # Chebyshev recurrence for higher-order diffusion.
                x2 = 2 * torch.bmm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        # Reshape and aggregate diffusion outputs.
        x = torch.reshape(x, shape=[self.num_matrices, batch_size, num_nodes, input_size])
        x = x.permute(1, 2, 3, 0)  # New shape: (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size, num_nodes, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1) + self.bias

        # Apply activation function.
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class MultiHead_D_GCN(nn.Module):
    """
    Multi-head Diffusion Graph Convolution layer.

    Forward Inputs:
        X: Node features, shape (batch_size, num_nodes, input_size)
        A_q, A_h: Adjacency matrices as described in D_GCN.
    """
    def __init__(self, in_channels, out_channels, orders, heads=4, activation='relu'):
        super(MultiHead_D_GCN, self).__init__()
        self.heads = heads
        # Create multiple D_GCN layers (one per head).
        self.head_convs = nn.ModuleList([D_GCN(in_channels, out_channels, orders, activation) for _ in range(heads)])
        # Project concatenated head outputs to the desired output dimension.
        self.output_projection = nn.Linear(out_channels * heads, out_channels)

    def forward(self, X, A_q, A_h):
        # Compute outputs for each head.
        outputs = [head(X, A_q, A_h) for head in self.head_convs]
        concatenated_output = torch.cat(outputs, dim=-1)  # Concatenate along feature dimension.
        return self.output_projection(concatenated_output)
    
class EdgeAwareSpatialAttention(nn.Module):
    """
    Edge-Aware Spatial Attention layer that updates node features and the adjacency matrix.
    
    Forward Inputs:
        X: Node features, shape (batch_size, num_nodes, embed_dim)
        A: Adjacency matrix, shape (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
    """
    def __init__(self, embed_dim, num_heads, adj_proj_dim, dropout=0.1):
        super(EdgeAwareSpatialAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.adj_proj_dim = adj_proj_dim
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Node feature projections.
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Project adjacency values to a higher-dimensional space.
        self.adj_proj_A = nn.Linear(1, self.head_dim * num_heads)

        # Output projections.
        self.node_out_proj = nn.Linear(embed_dim, embed_dim)
        self.adj_out_proj = nn.Linear(embed_dim, 1)

        self.scaling = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        batch_size, num_nodes, embed_dim = X.size()
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute adjacency embeddings.
        A_mean = A.mean(dim=-1, keepdim=True)  # Shape: (batch_size, num_nodes, 1)
        A_embedded = self.adj_proj_A(A_mean)
        A_embedded = A_embedded.view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute key, value, and query projections.
        K = self.key_proj(X).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value_proj(X).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Q = self.query_proj(X).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention scores.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # Modulate attention scores using the adjacency embedding.
        A_modulator = A_embedded.mean(dim=-1, keepdim=True).expand(-1, -1, num_nodes, num_nodes)
        attn_scores = attn_scores * A_modulator

        # Normalize attention scores.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Aggregate node features.
        node_attn_output = torch.matmul(attn_probs, V)
        node_attn_output = node_attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, embed_dim)
        updated_node_features = self.node_out_proj(node_attn_output)

        # Update adjacency matrix using an Einstein summation.
        adj_update = torch.einsum('bij,bjd->bijd', A, X)
        adj_update = self.adj_out_proj(adj_update).squeeze(-1)
        return updated_node_features, adj_update

class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention module over time steps.

    Forward Input:
        x: Tensor of shape (batch_size, num_nodes, time_steps)
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        # Convolution layers for key, query, and value for each head.
        self.key_convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1) for _ in range(num_heads)])
        self.query_convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1) for _ in range(num_heads)])
        self.value_convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1) for _ in range(num_heads)])
        self.output_projection = nn.Linear(hidden_dim * num_heads, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Transpose x to shape: (batch_size, time_steps, num_nodes)
        x_transposed = x.transpose(1, 2)
        head_outputs = []
        for i in range(self.num_heads):
            key = self.key_convs[i](x_transposed)
            query = self.query_convs[i](x_transposed)
            value = self.value_convs[i](x_transposed)
            # Transpose back to (batch_size, num_nodes, hidden_dim)
            key = key.transpose(1, 2)
            query = query.transpose(1, 2)
            value = value.transpose(1, 2)
            attention_scores = torch.bmm(query, key.transpose(1, 2))
            attention_weights = F.softmax(attention_scores / (self.hidden_dim ** 0.5), dim=-1)
            weighted_value = torch.bmm(attention_weights, value)
            head_outputs.append(weighted_value)
        concatenated_output = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated_output)
        return output

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer module combining multi-head temporal attention and a feed-forward network.

    Forward Input:
        X: Tensor of shape (batch_size, num_nodes, embed_dim)
    """
    def __init__(self, embed_dim, hidden_dim=64, num_heads=4, ff_hidden_dim=128, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.temporal_attention = TemporalAttention(input_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        temporal_output = self.temporal_attention(X)
        X = self.norm1(X + self.dropout(temporal_output))
        ff_output = self.feedforward(X)
        X = self.norm2(X + self.dropout(ff_output))
        return X

class SIGFormerBlock(nn.Module):
    """
    A single block of the SIGFormer model, integrating:
        - Multi-head Diffusion Graph Convolution (MultiHead_D_GCN)
        - Edge-Aware Spatial Attention
        - Temporal Transformer (with feed-forward network)
        - Residual connections and normalization
    
    Forward Inputs:
        X: Node features, shape (batch_size, num_nodes, in_channels)
        A_q: Adjacency matrix (query) with shape (batch_size, num_nodes, num_nodes)
        A_h: Adjacency matrix (hidden) with shape (batch_size, num_nodes, num_nodes)
    """
    def __init__(self, in_channels, out_channels, orders, heads=4, heads_temp=4, adj_proj_dim=64, ff_hidden_dim=128, dropout=0.1):
        super(SIGFormerBlock, self).__init__()
        self.multi_head_dgcn = MultiHead_D_GCN(in_channels, out_channels, orders, heads=heads)
        self.edge_aware_spatial_attention = EdgeAwareSpatialAttention(embed_dim=out_channels, num_heads=heads, adj_proj_dim=adj_proj_dim, dropout=dropout)
        self.temporal_transformer = TemporalTransformer(out_channels, hidden_dim=out_channels, num_heads=heads_temp, ff_hidden_dim=ff_hidden_dim, dropout=dropout)
        self.residual_projection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

        self.feedforward_node = nn.Sequential(
            nn.Linear(out_channels, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, out_channels),
        )
        self.feedforward_adj = nn.Sequential(
            nn.Linear(1, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 1),
        )
        self.node_norm1 = nn.LayerNorm(out_channels)
        self.node_norm2 = nn.LayerNorm(out_channels)
        self.node_norm3 = nn.LayerNorm(out_channels)
        self.node_norm4 = nn.LayerNorm(out_channels)
        self.adj_norm1 = nn.BatchNorm2d(1)
        self.adj_norm2 = nn.BatchNorm2d(1)

    def forward(self, X, A_q, A_h):
        # Apply multi-head DGCN.
        dgcn_out = self.multi_head_dgcn(X, A_q, A_h)
        # If input and output dimensions differ, project input.
        if self.residual_projection is not None:
            X = self.residual_projection(X)
        X = self.node_norm1(X + dgcn_out)

        # Apply edge-aware spatial attention.
        updated_nodes, updated_adjacency = self.edge_aware_spatial_attention(X, A_q)
        X = self.node_norm2(X + updated_nodes)

        # Apply temporal transformer.
        temporal_out = self.temporal_transformer(X)
        X = self.node_norm3(X + temporal_out)

        # Additional feed-forward update for nodes.
        updated_nodes = self.feedforward_node(X)
        X = self.node_norm4(X + updated_nodes)
        
        # Update adjacency matrix with residual connections and normalization.
        A_q = A_q + updated_adjacency
        A_q = self.adj_norm1(A_q.unsqueeze(1)).squeeze(1)
        updated_adjacency = self.feedforward_adj(A_q.unsqueeze(-1)).squeeze(-1)
        A_q = A_q + updated_adjacency
        A_q = self.adj_norm2(A_q.unsqueeze(1)).squeeze(1)
        return X, A_q

class SIGFormer(nn.Module):
    """
    Transformer Diffusion GNN (SIGFormer) that stacks multiple SIGFormerBlocks.
    
    Forward Inputs:
        X: Input features, shape (batch_size, time_steps, num_nodes)
        A_q: Adjacency matrix (query) with shape (batch_size, num_nodes, num_nodes)
        A_h: Adjacency matrix (hidden) with shape (batch_size, num_nodes, num_nodes)
    """
    def __init__(self, h, z, k, L=2, heads=4, heads_temp=4, ff_hidden_dim=128, adj_proj_dim=64, dropout=0.1):
        super(SIGFormer, self).__init__()
        self.time_dimension = h

        # First block stack.
        self.GNN1 = nn.ModuleList([
            SIGFormerBlock(h if i == 0 else z, z, k, heads=heads, heads_temp=heads_temp,
                            adj_proj_dim=adj_proj_dim, ff_hidden_dim=ff_hidden_dim, dropout=dropout)
            for i in range(L)
        ])

        # Second block stack with residual connections.
        self.GNN2 = nn.ModuleList([
            SIGFormerBlock(z, z, k, heads=heads, heads_temp=heads_temp,
                            adj_proj_dim=adj_proj_dim, ff_hidden_dim=ff_hidden_dim, dropout=dropout)
            for _ in range(L)
        ])

        # Final output layer stack.
        self.output_layer = nn.ModuleList([
            SIGFormerBlock(z, h if i == L - 1 else z, k, heads=heads, heads_temp=heads_temp,
                            adj_proj_dim=adj_proj_dim, ff_hidden_dim=ff_hidden_dim, dropout=dropout)
            for i in range(L)
        ])

        self.embedding = ResidualEmbedding(h)
        self.activation = nn.Sigmoid()  # Final activation function.
        # Alternatives:
        # self.activation = ScaledTanh()
        # self.activation = ClampedActivation()

    def forward(self, X, A_q, A_h):
        # Optionally, uncomment the following block to include a residual embedding.
        # x_mean = X.mean(dim=1, keepdim=True)
        # x_res = X - x_mean
        # x_stacked = torch.cat([X, x_res], dim=1)
        # X = self.embedding(x_stacked)

        # Transpose to shape (batch_size, num_nodes, time_steps) for processing.
        X = X.permute(0, 2, 1)
        for layer in self.GNN1:
            X, A_q = layer(X, A_q, A_h)
        residual_X = X
        residual_A_q = A_q
        for layer in self.GNN2:
            X, A_q = layer(X, A_q, A_h)
        X += residual_X
        A_q += residual_A_q
        for layer in self.output_layer:
            X, A_q = layer(X, A_q, A_h)
        X = self.activation(X)
        # Return output in shape: (batch_size, time_steps, num_nodes)
        return X.permute(0, 2, 1)

if __name__ == "__main__":
    # =======================
    # Test the SIGFormer model
    # =======================
    # Test parameters.
    batch_size = 2
    num_nodes = 150
    time_steps = 24
    embed_dim = 64
    adj_proj_dim = 32
    num_heads = 4
    num_heads_temp = 2
    diffusion_orders = 1
    hidden_dim = 128
    layers = 1

    # Generate random input data with shape: (batch_size, time_steps, num_nodes)
    X = torch.rand(batch_size, time_steps, num_nodes)
    A_q = torch.rand(batch_size, num_nodes, num_nodes)
    A_h = torch.rand(batch_size, num_nodes, num_nodes)

    # Initialize the SIGFormer model.
    model = SIGFormer(
        h=time_steps,
        z=embed_dim,
        k=diffusion_orders,
        L=layers,
        heads=num_heads,
        heads_temp=num_heads_temp,
        ff_hidden_dim=hidden_dim,
        adj_proj_dim=adj_proj_dim,
        dropout=0.1
    )

    # Forward pass.
    output = model(X, A_q, A_h)
    print("Input X shape:", X.shape)
    print("Output shape:", output.shape)
