"""
Attention mechanisms for point cloud processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for point clouds
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, 
                                   query: torch.Tensor, 
                                   key: torch.Tensor, 
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention
        
        Args:
            query: Query tensor (B, num_heads, N, d_k)
            key: Key tensor (B, num_heads, N, d_k)
            value: Value tensor (B, num_heads, N, d_k)
            mask: Attention mask
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor (B, N, d_model)
            key: Key tensor (B, N, d_model)
            value: Value tensor (B, N, d_model)
            mask: Attention mask
            
        Returns:
            output: Multi-head attention output (B, N, d_model)
            attention_weights: Attention weights
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attn_output)
        
        return output, attention_weights


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with feed-forward network
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(SelfAttentionBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, N, d_model)
            mask: Attention mask
            
        Returns:
            output: Self-attention block output (B, N, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PointTransformer(nn.Module):
    """
    Point Transformer for point cloud processing
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super(PointTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (based on 3D coordinates)
        self.pos_encoder = PositionalEncoding3D(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (B, N, input_dim)
            pos: 3D positions (B, N, 3)
            
        Returns:
            output: Transformed features (B, N, d_model)
        """
        # Project input features
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder(pos)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        return x


class PositionalEncoding3D(nn.Module):
    """
    3D positional encoding for point clouds
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super(PositionalEncoding3D, self).__init__()
        
        self.d_model = d_model
        
        # Create encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
        # 3D coordinate embedding
        self.coord_embed = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pos: 3D coordinates (B, N, 3)
            
        Returns:
            pos_encoding: Positional encoding (B, N, d_model)
        """
        # Embed 3D coordinates
        pos_encoding = self.coord_embed(pos)
        
        return pos_encoding


class LocalAttention(nn.Module):
    """
    Local attention mechanism for point clouds
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        k_neighbors: Number of nearest neighbors to consider
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, k_neighbors: int = 16):
        super(LocalAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        
    def get_knn_indices(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Get k-nearest neighbor indices
        
        Args:
            pos: Point positions (B, N, 3)
            
        Returns:
            knn_indices: KNN indices (B, N, k_neighbors)
        """
        B, N, _ = pos.shape
        
        # Compute pairwise distances
        pos_expanded_1 = pos.unsqueeze(2).expand(B, N, N, 3)
        pos_expanded_2 = pos.unsqueeze(1).expand(B, N, N, 3)
        distances = torch.sum((pos_expanded_1 - pos_expanded_2) ** 2, dim=3)
        
        # Get k nearest neighbors
        _, knn_indices = torch.topk(distances, self.k_neighbors, dim=2, largest=False)
        
        return knn_indices
    
    def forward(self, features: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Point features (B, N, d_model)
            pos: Point positions (B, N, 3)
            
        Returns:
            output: Locally attended features (B, N, d_model)
        """
        B, N, C = features.shape
        
        # Get k-nearest neighbors
        knn_indices = self.get_knn_indices(pos)  # (B, N, k_neighbors)
        
        # Gather neighbor features
        knn_indices_expanded = knn_indices.unsqueeze(-1).expand(B, N, self.k_neighbors, C)
        neighbor_features = torch.gather(features.unsqueeze(2).expand(B, N, N, C), 
                                       2, knn_indices_expanded)  # (B, N, k_neighbors, C)
        
        # Reshape for attention computation
        neighbor_features = neighbor_features.view(B * N, self.k_neighbors, C)
        query_features = features.view(B * N, 1, C)
        
        # Apply attention
        attended_features, _ = self.attention(query_features, neighbor_features, neighbor_features)
        
        # Reshape back
        output = attended_features.view(B, N, C)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross attention between different point sets
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super(CrossAttention, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                source_features: torch.Tensor, 
                target_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            source_features: Source point features (B, N1, d_model)
            target_features: Target point features (B, N2, d_model)
            
        Returns:
            output: Cross-attended features (B, N1, d_model)
        """
        attended_features, _ = self.attention(source_features, target_features, target_features)
        output = self.norm(source_features + attended_features)
        
        return output