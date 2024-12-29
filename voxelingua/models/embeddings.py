from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements rotary positional embeddings with enhanced features
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.feature_dim
        self.max_seq_len = config.max_sequence_length
        
        # Initialize frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))
        
        # Position-wise feed-forward
        self.pos_ff = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.LayerNorm(self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len if seq_len is not None else x.shape[1]
        
        # Generate position indices
        position_ids = torch.arange(seq_len, device=x.device).float()
        
        # Compute sinusoidal frequencies
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotary transformation
        cos = emb.cos()
        sin = emb.sin()
        
        # Enhanced position features
        pos_features = self.pos_ff(torch.cat([cos, sin], dim=-1))
        
        # Scale and combine
        return x * (1 + self.scale * pos_features)

class AdaptiveEmbedding(nn.Module):
    """
    Adaptive embedding layer with dynamic feature selection
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multiple embedding tables at different scales
        self.embedding_scales = [config.feature_dim // (2 ** i) for i in range(3)]
        self.embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size, dim)
            for dim in self.embedding_scales
        ])
        
        # Scale mixing network
        self.scale_mixer = nn.Sequential(
            nn.Linear(sum(self.embedding_scales), config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU()
        )
        
        # Dynamic feature selector
        self.feature_selector = nn.Sequential(
            nn.Linear(config.feature_dim, len(self.embedding_scales)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get embeddings at different scales
        scale_embeddings = [
            embedding(input_ids)
            for embedding in self.embeddings
        ]
        
        # Concatenate all scales
        concat_embeddings = torch.cat(scale_embeddings, dim=-1)
        
        # Mix scales
        mixed_embeddings = self.scale_mixer(concat_embeddings)
        
        # Compute feature importance
        feature_weights = self.feature_selector(mixed_embeddings)
        
        # Weight different scales
        weighted_embeddings = sum(
            emb * weight.unsqueeze(-1)
            for emb, weight in zip(scale_embeddings, feature_weights.unbind(-1))
        )
        
        return {
            'embeddings': weighted_embeddings,
            'scale_weights': feature_weights,
            'scale_embeddings': scale_embeddings
        }
