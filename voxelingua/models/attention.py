from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class AdaptiveAttentionSpan(nn.Module):
    """
    Implements adaptive attention span mechanism with dynamic pruning
    """
    def __init__(self, config):
        super().__init__()
        self.max_span = config.max_sequence_length
        self.span_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 4),
            nn.LayerNorm(config.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 4, config.num_heads),
            nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.ones(1) * config.initial_temperature)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict attention spans
        spans = self.span_predictor(x)  # [batch_size, num_heads]
        spans = spans * self.max_span
        
        # Create span mask
        position_ids = torch.arange(self.max_span, device=x.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        spans = spans.unsqueeze(-1)  # [batch_size, num_heads, 1]
        
        # Apply soft masking with learnable temperature
        mask = torch.exp(-torch.relu(position_ids - spans) / self.temperature)
        return mask

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism with adaptive feature aggregation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.feature_dim // config.num_heads
        
        # Multi-scale projections
        self.scale_factors = [1, 2, 4]
        self.scale_projections = nn.ModuleList([
            nn.Linear(config.feature_dim, config.feature_dim)
            for _ in self.scale_factors
        ])
        
        # Scale attention
        self.scale_attention = nn.Parameter(torch.ones(len(self.scale_factors)) / len(self.scale_factors))
        
        # Output projection
        self.output_projection = nn.Linear(config.feature_dim, config.feature_dim)
        
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = q.shape
        
        # Process at multiple scales
        scale_outputs = []
        scale_attentions = []
        
        for i, scale_factor in enumerate(self.scale_factors):
            # Project inputs
            q_scaled = self.scale_projections[i](q)
            k_scaled = self.scale_projections[i](k)
            v_scaled = self.scale_projections[i](v)
            
            # Reshape for scaled attention
            if scale_factor > 1:
                # Aggregate sequence using stride convolution
                q_scaled = F.avg_pool1d(
                    q_scaled.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor
                ).transpose(1, 2)
                k_scaled = F.avg_pool1d(
                    k_scaled.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor
                ).transpose(1, 2)
                v_scaled = F.avg_pool1d(
                    v_scaled.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor
                ).transpose(1, 2)
            
            # Compute attention
            scaled_attention = torch.matmul(q_scaled, k_scaled.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scaled_mask = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=scaled_attention.size(-1),
                    mode='nearest'
                ).squeeze(1)
                scaled_attention = scaled_attention.masked_fill(~scaled_mask.bool(), float('-inf'))
            
            scaled_attention = F.softmax(scaled_attention, dim=-1)
            
            # Apply attention
            scale_output = torch.matmul(scaled_attention, v_scaled)
            
            # Upsample back to original size if needed
            if scale_factor > 1:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=seq_len,
                    mode='linear'
                ).transpose(1, 2)
            
            scale_outputs.append(scale_output)
            scale_attentions.append(scaled_attention)
        
        # Combine scales using learned weights
        scale_weights = F.softmax(self.scale_attention, dim=0)
        output = sum(out * w for out, w in zip(scale_outputs, scale_weights))
        
        # Final projection
        output = self.output_projection(output)
        
        return output, scale_attentions

class LocalGlobalAttention(nn.Module):
    """
    Combines local and global attention patterns
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_window = config.local_window_size
        
        # Local attention
        self.local_attention = nn.MultiheadAttention(
            config.feature_dim,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Global attention
        self.global_attention = nn.MultiheadAttention(
            config.feature_dim,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Mixing parameter
        self.mix_ratio = nn.Parameter(torch.tensor(0.5))
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Local attention with sliding window
        local_outputs = []
        local_attentions = []
        
        for i in range(0, seq_len, self.local_window):
            end_idx = min(i + self.local_window, seq_len)
            local_x = x[:, i:end_idx]
            local_mask = mask[:, i:end_idx] if mask is not None else None
            
            local_out, local_attn = self.local_attention(
                local_x, local_x, local_x,
                key_padding_mask=local_mask,
                need_weights=True
            )
            
            local_outputs.append(local_out)
            local_attentions.append(local_attn)
        
        local_output = torch.cat(local_outputs, dim=1)
        
        # Global attention
        global_output, global_attention = self.global_attention(
            x, x, x,
            key_padding_mask=mask,
            need_weights=True
        )
        
        # Combine local and global attention
        mix_ratio = torch.sigmoid(self.mix_ratio)
        output = mix_ratio * local_output + (1 - mix_ratio) * global_output
        
        return output, {
            'local_attention': local_attentions,
            'global_attention': global_attention,
            'mix_ratio': mix_ratio
