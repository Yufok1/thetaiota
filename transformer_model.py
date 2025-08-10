#!/usr/bin/env python3
"""
Minimal 1-layer Transformer implementation for Phase 1 self-reflective AI agent.
Designed for GTX 1060 3GB constraints with introspection capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional

class SimpleAttention(nn.Module):
    """Single-head self-attention mechanism."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.sqrt_d_model = math.sqrt(d_model)
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights returned for introspection.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attn_weights: Attention weights [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x)  # [B, L, D]
        k = self.w_k(x)  # [B, L, D]
        v = self.w_v(x)  # [B, L, D]
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_d_model  # [B, L, L]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, L, D]
        output = self.w_o(attn_output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))

class TransformerLayer(nn.Module):
    """Single transformer layer with layer normalization."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SimpleAttention(d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and attention weights for introspection.
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class MinimalTransformer(nn.Module):
    """
    Multi-layer transformer (150M parameters) optimized for GTX 1060 3GB.
    Includes introspection capabilities for self-reflection.
    Default: 12 layers × 1024d × 4096 ff
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 1024,
        d_ff: int = 4096,
        max_seq_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        n_layers: int = 12
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Multiple transformer layers for 150M parameters
        self.n_layers = n_layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Output head for classification
        self.output_head = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
        # For introspection
        self.last_attention_weights = None
        self.last_hidden_states = None
        
    def _init_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with introspection data collection.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Combined embedding with dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Multiple transformer layers
        # Optional activation checkpointing for memory savings
        use_ckpt = self.training and getattr(self, "use_checkpointing", False)
        all_attn_weights = []
        
        for layer in self.transformer_layers:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                x, attn_weights = checkpoint(lambda inp: layer(inp, mask), x, use_reentrant=False)
            else:
                x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        
        # Use last layer attention weights for introspection
        attn_weights = all_attn_weights[-1] if all_attn_weights else None
        
        # Store for introspection
        self.last_attention_weights = attn_weights.detach()
        self.last_hidden_states = x.detach()
        
        # Global average pooling for sequence-level classification
        if mask is not None:
            # Use mask for proper averaging
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Classification head
        logits = self.output_head(x)  # [batch_size, num_classes]
        
        return logits
    
    def get_introspection_data(self) -> Dict[str, torch.Tensor]:
        """
        Return introspection data from the last forward pass.
        Used by the meta-controller for self-reflection.
        """
        return {
            'attention_weights': self.last_attention_weights,
            'hidden_states': self.last_hidden_states,
            'attention_entropy': self._compute_attention_entropy(),
            'hidden_mean': self.last_hidden_states.mean() if self.last_hidden_states is not None else None,
            'hidden_std': self.last_hidden_states.std() if self.last_hidden_states is not None else None,
        }
    
    def _compute_attention_entropy(self) -> Optional[torch.Tensor]:
        """Compute entropy of attention weights (measure of uncertainty)."""
        if self.last_attention_weights is None:
            return None
        
        # Compute entropy for each attention head
        # H(p) = -sum(p * log(p))
        attn = self.last_attention_weights
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1)  # [batch_size, seq_len]
        return entropy.mean()  # Average entropy
    
    def get_parameter_stats(self) -> Dict[str, float]:
        """Get statistics about model parameters for introspection."""
        stats = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[f"{name}_mean"] = param.data.mean().item()
                stats[f"{name}_std"] = param.data.std().item()
                if param.grad is not None:
                    stats[f"{name}_grad_norm"] = param.grad.norm().item()
        
        return stats
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def test_minimal_transformer():
    """Test the minimal transformer implementation."""
    print("Testing MinimalTransformer...")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration for GTX 1060 3GB
    model = MinimalTransformer(
        vocab_size=100,  # Small vocab for testing
        d_model=64,      # Reduced dimension
        d_ff=256,        # Reduced FF dimension  
        max_seq_len=32,  # Shorter sequences
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        
        # Test introspection
        introspection_data = model.get_introspection_data()
        print("Introspection data keys:", list(introspection_data.keys()))
        
        if introspection_data['attention_entropy'] is not None:
            print(f"Attention entropy: {introspection_data['attention_entropy'].item():.4f}")
        
        param_stats = model.get_parameter_stats()
        print(f"Parameter stats (sample): {list(param_stats.keys())[:5]}")
    
    # Test memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
        print(f"GPU memory used: {memory_used:.3f} GB")
        
        if memory_used > 2.8:  # Our limit
            print("WARNING: Exceeding 2.8GB memory limit!")
        else:
            print("Memory usage within limits")
    
    print("MinimalTransformer test completed!")

if __name__ == "__main__":
    test_minimal_transformer()