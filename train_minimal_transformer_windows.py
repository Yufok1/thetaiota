#!/usr/bin/env python3
"""
Enhanced MinimalTransformer Training for Agent Learning
=======================================================

Trains the MinimalTransformer model used by Phase 1/2/3 agents for:
- Task learning and decision making
- Pattern recognition and classification
- Self-improvement and adaptation
- Meta-learning capabilities

This model is separate from the conversational LM and focuses on
agent cognition and learning rather than text generation.

Usage:
  python train_minimal_transformer_windows.py --epochs 50 --lr 1e-3 --batch 32
"""

import os
import glob
import argparse
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Import our custom components
from transformer_model import MinimalTransformer
from toy_dataset import create_dataloaders
from memory_db import MemoryDB
from init_database import create_database


def log_memory_usage():
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   VRAM: {allocated:.2f}GB (Peak: {reserved:.2f}GB)")


def load_training_data(patterns: List[str] = None) -> Tuple[Any, Any, int]:
    """
    Load training data for agent learning tasks.
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader  
        num_classes: Number of classification classes
    """
    if patterns is None:
        patterns = ['*.md', '*.txt', '*.py']
    
    print(f"\n[DATA] Loading Agent Training Data:")
    
    # Create synthetic training data for agent learning tasks
    # This simulates the kinds of patterns agents need to learn
    train_size = 8000
    val_size = 2000
    batch_size = 32
    
    print(f"   Training Size: {train_size:,} samples")
    print(f"   Validation Size: {val_size:,} samples")
    print(f"   Batch Size: {batch_size}")
    print(f"   Data Type: Synthetic agent learning patterns")
    
    # Create dataloaders with curriculum learning
    train_loader, val_loader = create_dataloaders(
        dataset_type='sequence',
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size
    )
    
    # Determine number of classes from the dataset
    num_classes = 2  # Binary classification for agent decisions
    
    print(f"   Classes: {num_classes} (binary decision making)")
    
    return train_loader, val_loader, num_classes


def log_rich_training_step(epoch: int, step: int, total_steps: int, 
                          loss: float, avg_loss: float, model: MinimalTransformer, 
                          args: argparse.Namespace, val_metrics: Dict[str, float] = None):
    """Rich logging with detailed metrics and progress visualization."""
    
    # Progress bar
    progress = (step + 1) / total_steps
    bar_length = 20
    filled = int(bar_length * progress)
    bar = '#' * filled + '-' * (bar_length - filled)
    
    # Loss trend analysis
    trend = "STABLE"
    if val_metrics:
        if val_metrics.get('val_loss', 1.0) < avg_loss * 0.95:
            trend = "IMPROVING"
        elif val_metrics.get('val_loss', 1.0) > avg_loss * 1.05:
            trend = "WORSENING"
    
    # Convergence assessment
    convergence = "GOOD"
    if avg_loss < 0.5:
        convergence = "EXCELLENT - Learning patterns!"
    elif avg_loss < 1.0:
        convergence = "GOOD - Getting coherent..."
    elif avg_loss < 2.0:
        convergence = "OK - Still learning..."
    else:
        convergence = "POOR - Needs attention"
    
    print(f"\n[AGENT-TRANSFORMER] Training Step")
    print(f"   Epoch {epoch+1}/{args.epochs} | Step {step+1}/{total_steps}")
    print(f"   Progress: [{bar}] {progress*100:.1f}%")
    print(f"   Current Loss: {loss:.4f}")
    print(f"   Average Loss: {avg_loss:.4f}")
    if val_metrics:
        print(f"   Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        print(f"   Val Accuracy: {val_metrics.get('val_accuracy', 0):.3f}")
    print(f"   Trend: {trend}")
    log_memory_usage()
    print(f"   Convergence: {convergence}")


def test_agent_capabilities(model: MinimalTransformer, device: str, epoch: int):
    """Test the agent's learning capabilities with sample inputs."""
    
    print(f"\n[AGENT CAPABILITIES] Testing Learning (Epoch {epoch+1}):")
    
    model.eval()
    with torch.no_grad():
        # Test pattern recognition
        test_inputs = [
            torch.randint(0, 10, (1, 8)).to(device),   # Random sequence
            torch.randint(0, 10, (1, 12)).to(device),  # Longer sequence
            torch.randint(0, 10, (1, 6)).to(device),   # Shorter sequence
        ]
        
        for i, test_input in enumerate(test_inputs):
            try:
                output = model(test_input)
                probs = torch.softmax(output, dim=1)
                confidence = probs.max().item()
                prediction = torch.argmax(output, dim=1).item()
                
                status = "GOOD" if confidence > 0.7 else "OK" if confidence > 0.5 else "POOR"
                print(f"   {i+1}. Pattern {test_input.shape[1]} tokens -> Class {prediction} (conf: {confidence:.3f}) [{status}]")
                
            except Exception as e:
                print(f"   {i+1}. Test failed: {e} [ERROR]")


def main():
    parser = argparse.ArgumentParser(description="Enhanced MinimalTransformer Training for Agent Learning")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--d_model', type=int, default=1280, help="Model dimension")
    parser.add_argument('--d_ff', type=int, default=5120, help="Feed-forward dimension")
    parser.add_argument('--n_layers', type=int, default=12, help="Number of layers")
    parser.add_argument('--vocab_size', type=int, default=100, help="Vocabulary size")
    parser.add_argument('--max_seq_len', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision")
    parser.add_argument('--test_every', type=int, default=10, help="Test capabilities every N epochs")
    args = parser.parse_args()
    
    print("==> Enhanced MinimalTransformer Training for Agent Learning")
    print("=" * 60)
    
    # Log initial setup
    print(f"Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Architecture: {args.d_model}d x {args.n_layers} layers")
    print(f"   Feed-forward: {args.d_ff}")
    print(f"   Vocabulary: {args.vocab_size}")
    print(f"   Sequence Length: {args.max_seq_len}")
    print(f"   AMP: {'YES' if args.use_amp else 'NO'}")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load training data
    train_loader, val_loader, num_classes = load_training_data()
    
    # Create model
    print(f"\n[MODEL] Creating MinimalTransformer:")
    model = MinimalTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
        dropout=0.1,
        n_layers=args.n_layers
    )
    model.to(args.device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"   Model Type: Agent Learning Transformer")
    print(f"   Purpose: Task learning, pattern recognition, decision making")
    
    # Log VRAM after model creation
    print(f"\n[MEMORY] VRAM After Model Creation:")
    log_memory_usage()
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Training state
    best_val_loss = float('inf')
    steps_since_improvement = 0
    
    print(f"\n" + "="*60)
    print(f"[TRAINING] STARTING - Watch Agent Intelligence Emerge!")
    print(f"="*60)
    
    # Training loop
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional AMP
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Rich logging every 25 steps
            if num_batches % 25 == 0:
                avg_loss = epoch_loss / num_batches
                
                # Validation metrics
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(args.device), val_y.to(args.device)
                        val_logits = model(val_x)
                        val_loss += criterion(val_logits, val_y).item()
                        val_pred = torch.argmax(val_logits, dim=1)
                        val_correct += (val_pred == val_y).sum().item()
                        val_total += val_y.size(0)
                
                val_metrics = {
                    'val_loss': val_loss / len(val_loader),
                    'val_accuracy': val_correct / val_total
                }
                
                model.train()
                
                log_rich_training_step(epoch, num_batches, len(train_loader), 
                                     loss.item(), avg_loss, model, args, val_metrics)
                
                # Test agent capabilities
                test_agent_capabilities(model, args.device, epoch)
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\n[EPOCH] Epoch {epoch+1} Complete:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Steps Since Improvement: {steps_since_improvement}")
        
        # Test capabilities periodically
        if (epoch + 1) % args.test_every == 0 or epoch == args.epochs - 1:
            test_agent_capabilities(model, args.device, epoch)
        
        # Save checkpoint
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            steps_since_improvement = 0
            
            save_path = os.path.join('checkpoints', 'minimal_transformer.pt')
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'config': vars(args)
            }, save_path)
            print(f"   Saved best model to {save_path}")
        else:
            steps_since_improvement += 1
    
    # Final save
    final_path = os.path.join('checkpoints', 'minimal_transformer_final.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': args.epochs,
        'final_loss': avg_epoch_loss,
        'config': vars(args)
    }, final_path)
    
    print(f"\n" + "="*60)
    print(f"[TRAINING] COMPLETE!")
    print(f"="*60)
    print(f"Final Loss: {avg_epoch_loss:.4f}")
    print(f"Best Loss: {best_val_loss:.4f}")
    print(f"Model Saved: {final_path}")
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"Purpose: Agent learning, pattern recognition, decision making")
    print(f"Ready for Phase 1/2/3 agent integration!")


if __name__ == '__main__':
    main()
