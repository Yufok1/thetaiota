#!/usr/bin/env python3
"""
Generate toy dataset for Phase 1 self-reflective AI agent training.
Simple sequence classification task that can demonstrate learning patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class ToySequenceDataset(Dataset):
    """
    Simple sequence classification task:
    - Input: sequences of integers (0-9)
    - Task: classify if sequence sum is even (0) or odd (1)
    - This creates a learnable pattern that's not trivial
    """
    
    def __init__(self, size=10000, seq_len=8, vocab_size=10):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate random sequences
        self.sequences = []
        self.labels = []
        
        for _ in range(size):
            # Generate random sequence
            seq = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
            # Label: 1 if sum is odd, 0 if even
            label = sum(seq) % 2
            
            self.sequences.append(seq)
            self.labels.append(label)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class ToyMathDataset(Dataset):
    """
    Alternative: Simple arithmetic patterns
    - Input: [a, b, op] where op is 0=add, 1=sub, 2=mul
    - Output: result modulo 10 (single digit)
    """
    
    def __init__(self, size=10000):
        self.size = size
        self.data = []
        self.labels = []
        
        for _ in range(size):
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            op = random.randint(0, 2)  # 0=add, 1=sub, 2=mul
            
            if op == 0:
                result = (a + b) % 10
            elif op == 1:
                result = (a - b) % 10
            else:  # op == 2
                result = (a * b) % 10
            
            self.data.append([a, b, op])
            self.labels.append(result)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def create_dataloaders(dataset_type="sequence", train_size=8000, val_size=2000, batch_size=32):
    """Create train and validation dataloaders."""
    
    if dataset_type == "sequence":
        train_dataset = ToySequenceDataset(size=train_size)
        val_dataset = ToySequenceDataset(size=val_size)
    elif dataset_type == "math":
        train_dataset = ToyMathDataset(size=train_size)
        val_dataset = ToyMathDataset(size=val_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def test_dataset():
    """Test the dataset creation and show examples."""
    print("Testing Sequence Dataset:")
    seq_dataset = ToySequenceDataset(size=10)
    for i in range(5):
        seq, label = seq_dataset[i]
        print(f"  Sequence: {seq.tolist()}, Sum: {sum(seq.tolist())}, Label (odd): {label.item()}")
    
    print("\nTesting Math Dataset:")
    math_dataset = ToyMathDataset(size=10)
    ops = ["add", "sub", "mul"]
    for i in range(5):
        data, label = math_dataset[i]
        a, b, op = data.tolist()
        print(f"  {a} {ops[op]} {b} = {label.item()} (mod 10)")
    
    print("\nTesting DataLoaders:")
    train_loader, val_loader = create_dataloaders(dataset_type="sequence", batch_size=4)
    batch_x, batch_y = next(iter(train_loader))
    print(f"  Train batch shape: {batch_x.shape}, {batch_y.shape}")
    print(f"  Sample batch:")
    for i in range(min(2, batch_x.size(0))):
        print(f"    {batch_x[i].tolist()} -> {batch_y[i].item()}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_dataset()