#!/usr/bin/env python3
"""
Phase 2: Curriculum Dataset - Multi-level difficulty datasets for progressive learning.
Enables the agent to request harder examples when it detects confidence issues.
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from enum import Enum

class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXPERT = 3

class CurriculumSequenceDataset(Dataset):
    """
    Enhanced sequence dataset with multiple difficulty levels.
    Tasks progress from simple patterns to complex multi-step reasoning.
    """
    
    def __init__(self, size: int = 1000, difficulty: DifficultyLevel = DifficultyLevel.EASY, 
                 seq_len_range: Tuple[int, int] = (4, 12), vocab_size: int = 10):
        self.size = size
        self.difficulty = difficulty
        self.seq_len_range = seq_len_range
        self.vocab_size = vocab_size
        
        self.sequences = []
        self.labels = []
        self.metadata = []  # Store difficulty-specific info
        
        self._generate_curriculum_data()
    
    def _generate_curriculum_data(self):
        """Generate data based on difficulty level."""
        generators = {
            DifficultyLevel.EASY: self._generate_easy_data,
            DifficultyLevel.MEDIUM: self._generate_medium_data,
            DifficultyLevel.HARD: self._generate_hard_data,
            DifficultyLevel.EXPERT: self._generate_expert_data
        }
        
        generator = generators[self.difficulty]
        generator()
    
    def _generate_easy_data(self):
        """Easy: Simple even/odd sum classification, short sequences."""
        for _ in range(self.size):
            seq_len = random.randint(4, 6)  # Short sequences
            seq = [random.randint(0, 4) for _ in range(seq_len)]  # Small numbers
            label = sum(seq) % 2
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.metadata.append({
                "difficulty": "easy",
                "pattern": "even_odd_sum",
                "seq_len": seq_len,
                "sum": sum(seq)
            })
    
    def _generate_medium_data(self):
        """Medium: Longer sequences, larger vocab, multiple pattern types."""
        patterns = ['even_odd_sum', 'divisible_by_3', 'max_value_parity']
        
        for _ in range(self.size):
            seq_len = random.randint(6, 10)
            seq = [random.randint(0, self.vocab_size-1) for _ in range(seq_len)]
            pattern = random.choice(patterns)
            
            if pattern == 'even_odd_sum':
                label = sum(seq) % 2
            elif pattern == 'divisible_by_3':
                label = 1 if sum(seq) % 3 == 0 else 0
            elif pattern == 'max_value_parity':
                label = max(seq) % 2
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.metadata.append({
                "difficulty": "medium",
                "pattern": pattern,
                "seq_len": seq_len,
                "sum": sum(seq),
                "max_val": max(seq)
            })
    
    def _generate_hard_data(self):
        """Hard: Complex patterns, longer sequences, multiple conditions."""
        for _ in range(self.size):
            seq_len = random.randint(8, 12)
            seq = [random.randint(0, self.vocab_size-1) for _ in range(seq_len)]
            
            # Complex rule: label = 1 if (sum is even AND max > 5) OR (sum is odd AND contains 0)
            sum_even = sum(seq) % 2 == 0
            max_gt_5 = max(seq) > 5
            contains_zero = 0 in seq
            sum_odd = not sum_even
            
            label = 1 if (sum_even and max_gt_5) or (sum_odd and contains_zero) else 0
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.metadata.append({
                "difficulty": "hard",
                "pattern": "complex_boolean_logic",
                "seq_len": seq_len,
                "sum": sum(seq),
                "max_val": max(seq),
                "contains_zero": contains_zero,
                "rule_components": {
                    "sum_even": sum_even,
                    "max_gt_5": max_gt_5,
                    "sum_odd": sum_odd,
                    "contains_zero": contains_zero
                }
            })
    
    def _generate_expert_data(self):
        """Expert: Multi-step reasoning, sequence dependencies."""
        for _ in range(self.size):
            seq_len = random.randint(self.seq_len_range[0], self.seq_len_range[1])
            seq = [random.randint(0, self.vocab_size-1) for _ in range(seq_len)]
            
            # Expert rule: Count ascending pairs, label = 1 if count is prime
            ascending_pairs = 0
            for i in range(len(seq) - 1):
                if seq[i] < seq[i + 1]:
                    ascending_pairs += 1
            
            # Check if ascending_pairs is prime
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            
            label = 1 if is_prime(ascending_pairs) else 0
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.metadata.append({
                "difficulty": "expert",
                "pattern": "prime_ascending_pairs",
                "seq_len": seq_len,
                "ascending_pairs": ascending_pairs,
                "is_prime": is_prime(ascending_pairs)
            })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Pad sequences to maximum length for batching
        max_len = self.seq_len_range[1]
        seq = self.sequences[idx]
        
        # Pad with a special token (vocab_size) if needed
        if len(seq) < max_len:
            seq = seq + [self.vocab_size] * (max_len - len(seq))
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_metadata(self, idx):
        """Get metadata for a specific example."""
        return self.metadata[idx]
    
    def get_difficulty_stats(self) -> Dict[str, Any]:
        """Get statistics about this difficulty level."""
        if not self.metadata:
            return {}
        
        patterns = [meta['pattern'] for meta in self.metadata]
        seq_lens = [meta['seq_len'] for meta in self.metadata]
        
        stats = {
            "difficulty": self.difficulty.name,
            "size": len(self.sequences),
            "patterns": list(set(patterns)),
            "avg_seq_len": np.mean(seq_lens),
            "seq_len_range": [min(seq_lens), max(seq_lens)],
            "label_distribution": {
                "0": sum(1 for label in self.labels if label == 0),
                "1": sum(1 for label in self.labels if label == 1)
            }
        }
        
        # Add pattern-specific stats
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        stats["pattern_distribution"] = pattern_counts
        
        return stats

class CurriculumManager:
    """
    Manages curriculum progression and creates appropriate datasets.
    Used by the agent to request harder examples when needed.
    """
    
    def __init__(self, base_size: int = 1000, batch_size: int = 32):
        self.base_size = base_size
        self.batch_size = batch_size
        self.current_level = DifficultyLevel.EASY
        self.level_history = [DifficultyLevel.EASY]
        
        # Performance tracking for curriculum decisions
        self.level_performance = {level: [] for level in DifficultyLevel}
        
    def create_dataloader(self, difficulty: DifficultyLevel, size: int = None, 
                         test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create train/test dataloaders for a specific difficulty."""
        if size is None:
            size = self.base_size
        
        dataset = CurriculumSequenceDataset(
            size=size, 
            difficulty=difficulty,
            seq_len_range=(4, 12),
            vocab_size=10
        )
        
        # Split into train/test
        test_size = int(size * test_split)
        train_size = size - test_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, size))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def should_advance_curriculum(self, current_performance: float, 
                                stability_threshold: int = 3) -> bool:
        """
        Decide if the agent should advance to the next difficulty level.
        
        Args:
            current_performance: Recent accuracy on current level
            stability_threshold: Number of consistent good performances needed
        """
        self.level_performance[self.current_level].append(current_performance)
        
        # Keep only recent performance history
        if len(self.level_performance[self.current_level]) > 10:
            self.level_performance[self.current_level] = self.level_performance[self.current_level][-10:]
        
        recent_performance = self.level_performance[self.current_level][-stability_threshold:]
        
        # Advancement criteria (slightly easier to progress; require windowed stability)
        if len(recent_performance) >= stability_threshold:
            avg_performance = float(np.mean(recent_performance))
            improvements = sum(1 for p in recent_performance if p >= avg_performance)
            advancement_thresholds = {
                DifficultyLevel.EASY: 0.75,
                DifficultyLevel.MEDIUM: 0.72,
                DifficultyLevel.HARD: 0.68,
                DifficultyLevel.EXPERT: 0.65
            }
            threshold = advancement_thresholds.get(self.current_level, 0.7)
            if avg_performance >= threshold and improvements >= (stability_threshold - 1) and self.current_level != DifficultyLevel.EXPERT:
                return True
        
        return False
    
    def advance_curriculum(self) -> bool:
        """Advance to the next difficulty level if possible."""
        level_progression = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM, 
            DifficultyLevel.HARD,
            DifficultyLevel.EXPERT
        ]
        
        current_idx = level_progression.index(self.current_level)
        
        if current_idx < len(level_progression) - 1:
            self.current_level = level_progression[current_idx + 1]
            self.level_history.append(self.current_level)
            return True
        
        return False
    
    def create_mixed_difficulty_loader(self, difficulty_weights: Dict[DifficultyLevel, float], 
                                     total_size: int = 1000) -> DataLoader:
        """
        Create a dataloader with mixed difficulty examples.
        Useful for creating 'hard examples' datasets.
        """
        combined_sequences = []
        combined_labels = []
        
        for difficulty, weight in difficulty_weights.items():
            size = int(total_size * weight)
            if size > 0:
                dataset = CurriculumSequenceDataset(
                    size=size,
                    difficulty=difficulty,
                    seq_len_range=(4, 12),
                    vocab_size=10
                )
                
                for i in range(len(dataset)):
                    seq, label = dataset[i]
                    combined_sequences.append(seq)
                    combined_labels.append(label)
        
        # Create combined dataset
        class MixedDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences = sequences
                self.labels = labels
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx], self.labels[idx]
        
        mixed_dataset = MixedDataset(combined_sequences, combined_labels)
        return DataLoader(mixed_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about curriculum progression."""
        return {
            "current_level": self.current_level.name,
            "level_history": [level.name for level in self.level_history],
            "level_performance": {
                level.name: perf_list for level, perf_list in self.level_performance.items() if perf_list
            },
            "progression_count": len(set(self.level_history))
        }

def test_curriculum_dataset():
    """Test the curriculum dataset implementation."""
    print("Testing Curriculum Dataset...")
    
    manager = CurriculumManager()
    
    # Test all difficulty levels
    for difficulty in DifficultyLevel:
        print(f"\\nTesting {difficulty.name} difficulty:")
        
        train_loader, test_loader = manager.create_dataloader(difficulty, size=100)
        
        # Get a sample batch
        batch_x, batch_y = next(iter(train_loader))
        print(f"  Batch shape: {batch_x.shape}, Labels: {batch_y.shape}")
        
        # Show a few examples with metadata
        dataset = train_loader.dataset.dataset  # Get the underlying dataset
        stats = dataset.get_difficulty_stats()
        
        print(f"  Stats: {stats['patterns']}")
        print(f"  Avg sequence length: {stats['avg_seq_len']:.1f}")
        print(f"  Label distribution: {stats['label_distribution']}")
        
        # Show one example with explanation
        seq, label = dataset[0]
        metadata = dataset.get_metadata(0)
        print(f"  Example: {seq.tolist()[:8]}... -> {label.item()}")
        print(f"  Pattern: {metadata['pattern']}")
    
    # Test curriculum advancement
    print(f"\\nTesting curriculum advancement:")
    print(f"  Starting level: {manager.current_level.name}")
    
    # Simulate good performance
    for _ in range(5):
        should_advance = manager.should_advance_curriculum(0.85)
        if should_advance:
            advanced = manager.advance_curriculum()
            if advanced:
                print(f"  Advanced to: {manager.current_level.name}")
    
    print(f"  Final level: {manager.current_level.name}")
    
    # Test mixed difficulty
    print(f"\\nTesting mixed difficulty dataset:")
    mixed_loader = manager.create_mixed_difficulty_loader({
        DifficultyLevel.EASY: 0.3,
        DifficultyLevel.MEDIUM: 0.4,
        DifficultyLevel.HARD: 0.3
    })
    
    batch_x, batch_y = next(iter(mixed_loader))
    print(f"  Mixed batch shape: {batch_x.shape}")
    
    print("Curriculum Dataset test completed!")

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_curriculum_dataset()