#!/usr/bin/env python3
"""
Phase 1: Minimal Self-Reflective Learner
Complete training loop with introspection and self-update capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - visualization will be skipped")
import time
import os
from typing import Dict, List, Any

from init_database import create_database
from memory_db import MemoryDB, IntrospectionEntry, Task
from transformer_model import MinimalTransformer
from toy_dataset import create_dataloaders

class SelfReflectiveLearner:
    """
    Phase 1 self-reflective learning agent.
    Combines transformer model with introspection and self-update capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        
        # Initialize components
        self._init_database()
        self._init_model()
        self._init_optimizer()
        self._init_data()
        
        # Self-update state
        self.plateau_steps = 0
        self.best_val_loss = float('inf')
        self.last_self_update_step = -1
        
        print(f"SelfReflectiveLearner initialized on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
    
    def _init_database(self):
        """Initialize the memory database."""
        db_path = self.config.get('db_path', 'agent_memory.db')
        if not os.path.exists(db_path):
            create_database(db_path)
        self.memory = MemoryDB(db_path)
        # Keep connection open for the agent's lifetime
        self.memory_conn = self.memory.conn
        
        # Log initialization
        self.memory.log_meta_event(
            event_type="agent_init",
            info={
                "config": self.config,
                "device": str(self.device),
                "timestamp": time.time()
            }
        )
    
    def _init_model(self):
        """Initialize the transformer model."""
        model_config = self.config.get('model', {})
        self.model = MinimalTransformer(
            vocab_size=model_config.get('vocab_size', 100),
            d_model=model_config.get('d_model', 64),
            d_ff=model_config.get('d_ff', 256),
            max_seq_len=model_config.get('max_seq_len', 32),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _init_optimizer(self):
        """Initialize the optimizer."""
        opt_config = self.config.get('optimizer', {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-3),
            weight_decay=opt_config.get('weight_decay', 1e-5)
        )
    
    def _init_data(self):
        """Initialize the data loaders."""
        data_config = self.config.get('data', {})
        self.train_loader, self.val_loader = create_dataloaders(
            dataset_type=data_config.get('type', 'sequence'),
            train_size=data_config.get('train_size', 8000),
            val_size=data_config.get('val_size', 2000),
            batch_size=data_config.get('batch_size', 32)
        )
    
    def train_batch(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict[str, float]:
        """Train on a single batch and collect introspection data."""
        self.model.train()
        
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
        # Forward pass
        logits = self.model(batch_x)
        loss = self.criterion(logits, batch_y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == batch_y).float().mean()
            confidence = torch.softmax(logits, dim=1).max(dim=1)[0].mean()
        
        # Collect introspection data
        introspection_data = self.model.get_introspection_data()
        param_stats = self.model.get_parameter_stats()
        
        # Memory usage
        memory_usage = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
        
        # Compile batch metrics
        batch_metrics = {
            'train_loss': loss.item(),
            'train_accuracy': accuracy.item(),
            'gradient_norm': gradient_norm.item(),
            'confidence': confidence.item(),
            'memory_usage_gb': memory_usage,
            'attention_entropy': introspection_data['attention_entropy'].item() if introspection_data['attention_entropy'] is not None else 0,
            'hidden_mean': introspection_data['hidden_mean'].item() if introspection_data['hidden_mean'] is not None else 0,
            'hidden_std': introspection_data['hidden_std'].item() if introspection_data['hidden_std'] is not None else 0,
        }
        
        # Log to memory
        # Ensure DB connection is active (context managers may have been used)
        try:
            self.memory.ensure_connection()
        except Exception:
            pass
        self.memory.log_batch_metrics(
            step=self.global_step,
            metrics=batch_metrics,
            aux_data={
                'batch_size': batch_x.size(0),
                'seq_len': batch_x.size(1) if len(batch_x.shape) > 1 else None,
                'param_stats_sample': {k: v for k, v in list(param_stats.items())[:5]}
            }
        )
        
        return batch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        val_confidence = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == batch_y).float().mean()
                confidence = torch.softmax(logits, dim=1).max(dim=1)[0].mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                val_confidence += confidence.item()
                num_batches += 1
        
        val_metrics = {
            'val_loss': val_loss / num_batches,
            'val_accuracy': val_accuracy / num_batches,
            'val_confidence': val_confidence / num_batches,
        }
        
        # Log validation metrics
        self.memory.log_batch_metrics(
            step=self.global_step,
            metrics=val_metrics
        )
        
        return val_metrics
    
    def check_self_update_triggers(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if self-update should be triggered based on introspection.
        Returns True if self-update was triggered.
        """
        current_val_loss = val_metrics['val_loss']
        current_confidence = val_metrics['val_confidence']
        
        # Update plateau tracking
        if current_val_loss < self.best_val_loss - 0.001:  # Improvement threshold
            self.best_val_loss = current_val_loss
            self.plateau_steps = 0
        else:
            self.plateau_steps += 1
        
        # Get thresholds from config (using default values to avoid DB issues in demo)
        plateau_threshold = 3
        confidence_threshold = 0.7
        
        # Check trigger conditions
        triggers = []
        
        if self.plateau_steps >= plateau_threshold:
            triggers.append(("plateau_detected", f"No improvement for {self.plateau_steps} evaluations"))
        
        if current_confidence < confidence_threshold:
            triggers.append(("low_confidence", f"Average confidence {current_confidence:.3f} below threshold {confidence_threshold}"))
        
        # Trigger self-update if conditions met and not too recent
        steps_since_last_update = self.global_step - self.last_self_update_step
        min_steps_between_updates = self.config.get('min_steps_between_updates', 100)
        
        if triggers and steps_since_last_update >= min_steps_between_updates:
            self._perform_self_update(triggers, val_metrics)
            return True
        
        return False
    
    def _perform_self_update(self, triggers: List[tuple], val_metrics: Dict[str, float]):
        """
        Perform self-update based on detected issues.
        This is the core self-reflection and improvement mechanism.
        """
        self.last_self_update_step = self.global_step
        
        print(f"\\n=== SELF-UPDATE TRIGGERED at step {self.global_step} ===")
        for trigger_type, reason in triggers:
            print(f"  Trigger: {trigger_type} - {reason}")
        
        # Log the self-update event
        self.memory.log_meta_event(
            event_type="self_update_start",
            info={
                "triggers": triggers,
                "val_metrics": val_metrics,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "plateau_steps": self.plateau_steps
            }
        )
        
        # Save model state before update
        pre_update_state = self.model.state_dict()
        pre_update_loss = val_metrics['val_loss']
        
        # Self-update strategy: Fine-tune on a few synthetic hard examples
        self._fine_tune_on_hard_examples()
        
        # Evaluate the effect
        post_update_metrics = self.validate()
        post_update_loss = post_update_metrics['val_loss']
        
        improvement = pre_update_loss - post_update_loss
        
        print(f"  Pre-update val_loss: {pre_update_loss:.4f}")
        print(f"  Post-update val_loss: {post_update_loss:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        # If update made things worse, revert
        if improvement < -0.01:  # Tolerance for small fluctuations
            print("  Self-update made performance worse - reverting!")
            self.model.load_state_dict(pre_update_state)
        else:
            print("  Self-update accepted!")
            self.plateau_steps = 0  # Reset plateau counter
            if post_update_loss < self.best_val_loss:
                self.best_val_loss = post_update_loss
        
        # Log the outcome
        self.memory.log_meta_event(
            event_type="self_update_complete",
            info={
                "pre_update_loss": pre_update_loss,
                "post_update_loss": post_update_loss,
                "improvement": improvement,
                "reverted": improvement < -0.01,
                "global_step": self.global_step
            }
        )
        
        print("=== SELF-UPDATE COMPLETE ===\\n")
    
    def _fine_tune_on_hard_examples(self):
        """
        Generate and fine-tune on synthetic hard examples.
        This is a simple self-update strategy for Phase 1.
        """
        print("  Generating hard examples for fine-tuning...")
        
        # Create a small dataset of potentially harder examples
        # For sequence classification: longer sequences or edge cases
        from toy_dataset import ToySequenceDataset
        from torch.utils.data import DataLoader
        
        hard_dataset = ToySequenceDataset(size=100, seq_len=12)  # Slightly longer sequences
        hard_loader = DataLoader(hard_dataset, batch_size=16, shuffle=True)
        
        # Fine-tune for a few steps
        self.model.train()
        fine_tune_steps = 10
        
        for step, (batch_x, batch_y) in enumerate(hard_loader):
            if step >= fine_tune_steps:
                break
            
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            print(f"    Fine-tune step {step+1}/{fine_tune_steps}, loss: {loss.item():.4f}")
    
    def train(self, num_epochs: int = 10):
        """Main training loop with self-reflection."""
        print(f"Starting training for {num_epochs} epochs...")
        
        # Tracking for visualization
        train_losses = []
        val_losses = []
        val_accuracies = []
        self_update_steps = []
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.train_loader:
                batch_metrics = self.train_batch(batch_x, batch_y)
                epoch_train_loss += batch_metrics['train_loss']
                num_batches += 1
                self.global_step += 1
                
                # Periodic validation and self-update check
                if self.global_step % self.config.get('val_frequency', 50) == 0:
                    val_metrics = self.validate()
                    
                    print(f"  Step {self.global_step}: train_loss={batch_metrics['train_loss']:.4f}, "
                          f"val_loss={val_metrics['val_loss']:.4f}, "
                          f"val_acc={val_metrics['val_accuracy']:.3f}")
                    
                    # Check for self-update triggers
                    if self.check_self_update_triggers(val_metrics):
                        self_update_steps.append(self.global_step)
            
            # End of epoch validation
            val_metrics = self.validate()
            avg_train_loss = epoch_train_loss / num_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_metrics['val_loss'])
            val_accuracies.append(val_metrics['val_accuracy'])
            
            print(f"  Epoch summary: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_acc={val_metrics['val_accuracy']:.3f}")
        
        print("\\nTraining completed!")
        self._create_visualization(train_losses, val_losses, val_accuracies, self_update_steps)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'self_update_steps': self_update_steps
        }
    
    def _create_visualization(self, train_losses, val_losses, val_accuracies, self_update_steps):
        """Create training visualization with self-update markers."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping visualization - matplotlib not available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
        
        # Mark self-update points (convert steps to approximate epochs)
        for step in self_update_steps:
            epoch_approx = step / (len(train_losses) * len(self.train_loader))
            if epoch_approx <= len(train_losses):
                ax1.axvline(x=epoch_approx, color='orange', linestyle='--', alpha=0.8)
                ax1.text(epoch_approx, max(train_losses + val_losses) * 0.9, 'Self-Update', 
                        rotation=90, fontsize=8, color='orange')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss with Self-Updates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"phase1_training_results_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training visualization saved: {plot_path}")
        
        plt.show()

def main():
    """Run the Phase 1 self-reflective learning demonstration."""
    
    # Configuration
    config = {
        'model': {
            'vocab_size': 100,
            'd_model': 64,
            'd_ff': 256,
            'max_seq_len': 32,
            'num_classes': 2,
            'dropout': 0.1
        },
        'optimizer': {
            'lr': 1e-3,
            'weight_decay': 1e-5
        },
        'data': {
            'type': 'sequence',
            'train_size': 8000,
            'val_size': 2000,
            'batch_size': 32
        },
        'val_frequency': 50,
        'min_steps_between_updates': 100,
        'db_path': 'phase1_memory.db'
    }
    
    # Create and train the self-reflective learner
    learner = SelfReflectiveLearner(config)
    results = learner.train(num_epochs=5)
    
    print(f"\\nTraining Results:")
    print(f"  Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"  Final val loss: {results['val_losses'][-1]:.4f}")
    print(f"  Final val accuracy: {results['val_accuracies'][-1]:.4f}")
    print(f"  Self-updates triggered: {len(results['self_update_steps'])}")
    
    # Demonstrate introspection capability
    print(f"\\n=== PHASE 1 INTROSPECTION DEMO ===")
    with learner.memory as memory:
        explanation = memory.explain_decision(step_id=learner.global_step - 10)
        print(f"Agent explanation for recent activity:")
        print(f"  {explanation['summary']}")

if __name__ == "__main__":
    main()