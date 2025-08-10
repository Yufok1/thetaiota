#!/usr/bin/env python3
"""
Phase 2: Self-Reflective Agent with Task-Summoning & Meta-Learning
Integrates learned meta-controller, task spawning, and curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from init_database import create_database
from memory_db import MemoryDB, Task
from transformer_model import MinimalTransformer
from meta_controller import MetaController, MetaAction, MetaObservation
from task_spawner import TaskSpawner
from curriculum_dataset import CurriculumManager, DifficultyLevel

class Phase2Agent:
    """
    Phase 2: Advanced Self-Reflective AI Agent
    Features:
    - Learned meta-controller (replaces hard-coded rules)
    - Task spawning based on weakness detection
    - Curriculum learning with progressive difficulty
    - Task queue execution system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        
        # Identity and checkpoint config
        self.agent_id = str(self.config.get('agent_id') or Path(self.config.get('db_path', 'phase2')).stem)
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{self.agent_id}_latest.pt"
        self.load_checkpoint_enabled = bool(self.config.get('load_checkpoint', True))
        self.save_every_epochs = int(self.config.get('checkpoint_every_epochs', 1))
        
        print(f"Initializing Phase 2 Agent on {self.device}")
        
        # Initialize core components
        self._init_database()
        self._init_model()
        self._init_optimizer()
        self._init_meta_controller()
        self._init_task_system()
        self._init_curriculum()
        
        # Load checkpoint if available
        if self.load_checkpoint_enabled:
            self._load_checkpoint_if_available()

        # Optional quorum checker (injected by service). Should be a callable(action_name:str, step:int, metrics:dict)->bool
        self.quorum_checker = self.config.get('quorum_checker')
        
        # Training state
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0
        self.last_meta_update_step = 0
        
        print(f"Phase 2 Agent initialized - Model: {self.model.count_parameters():,} params, "
              f"MetaController: {sum(p.numel() for p in self.meta_controller.net.parameters())} params")
    
    def _init_database(self):
        """Initialize memory database."""
        db_path = self.config.get('db_path', 'phase2_memory.db')
        if not os.path.exists(db_path):
            create_database(db_path)
        self.memory = MemoryDB(db_path)
        
        self.memory.log_meta_event(
            event_type="phase2_agent_init",
            info={"config": self.config, "device": str(self.device)}
        )
    
    def _init_model(self):
        """Initialize the transformer model."""
        model_config = self.config.get('model', {})
        self.model = MinimalTransformer(
            vocab_size=model_config.get('vocab_size', 11),  # +1 for padding token
            d_model=model_config.get('d_model', 64),
            d_ff=model_config.get('d_ff', 256),
            max_seq_len=model_config.get('max_seq_len', 16),
            num_classes=2,
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Memory efficiency settings
        self.use_amp = self.config.get('use_amp', False)
        self.grad_accum_steps = max(1, self.config.get('grad_accum_steps', 1))
        self.use_checkpointing = self.config.get('use_checkpointing', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Enable activation checkpointing if requested
        if self.use_checkpointing:
            self.model.use_checkpointing = True
        
        # Optional PyTorch compile for speed
        if self.config.get('use_torch_compile', False):
            try:
                self.model = torch.compile(self.model)
                print("PyTorch compile enabled")
            except Exception as e:
                print(f"PyTorch compile failed: {e}")
                pass
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config.get('optimizer', {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-3),
            weight_decay=opt_config.get('weight_decay', 1e-5)
        )
    
    def _init_meta_controller(self):
        """Initialize the learned meta-controller."""
        # Phase 3: Pass memory database for recursive decision analysis
        self.meta_controller = MetaController(self.device, memory_db=self.memory)
        self.meta_training_frequency = self.config.get('meta_training_frequency', 10)
    
    def _init_task_system(self):
        """Initialize task spawning and queue system."""
        self.task_spawner = TaskSpawner(self.memory)
        self.active_tasks = []  # Currently executing tasks
        self.completed_tasks = []
        
        # Task execution config
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 3)
        self.task_execution_frequency = self.config.get('task_execution_frequency', 20)
    
    def _init_curriculum(self):
        """Initialize curriculum learning system."""
        self.curriculum = CurriculumManager(
            base_size=self.config.get('curriculum_size', 1000),
            batch_size=self.config.get('batch_size', 32)
        )
        
        # Create initial dataloaders
        self._update_curriculum_data()
    
    def _update_curriculum_data(self):
        """Update dataloaders based on current curriculum level."""
        self.train_loader, self.val_loader = self.curriculum.create_dataloader(
            difficulty=self.curriculum.current_level,
            size=self.config.get('curriculum_size', 1000)
        )
    
    def train_batch(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict[str, float]:
        """Enhanced batch training with Phase 2 features + memory efficiency."""
        self.model.train()
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
        # Gradient accumulation logic
        is_accum_step = ((self.global_step + 1) % self.grad_accum_steps) != 0
        if (self.global_step % self.grad_accum_steps) == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional AMP
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y) / self.grad_accum_steps
            loss.backward()
        
        # Gradient clipping and optimizer step
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        if not is_accum_step:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == batch_y).float().mean()
            confidence = torch.softmax(logits, dim=1).max(dim=1)[0].mean()
        
        # Get introspection data
        introspection_data = self.model.get_introspection_data()
        
        # Compile metrics
        batch_metrics = {
            'train_loss': loss.item(),
            'train_accuracy': accuracy.item(),
            'gradient_norm': gradient_norm.item(),
            'confidence': confidence.item(),
            'attention_entropy': introspection_data['attention_entropy'].item() if introspection_data['attention_entropy'] is not None else 2.0,
            'memory_usage_gb': torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0.0
        }
        
        # Log to memory
        self.memory.log_batch_metrics(
            step=self.global_step,
            metrics=batch_metrics
        )
        
        return batch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation with enhanced Phase 2 features."""
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
            'val_confidence': val_confidence / num_batches
        }
        
        # Log validation metrics
        self.memory.log_batch_metrics(
            step=self.global_step,
            metrics=val_metrics
        )
        
        # Update improvement tracking
        if val_metrics['val_loss'] < self.best_val_loss - 0.001:
            self.best_val_loss = val_metrics['val_loss']
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        return val_metrics
    
    def meta_decision_cycle(self, current_metrics: Dict[str, float]) -> bool:
        """
        Phase 2: Use learned meta-controller to make decisions.
        Returns True if significant action was taken.
        """
        # Create observation for meta-controller
        observation = self.meta_controller.observe(
            metrics=current_metrics,
            steps_since_improvement=self.steps_since_improvement,
            steps_since_last_update=self.global_step - self.last_meta_update_step
        )
        
        # Get meta-controller decision
        action, confidence, reasoning = self.meta_controller.decide_action(observation)
        
        # Rich meta-controller decision display
        print(f"\n🤖 Meta-Controller Decision (Step {self.global_step}):")
        print(f"   🎯 Action: {action.name}")
        print(f"   🎲 Confidence: {confidence:.3f} {'🔥' if confidence > 0.8 else '🤔' if confidence > 0.5 else '😕'}")
        print(f"   💭 Reasoning: {reasoning}")
        
        # Show observation details
        print(f"   📊 Observed State:")
        print(f"      Val Loss: {observation.val_loss:.4f}")
        print(f"      Model Confidence: {observation.confidence:.3f}")
        print(f"      Steps since improvement: {observation.steps_since_improvement}")
        print(f"      Steps since last meta-action: {observation.steps_since_last_update}")
        print(f"      Gradient norm: {observation.gradient_norm:.3f}")
        print(f"      Attention entropy: {observation.attention_entropy:.3f}")
        
        # Log the decision
        self.memory.log_meta_event(
            event_type="meta_decision",
            info={
                "step": self.global_step,
                "action": action.name,
                "confidence": confidence,
                "reasoning": reasoning,
                "observation": {
                    "val_loss": observation.val_loss,
                    "confidence": observation.confidence,
                    "steps_since_improvement": observation.steps_since_improvement
                }
            }
        )
        
        # Execute the action
        reward = self._execute_meta_action(action, current_metrics)
        
        # Provide feedback to meta-controller
        self.meta_controller.receive_reward(reward)
        
        # Train meta-controller periodically
        if self.global_step % self.meta_training_frequency == 0:
            # Train on a smaller batch to ensure progress with limited experience
            meta_loss = self.meta_controller.train_step(batch_size=8)
            if meta_loss is not None:
                print(f"    Meta-controller training loss: {meta_loss:.4f}")
        
        return action != MetaAction.CONTINUE_TRAINING
    
    def _execute_meta_action(self, action: MetaAction, metrics: Dict[str, float]) -> float:
        """Execute a meta-controller action and return reward."""
        pre_action_loss = metrics.get('val_loss', 1.0)
        reward = 0.0
        
        if action == MetaAction.CONTINUE_TRAINING:
            reward = 0.1  # Small positive reward for continuing when appropriate
        
        elif action == MetaAction.FINE_TUNE_NOW:
            # Quorum gate (if provided)
            if callable(getattr(self, 'quorum_checker', None)):
                approved = False
                try:
                    approved = self.quorum_checker("FINE_TUNE_NOW", self.global_step, metrics)
                except Exception as e:
                    print(f"    Quorum check failed (proceeding): {e}")
                    approved = True
                if not approved:
                    print("    Quorum rejected self-update; skipping fine-tune")
                    return 0.0

            print("    Executing: Fine-tuning on hard examples")
            self._fine_tune_on_hard_examples()
            self.last_meta_update_step = self.global_step
            
            # Measure improvement
            post_metrics = self.validate()
            improvement = pre_action_loss - post_metrics['val_loss']
            reward = improvement * 5.0  # Scale reward based on improvement

            # Save checkpoint immediately after approved update
            try:
                self._save_checkpoint()
                # Also mark/update an official checkpoint reference
                official_path = self.checkpoint_dir / "official_latest.pt"
                payload = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'timestamp': time.time(),
                    'agent_id': self.agent_id,
                }
                torch.save(payload, str(official_path))
                # Log
                self.memory.log_meta_event(
                    event_type="official_checkpoint",
                    info={"path": str(official_path), "step": self.global_step, "improvement": float(improvement)}
                )
            except Exception as e:
                print(f"    Checkpoint save failed: {e}")
        
        elif action == MetaAction.SPAWN_DATA_COLLECTION:
            print("    Executing: Spawning data collection tasks")
            self._spawn_and_execute_tasks(force_spawn=True)
            reward = 0.2  # Moderate reward for proactive task spawning
        
        elif action == MetaAction.ADJUST_LEARNING_RATE:
            print("    Executing: Adjusting learning rate")
            self._adjust_learning_rate(metrics)
            reward = 0.15  # Moderate reward for hyperparameter adjustment
        
        elif action == MetaAction.PAUSE_AND_ANALYZE:
            print("    Executing: Pausing for analysis")
            analysis = self._analyze_current_state(metrics)
            print(f"      Analysis: {analysis}")
            reward = 0.05  # Small reward for analysis
        
        return reward
    
    def _fine_tune_on_hard_examples(self):
        """Create and fine-tune on harder examples."""
        # Create mixed difficulty dataset with emphasis on hard examples
        hard_loader = self.curriculum.create_mixed_difficulty_loader({
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.5,
            DifficultyLevel.EXPERT: 0.2
        }, total_size=200)
        
        # Fine-tune for a few steps
        self.model.train()
        for i, (batch_x, batch_y) in enumerate(hard_loader):
            if i >= 5:  # Limit fine-tuning steps
                break
            
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def _adjust_learning_rate(self, metrics: Dict[str, float]):
        """Adjust learning rate based on current conditions."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Simple heuristic adjustment
        if metrics.get('gradient_norm', 1.0) > 2.0:
            # Large gradients -> reduce LR
            new_lr = current_lr * 0.8
        elif metrics.get('gradient_norm', 1.0) < 0.1:
            # Small gradients -> increase LR
            new_lr = current_lr * 1.2
        else:
            return  # No adjustment needed
        
        # Apply bounds
        new_lr = max(1e-5, min(1e-2, new_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        print(f"      Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
    
    def _analyze_current_state(self, metrics: Dict[str, float]) -> str:
        """Analyze current training state and return summary."""
        analysis_parts = []
        
        if metrics.get('val_loss', 0) > 0.7:
            analysis_parts.append("high validation loss")
        if metrics.get('val_confidence', 1.0) < 0.6:
            analysis_parts.append("low model confidence")
        if self.steps_since_improvement > 5:
            analysis_parts.append(f"no improvement for {self.steps_since_improvement} steps")
        
        if not analysis_parts:
            return "training appears stable"
        
        return ", ".join(analysis_parts)
    
    def _spawn_and_execute_tasks(self, force_spawn: bool = False):
        """Spawn tasks based on weaknesses and execute them."""
        if self.global_step % self.task_execution_frequency != 0 and not force_spawn:
            return
        
        # Get current metrics for task spawning
        current_metrics = {}
        recent_val_loss = self.memory.get_recent_metrics("val_loss", n=1)
        recent_confidence = self.memory.get_recent_metrics("val_confidence", n=1)
        recent_grad_norm = self.memory.get_recent_metrics("gradient_norm", n=1)
        recent_entropy = self.memory.get_recent_metrics("attention_entropy", n=1)
        
        if recent_val_loss:
            current_metrics['val_loss'] = recent_val_loss[0]['value']
        if recent_confidence:
            current_metrics['val_confidence'] = recent_confidence[0]['value']
        if recent_grad_norm:
            current_metrics['gradient_norm'] = recent_grad_norm[0]['value']
        if recent_entropy:
            current_metrics['attention_entropy'] = recent_entropy[0]['value']
        
        # Spawn tasks based on detected weaknesses
        spawned_tasks = self.task_spawner.analyze_and_spawn(
            current_metrics=current_metrics,
            introspection_data={},
            step=self.global_step
        )
        
        if spawned_tasks:
            print(f"    Spawned {len(spawned_tasks)} tasks:")
            for task in spawned_tasks:
                print(f"      - {task.prompt}")
                self.memory.enqueue_task(task)
                self.active_tasks.append(task)
            
            # Execute some tasks immediately
            self._execute_pending_tasks()
    
    def _execute_pending_tasks(self):
        """Execute pending tasks from the queue."""
        # For Phase 2, we'll implement basic task execution
        # More sophisticated execution will come in later phases
        
        executed_count = 0
        for task in self.active_tasks[:2]:  # Execute max 2 tasks
            if task.metadata.get('task_type') == 'fine_tune_hard_examples':
                print(f"      Executing: {task.prompt}")
                self._fine_tune_on_hard_examples()
                task.status = 'completed'
                executed_count += 1
            elif task.metadata.get('task_type') == 'adjust_hyperparams':
                print(f"      Executing: {task.prompt}")
                # Get recent metrics for adjustment
                recent_metrics = {}
                recent_grad = self.memory.get_recent_metrics("gradient_norm", n=1)
                if recent_grad:
                    recent_metrics['gradient_norm'] = recent_grad[0]['value']
                self._adjust_learning_rate(recent_metrics)
                task.status = 'completed'
                executed_count += 1
        
        # Move completed tasks
        self.active_tasks = [t for t in self.active_tasks if t.status != 'completed']
        if executed_count > 0:
            print(f"      Executed {executed_count} tasks")
    
    def _check_curriculum_advancement(self, val_accuracy: float):
        """Check if we should advance curriculum difficulty."""
        should_advance = self.curriculum.should_advance_curriculum(val_accuracy)
        
        if should_advance:
            advanced = self.curriculum.advance_curriculum()
            if advanced:
                print(f"    Curriculum advanced to: {self.curriculum.current_level.name}")
                self._update_curriculum_data()
                
                # Log curriculum advancement
                self.memory.log_meta_event(
                    event_type="curriculum_advanced",
                    info={
                        "step": self.global_step,
                        "new_level": self.curriculum.current_level.name,
                        "trigger_accuracy": val_accuracy
                    }
                )
    
    def train(self, num_epochs: int = 5):
        """Main training loop with Phase 2 enhancements."""
        print(f"Starting Phase 2 training for {num_epochs} epochs...")
        print(f"Starting curriculum level: {self.curriculum.current_level.name}")
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch+1}/{num_epochs} (Level: {self.curriculum.current_level.name})")
            
            epoch_train_loss = 0.0
            num_batches = 0
            
            # Training loop
            for batch_x, batch_y in self.train_loader:
                batch_metrics = self.train_batch(batch_x, batch_y)
                epoch_train_loss += batch_metrics['train_loss']
                num_batches += 1
                self.global_step += 1
                
                # Periodic validation and meta-decisions
                if self.global_step % 25 == 0:
                    val_metrics = self.validate()
                    
                    # Combine metrics for meta-controller
                    combined_metrics = {**batch_metrics, **val_metrics}
                    
                    # Rich training progress display
                    self._log_rich_training_step(combined_metrics)
                    
                    # Phase 2: Meta-decision making
                    meta_action_taken = self.meta_decision_cycle(combined_metrics)
                    
                    # Phase 2: Task spawning and execution
                    self._spawn_and_execute_tasks()
                    
                    # Phase 2: Curriculum advancement check
                    self._check_curriculum_advancement(val_metrics['val_accuracy'])
            
            # End of epoch summary
            avg_train_loss = epoch_train_loss / num_batches
            final_val_metrics = self.validate()
            
            print(f"  Epoch summary: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={final_val_metrics['val_loss']:.4f}, "
                  f"val_acc={final_val_metrics['val_accuracy']:.3f}")
            # Save checkpoint periodically
            if (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint()
        
        # Final statistics
        self._print_training_summary()
    
    def _print_training_summary(self):
        """Print comprehensive training summary."""
        print(f"\\n=== PHASE 2 TRAINING SUMMARY ===")
        
        # Meta-controller stats
        meta_stats = self.meta_controller.get_stats()
        print(f"Meta-Controller:")
        print(f"  Training steps: {meta_stats.get('total_steps', 0)}")
        print(f"  Avg recent reward: {meta_stats.get('avg_recent_reward', 0):.3f}")
        
        # Task spawning stats
        spawn_stats = self.task_spawner.get_spawn_stats()
        print(f"Task Spawning:")
        print(f"  Total spawned: {spawn_stats.get('total_spawned', 0)}")
        print(f"  Most common weakness: {spawn_stats.get('most_common_weakness', 'None')}")
        
        # Curriculum stats
        curriculum_stats = self.curriculum.get_curriculum_stats()
        print(f"Curriculum Learning:")
        print(f"  Final level: {curriculum_stats['current_level']}")
        print(f"  Progression: {' -> '.join(curriculum_stats['level_history'])}")
        
        print("=== Phase 2 Complete! ===")

    def _log_rich_training_step(self, metrics: Dict[str, float]):
        """Enhanced training step logging with fascinating details."""
        step = self.global_step
        
        # Core metrics with visual bars
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        val_acc = metrics['val_accuracy']
        
        print(f"\n🧠 Step {step}: Learning Progress")
        print(f"   📉 Train Loss: {train_loss:.4f} {'█' * min(10, int((2.0-train_loss)*10))}{'░' * (10-min(10, int((2.0-train_loss)*10)))}")
        print(f"   📊 Val Loss:   {val_loss:.4f} {'█' * min(10, int((2.0-val_loss)*10))}{'░' * (10-min(10, int((2.0-val_loss)*10)))}")
        print(f"   🎯 Val Acc:    {val_acc:.3f} {'█' * min(10, int(val_acc*10))}{'░' * (10-min(10, int(val_acc*10)))}")
        
        # Memory usage
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"   💾 VRAM: {allocated:.2f}GB / Peak: {max_allocated:.2f}GB")
        except Exception:
            pass
        
        # Gradient health
        grad_norm = metrics.get('gradient_norm', 0.0)
        confidence = metrics.get('confidence', 0.0)
        print(f"   ⚡ Grad Norm: {grad_norm:.3f} {'⚠️' if grad_norm > 2.0 else '✅' if grad_norm > 0.1 else '😴'}")
        print(f"   🎲 Confidence: {confidence:.3f} {'🔥' if confidence > 0.8 else '🤔' if confidence > 0.5 else '😕'}")
        
        # Learning dynamics
        if hasattr(self, '_last_val_loss') and self._last_val_loss is not None:
            delta = self._last_val_loss - val_loss
            trend = "📈 Improving" if delta > 0.01 else "📉 Worsening" if delta < -0.01 else "➡️ Stable"
            print(f"   📍 Trend: {trend} (Δ{delta:+.4f})")
        self._last_val_loss = val_loss
        
        # Attention insights (if available)
        try:
            attn_entropy = metrics.get('attention_entropy', None)
            if attn_entropy is not None:
                focus_level = "🎯 Focused" if attn_entropy < 2.0 else "👁️ Scanning" if attn_entropy < 3.0 else "🌀 Dispersed"
                print(f"   👀 Attention: {focus_level} (entropy: {attn_entropy:.2f})")
        except Exception:
            pass
        
        # Meta-controller readiness
        steps_since_improvement = getattr(self, 'steps_since_improvement', 0)
        if steps_since_improvement > 5:
            print(f"   🤖 Meta-Controller: 🚨 Plateau detected ({steps_since_improvement} steps)")
        elif steps_since_improvement > 0:
            print(f"   🤖 Meta-Controller: 👁️ Monitoring ({steps_since_improvement} steps)")
        else:
            print(f"   🤖 Meta-Controller: ✅ Model improving")

    # ---------------------- Checkpointing ----------------------
    def _save_checkpoint(self):
        try:
            payload = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'timestamp': time.time(),
                'agent_id': self.agent_id,
            }
            # Persist meta-controller state (small; helps resume policy)
            try:
                payload['meta_controller'] = {
                    'net': self.meta_controller.net.state_dict() if hasattr(self, 'meta_controller') else {},
                    'opt': self.meta_controller.optimizer.state_dict() if hasattr(self, 'meta_controller') else {},
                    'total_steps': int(getattr(self.meta_controller, 'total_steps', 0))
                }
            except Exception:
                pass
            torch.save(payload, str(self.checkpoint_path))
            print(f"Saved checkpoint: {self.checkpoint_path}")
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def _load_checkpoint_if_available(self):
        try:
            if self.checkpoint_path.exists():
                try:
                    payload = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=True)
                except TypeError:
                    payload = torch.load(str(self.checkpoint_path), map_location=self.device)
                model_state = payload.get('model')
                if model_state:
                    self.model.load_state_dict(model_state)
                opt_state = payload.get('optimizer')
                if opt_state:
                    self.optimizer.load_state_dict(opt_state)
                self.global_step = int(payload.get('global_step', self.global_step))
                # Restore meta-controller if present
                try:
                    meta_state = payload.get('meta_controller')
                    if meta_state and hasattr(self, 'meta_controller'):
                        net_state = meta_state.get('net') or {}
                        if net_state:
                            self.meta_controller.net.load_state_dict(net_state)
                        opt_state = meta_state.get('opt') or {}
                        if opt_state:
                            self.meta_controller.optimizer.load_state_dict(opt_state)
                        if 'total_steps' in meta_state:
                            self.meta_controller.total_steps = int(meta_state['total_steps'])
                except Exception:
                    pass
                print(f"Loaded checkpoint for {self.agent_id} @ step {self.global_step}")
        except Exception as e:
            print(f"Checkpoint load failed (continuing fresh): {e}")

    # ---------------------- Official Checkpoint Promotion/Pull ----------------------
    def save_official_checkpoint_from_current(self) -> str:
        """Save the current model/optimizer state as the federation's official checkpoint."""
        official_path = self.checkpoint_dir / "official_latest.pt"
        try:
            payload = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'timestamp': time.time(),
                'agent_id': self.agent_id,
            }
            torch.save(payload, str(official_path))
            # Append to guardian ledger if available
            try:
                from guardian.validator import append_ledger
                ledger = os.path.join(str(self.checkpoint_dir), 'ledger.json')
                meta = {"agent": self.agent_id, "step": int(self.global_step)}
                append_ledger(str(official_path), ledger, meta)
            except Exception:
                pass
            print(f"Saved OFFICIAL checkpoint: {official_path}")
            return str(official_path)
        except Exception as e:
            print(f"Official checkpoint save failed: {e}")
            return ""

    def load_official_checkpoint(self) -> bool:
        """Load the federation's official checkpoint if available."""
        official_path = self.checkpoint_dir / "official_latest.pt"
        try:
            if not official_path.exists():
                print("No OFFICIAL checkpoint found to pull.")
                return False
            try:
                payload = torch.load(str(official_path), map_location=self.device, weights_only=True)
            except TypeError:
                payload = torch.load(str(official_path), map_location=self.device)
            model_state = payload.get('model')
            if model_state:
                self.model.load_state_dict(model_state)
            opt_state = payload.get('optimizer')
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
            self.global_step = int(payload.get('global_step', self.global_step))
            print(f"Loaded OFFICIAL checkpoint @ step {self.global_step}")
            return True
        except Exception as e:
            print(f"Official checkpoint load failed: {e}")
            return False

    def get_official_checkpoint_step(self) -> int:
        """Return the step stored in the official checkpoint, or -1 if missing."""
        official_path = self.checkpoint_dir / "official_latest.pt"
        try:
            if not official_path.exists():
                return -1
            try:
                payload = torch.load(str(official_path), map_location=self.device, weights_only=True)
            except TypeError:
                payload = torch.load(str(official_path), map_location=self.device)
            return int(payload.get('global_step', -1))
        except Exception:
            return -1

    def load_official_checkpoint_if_newer(self) -> bool:
        """Load official checkpoint only if it is ahead of current step."""
        official_step = self.get_official_checkpoint_step()
        if official_step <= 0 or official_step <= self.global_step:
            return False
        return self.load_official_checkpoint()

def main():
    """Run Phase 2 demonstration."""
    print("=== PHASE 2: TASK-SUMMONING & META-LEARNING ===\\n")
    
    config = {
        'model': {
            'vocab_size': 11,  # 0-9 + padding token
            'd_model': 64,
            'd_ff': 256,
            'max_seq_len': 16,
            'dropout': 0.1
        },
        'optimizer': {
            'lr': 2e-3,
            'weight_decay': 1e-5
        },
        'curriculum_size': 800,
        'batch_size': 16,
        'meta_training_frequency': 15,
        'task_execution_frequency': 30,
        'max_concurrent_tasks': 2,
        'db_path': 'phase2_demo.db'
    }
    
    # Create and train Phase 2 agent
    agent = Phase2Agent(config)
    agent.train(num_epochs=3)

if __name__ == "__main__":
    import os
    main()