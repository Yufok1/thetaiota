#!/usr/bin/env python3
"""
Phase 2: Task Spawning System - Automatic generation of subtasks based on agent weaknesses.
Implements self-reflection and task decomposition capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from memory_db import Task, MemoryDB, IntrospectionEntry

class TaskType(Enum):
    """Types of tasks the agent can spawn."""
    FINE_TUNE_HARD_EXAMPLES = "fine_tune_hard_examples"
    COLLECT_MORE_DATA = "collect_more_data" 
    ANALYZE_ERRORS = "analyze_errors"
    ADJUST_HYPERPARAMS = "adjust_hyperparams"
    VALIDATE_PERFORMANCE = "validate_performance"
    GENERATE_CURRICULUM = "generate_curriculum"

@dataclass
class WeaknessSignal:
    """Represents a detected weakness that could spawn a task."""
    signal_type: str
    severity: float  # 0.0 = minor, 1.0 = critical
    description: str
    suggested_action: TaskType
    context: Dict[str, Any]
    
class TaskSpawner:
    """
    Analyzes agent performance and spawns appropriate subtasks.
    This is where the agent becomes self-reflective about its weaknesses.
    """
    
    def __init__(self, memory_db: MemoryDB):
        self.memory = memory_db
        self.spawn_history = []
        self.weakness_detectors = {
            'low_confidence': self._detect_low_confidence,
            'plateau_performance': self._detect_plateau,
            'error_patterns': self._detect_error_patterns,
            'gradient_issues': self._detect_gradient_problems,
            'attention_dispersion': self._detect_attention_issues
        }
        
        # Thresholds for weakness detection
        self.thresholds = {
            'confidence_threshold': 0.6,
            'plateau_steps': 5,
            'gradient_norm_min': 0.01,
            'gradient_norm_max': 5.0,
            'attention_entropy_min': 1.0,
            'attention_entropy_max': 3.5
        }
    
    def analyze_and_spawn(self, current_metrics: Dict[str, float], 
                         introspection_data: Dict[str, Any],
                         step: int) -> List[Task]:
        """
        Main entry point: analyze current state and spawn appropriate tasks.
        """
        # Detect weaknesses
        weaknesses = self._detect_weaknesses(current_metrics, introspection_data, step)
        
        # Generate tasks based on weaknesses
        spawned_tasks = []
        for weakness in weaknesses:
            task = self._weakness_to_task(weakness, step)
            if task and self._should_spawn_task(task):
                spawned_tasks.append(task)
                self.spawn_history.append((step, weakness.signal_type, task.prompt))
        
        # Log task spawning event
        if spawned_tasks:
            self.memory.log_meta_event(
                event_type="tasks_spawned",
                info={
                    "step": step,
                    "num_tasks": len(spawned_tasks),
                    "weaknesses": [w.signal_type for w in weaknesses],
                    "task_types": [t.objective for t in spawned_tasks]
                }
            )
        
        return spawned_tasks
    
    def _detect_weaknesses(self, metrics: Dict[str, float], 
                          introspection_data: Dict[str, Any], 
                          step: int) -> List[WeaknessSignal]:
        """Run all weakness detectors and return found issues."""
        weaknesses = []
        
        for detector_name, detector_func in self.weakness_detectors.items():
            weakness = detector_func(metrics, introspection_data, step)
            if weakness:
                weaknesses.append(weakness)
        
        # Sort by severity (most critical first)
        weaknesses.sort(key=lambda w: w.severity, reverse=True)
        
        return weaknesses
    
    def _detect_low_confidence(self, metrics: Dict[str, float], 
                              introspection_data: Dict[str, Any], 
                              step: int) -> Optional[WeaknessSignal]:
        """Detect if the model has low confidence in its predictions."""
        confidence = metrics.get('val_confidence', 1.0)
        
        if confidence < self.thresholds['confidence_threshold']:
            severity = (self.thresholds['confidence_threshold'] - confidence) / self.thresholds['confidence_threshold']
            
            return WeaknessSignal(
                signal_type="low_confidence",
                severity=severity,
                description=f"Model confidence {confidence:.3f} below threshold {self.thresholds['confidence_threshold']}",
                suggested_action=TaskType.COLLECT_MORE_DATA,
                context={"confidence": confidence, "threshold": self.thresholds['confidence_threshold']}
            )
        
        return None
    
    def _detect_plateau(self, metrics: Dict[str, float], 
                       introspection_data: Dict[str, Any], 
                       step: int) -> Optional[WeaknessSignal]:
        """Detect if learning has plateaued."""
        # Use memory to check recent loss trend
        recent_losses = self.memory.get_recent_metrics("val_loss", n=self.thresholds['plateau_steps'])
        
        if len(recent_losses) >= self.thresholds['plateau_steps']:
            # Sort entries by timestamp first
            sorted_entries = sorted(recent_losses, key=lambda x: x['timestamp'])
            losses = [entry['value'] for entry in sorted_entries]
            
            # Check if improvement is minimal
            improvement = losses[0] - losses[-1]  # Positive = improvement for loss
            
            if improvement < 0.001:  # Very small improvement threshold
                severity = 0.7  # High severity for plateau
                
                return WeaknessSignal(
                    signal_type="plateau_performance",
                    severity=severity,
                    description=f"No significant improvement in last {len(losses)} evaluations",
                    suggested_action=TaskType.FINE_TUNE_HARD_EXAMPLES,
                    context={"recent_losses": losses, "improvement": improvement}
                )
        
        return None
    
    def _detect_error_patterns(self, metrics: Dict[str, float], 
                              introspection_data: Dict[str, Any], 
                              step: int) -> Optional[WeaknessSignal]:
        """Detect patterns in errors that suggest specific weaknesses."""
        val_accuracy = metrics.get('val_accuracy', 1.0)
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        
        # Check for overfitting (train loss << val loss)
        if train_loss < val_loss - 0.2:
            severity = min((val_loss - train_loss) / 0.5, 1.0)
            
            return WeaknessSignal(
                signal_type="overfitting_pattern",
                severity=severity,
                description=f"Overfitting detected: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}",
                suggested_action=TaskType.ADJUST_HYPERPARAMS,
                context={"train_loss": train_loss, "val_loss": val_loss, "gap": val_loss - train_loss}
            )
        
        # Check for poor accuracy despite reasonable loss
        if val_accuracy < 0.6 and val_loss < 0.7:
            severity = (0.6 - val_accuracy) / 0.6
            
            return WeaknessSignal(
                signal_type="accuracy_loss_mismatch",
                severity=severity,
                description=f"Low accuracy ({val_accuracy:.3f}) despite reasonable loss ({val_loss:.3f})",
                suggested_action=TaskType.ANALYZE_ERRORS,
                context={"val_accuracy": val_accuracy, "val_loss": val_loss}
            )
        
        return None
    
    def _detect_gradient_problems(self, metrics: Dict[str, float], 
                                 introspection_data: Dict[str, Any], 
                                 step: int) -> Optional[WeaknessSignal]:
        """Detect gradient-related training issues."""
        grad_norm = metrics.get('gradient_norm', 1.0)
        
        # Vanishing gradients
        if grad_norm < self.thresholds['gradient_norm_min']:
            severity = (self.thresholds['gradient_norm_min'] - grad_norm) / self.thresholds['gradient_norm_min']
            
            return WeaknessSignal(
                signal_type="vanishing_gradients",
                severity=severity,
                description=f"Very small gradient norm: {grad_norm:.6f}",
                suggested_action=TaskType.ADJUST_HYPERPARAMS,
                context={"gradient_norm": grad_norm, "threshold": self.thresholds['gradient_norm_min']}
            )
        
        # Exploding gradients
        if grad_norm > self.thresholds['gradient_norm_max']:
            severity = min((grad_norm - self.thresholds['gradient_norm_max']) / self.thresholds['gradient_norm_max'], 1.0)
            
            return WeaknessSignal(
                signal_type="exploding_gradients",
                severity=severity,
                description=f"Very large gradient norm: {grad_norm:.3f}",
                suggested_action=TaskType.ADJUST_HYPERPARAMS,
                context={"gradient_norm": grad_norm, "threshold": self.thresholds['gradient_norm_max']}
            )
        
        return None
    
    def _detect_attention_issues(self, metrics: Dict[str, float], 
                                introspection_data: Dict[str, Any], 
                                step: int) -> Optional[WeaknessSignal]:
        """Detect issues with attention patterns."""
        attention_entropy = metrics.get('attention_entropy', 2.0)
        
        # Too focused (low entropy) - might be overfitting to spurious patterns
        if attention_entropy < self.thresholds['attention_entropy_min']:
            severity = (self.thresholds['attention_entropy_min'] - attention_entropy) / self.thresholds['attention_entropy_min']
            
            return WeaknessSignal(
                signal_type="attention_too_focused",
                severity=severity,
                description=f"Attention too focused (entropy: {attention_entropy:.3f}), may be overfitting",
                suggested_action=TaskType.COLLECT_MORE_DATA,
                context={"attention_entropy": attention_entropy, "threshold": self.thresholds['attention_entropy_min']}
            )
        
        # Too dispersed (high entropy) - not learning patterns effectively
        if attention_entropy > self.thresholds['attention_entropy_max']:
            severity = min((attention_entropy - self.thresholds['attention_entropy_max']) / 2.0, 1.0)
            
            return WeaknessSignal(
                signal_type="attention_too_dispersed",
                severity=severity,
                description=f"Attention too dispersed (entropy: {attention_entropy:.3f}), not focusing on patterns",
                suggested_action=TaskType.FINE_TUNE_HARD_EXAMPLES,
                context={"attention_entropy": attention_entropy, "threshold": self.thresholds['attention_entropy_max']}
            )
        
        return None
    
    def _weakness_to_task(self, weakness: WeaknessSignal, step: int) -> Optional[Task]:
        """Convert a detected weakness into a concrete task."""
        
        task_generators = {
            TaskType.FINE_TUNE_HARD_EXAMPLES: self._create_fine_tune_task,
            TaskType.COLLECT_MORE_DATA: self._create_data_collection_task,
            TaskType.ANALYZE_ERRORS: self._create_error_analysis_task,
            TaskType.ADJUST_HYPERPARAMS: self._create_hyperparam_task,
            TaskType.VALIDATE_PERFORMANCE: self._create_validation_task,
            TaskType.GENERATE_CURRICULUM: self._create_curriculum_task
        }
        
        generator = task_generators.get(weakness.suggested_action)
        if generator:
            return generator(weakness, step)
        
        return None
    
    def _create_fine_tune_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create a fine-tuning task."""
        priority = int(weakness.severity * 3)  # 0-3 priority based on severity
        
        return Task(
            prompt=f"Fine-tune on challenging examples to address {weakness.signal_type}",
            priority=priority,
            objective="Improve model performance on difficult cases",
            created_by="self_spawn",
            metadata={
                "task_type": "fine_tune_hard_examples",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _create_data_collection_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create a data collection task."""
        priority = max(1, int(weakness.severity * 3))
        
        return Task(
            prompt=f"Collect additional training data to address {weakness.signal_type}",
            priority=priority,
            objective="Gather more diverse training examples",
            created_by="self_spawn",
            metadata={
                "task_type": "collect_more_data",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _create_error_analysis_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create an error analysis task."""
        return Task(
            prompt=f"Analyze error patterns related to {weakness.signal_type}",
            priority=2,
            objective="Understand systematic errors and failure modes",
            created_by="self_spawn",
            metadata={
                "task_type": "analyze_errors",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _create_hyperparam_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create a hyperparameter adjustment task."""
        priority = 1 if weakness.severity > 0.7 else 2
        
        return Task(
            prompt=f"Adjust hyperparameters to address {weakness.signal_type}",
            priority=priority,
            objective="Optimize training hyperparameters",
            created_by="self_spawn",
            metadata={
                "task_type": "adjust_hyperparams",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _create_validation_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create a validation task."""
        return Task(
            prompt=f"Validate model performance after addressing {weakness.signal_type}",
            priority=3,
            objective="Verify improvement from recent changes",
            created_by="self_spawn",
            metadata={
                "task_type": "validate_performance",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _create_curriculum_task(self, weakness: WeaknessSignal, step: int) -> Task:
        """Create a curriculum generation task."""
        return Task(
            prompt=f"Generate curriculum to systematically address {weakness.signal_type}",
            priority=2,
            objective="Create structured learning progression",
            created_by="self_spawn",
            metadata={
                "task_type": "generate_curriculum",
                "weakness_type": weakness.signal_type,
                "severity": weakness.severity,
                "context": weakness.context,
                "spawn_step": step
            }
        )
    
    def _should_spawn_task(self, task: Task) -> bool:
        """Decide whether to actually spawn this task (avoid duplicates, etc.)."""
        # Check recent spawn history to avoid duplicate tasks
        recent_spawns = [spawn for spawn in self.spawn_history[-10:]]  # Last 10 spawns
        
        task_type = task.metadata.get('task_type', '')
        weakness_type = task.metadata.get('weakness_type', '')
        
        # Count similar recent tasks
        similar_count = sum(1 for _, w_type, prompt in recent_spawns 
                          if w_type == weakness_type or task_type in prompt)
        
        # Don't spawn too many similar tasks
        if similar_count >= 2:
            return False
        
        return True
    
    def get_spawn_stats(self) -> Dict[str, Any]:
        """Get statistics about task spawning behavior."""
        if not self.spawn_history:
            return {"total_spawned": 0}
        
        recent_spawns = self.spawn_history[-20:]  # Last 20
        weakness_counts = {}
        
        for step, weakness_type, prompt in recent_spawns:
            weakness_counts[weakness_type] = weakness_counts.get(weakness_type, 0) + 1
        
        return {
            "total_spawned": len(self.spawn_history),
            "recent_spawned": len(recent_spawns),
            "weakness_distribution": weakness_counts,
            "most_common_weakness": max(weakness_counts, key=weakness_counts.get) if weakness_counts else None
        }

def test_task_spawner():
    """Test the TaskSpawner implementation."""
    print("Testing TaskSpawner...")
    
    # Create test database
    from init_database import create_database
    create_database("test_spawner.db")
    
    with MemoryDB("test_spawner.db") as memory:
        spawner = TaskSpawner(memory)
        
        # Test weakness detection with various scenarios
        test_scenarios = [
            {
                'name': 'Low Confidence',
                'metrics': {'val_confidence': 0.3, 'val_loss': 0.8, 'val_accuracy': 0.5},
                'introspection': {}
            },
            {
                'name': 'Gradient Issues',
                'metrics': {'gradient_norm': 0.001, 'val_confidence': 0.7},
                'introspection': {}
            },
            {
                'name': 'Attention Problems',
                'metrics': {'attention_entropy': 0.5, 'val_confidence': 0.8},
                'introspection': {}
            },
            {
                'name': 'Normal Performance',
                'metrics': {'val_confidence': 0.8, 'val_loss': 0.4, 'gradient_norm': 1.0, 'attention_entropy': 2.0},
                'introspection': {}
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\\nTesting {scenario['name']}:")
            
            # Add some fake loss history for plateau detection
            if i == 1:  # Add plateau for gradient test
                for j in range(6):
                    memory.log_introspection(IntrospectionEntry(step=j, type="val_loss", value=0.75))
            
            spawned_tasks = spawner.analyze_and_spawn(
                current_metrics=scenario['metrics'],
                introspection_data=scenario['introspection'],
                step=i * 10
            )
            
            print(f"  Spawned {len(spawned_tasks)} tasks:")
            for task in spawned_tasks:
                print(f"    - {task.prompt} (priority: {task.priority})")
                print(f"      Weakness: {task.metadata.get('weakness_type', 'unknown')}")
        
        # Test stats
        stats = spawner.get_spawn_stats()
        print(f"\\nSpawning statistics:")
        print(f"  Total spawned: {stats['total_spawned']}")
        print(f"  Weakness distribution: {stats['weakness_distribution']}")
        
        print("TaskSpawner test completed!")

if __name__ == "__main__":
    test_task_spawner()