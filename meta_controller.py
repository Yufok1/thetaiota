#!/usr/bin/env python3
"""
Phase 2/3: MetaController - Learned decision-making with recursive self-analysis.
Replaces hard-coded rules with a neural network that learns optimal strategies.
Enhanced in Phase 3 to read and learn from its own past decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class MetaAction(Enum):
    """Enhanced set of actions the meta-controller can take."""
    CONTINUE_TRAINING = 0
    FINE_TUNE_NOW = 1
    SPAWN_DATA_COLLECTION = 2
    ADJUST_LEARNING_RATE = 3
    PAUSE_AND_ANALYZE = 4
    INCREASE_BATCH_SIZE = 5
    DECREASE_BATCH_SIZE = 6
    SWITCH_OPTIMIZER = 7
    GRADIENT_CLIPPING = 8
    CURRICULUM_ADVANCE = 9
    EMERGENCY_STOP = 10

@dataclass
class MetaObservation:
    """Rich observation state for the enhanced meta-controller."""
    # Core metrics
    val_loss: float
    train_loss: float
    gradient_norm: float
    confidence: float
    
    # Trend analysis
    val_loss_trend: float
    train_loss_trend: float
    gradient_trend: float
    confidence_trend: float
    
    # Training dynamics
    steps_since_improvement: int
    steps_since_last_update: int
    memory_usage: float
    attention_entropy: float
    
    # Advanced metrics
    loss_variance: float
    gradient_variance: float
    learning_rate: float
    batch_size: int
    
    # Historical context
    last_action_reward: float = 0.0
    recent_action_success_rate: float = 0.5
    decision_history_embedding: List[float] = None
    
    # Multi-agent coordination
    agent_coordination_score: float = 0.5
    federation_health: float = 1.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input."""
        base_features = [
            # Core metrics (4)
            self.val_loss, self.train_loss, self.gradient_norm, self.confidence,
            
            # Trend analysis (4)
            self.val_loss_trend, self.train_loss_trend, self.gradient_trend, self.confidence_trend,
            
            # Training dynamics (4)
            float(self.steps_since_improvement), float(self.steps_since_last_update),
            self.memory_usage, self.attention_entropy,
            
            # Advanced metrics (4)
            self.loss_variance, self.gradient_variance, self.learning_rate, float(self.batch_size),
            
            # Historical context (2)
            self.last_action_reward, self.recent_action_success_rate,
            
            # Multi-agent coordination (2)
            self.agent_coordination_score, self.federation_health,
        ]
        
        # Add decision history embedding if available
        if self.decision_history_embedding:
            base_features.extend(self.decision_history_embedding[:20])  # 20 features
        else:
            base_features.extend([0.0] * 20)  # Pad with zeros
            
        return torch.tensor(base_features, dtype=torch.float32)

class MetaControllerNet(nn.Module):
    """
    Enhanced neural network for meta-controller decisions.
    Properly sized for managing large-scale AI systems.
    """
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 512, num_actions: int = 11):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Deep architecture for complex decision making
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize with proper scaling
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits."""
        return self.net(observation)
    
    def get_action_probs(self, observation: torch.Tensor) -> torch.Tensor:
        """Get action probabilities using softmax."""
        logits = self.forward(observation)
        return torch.softmax(logits, dim=-1)
    
    def select_action(self, observation: torch.Tensor, epsilon: float = 0.1) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        Returns (action_id, confidence_score)
        """
        probs = self.get_action_probs(observation)
        
        if np.random.random() < epsilon:
            # Explore: random action
            action = np.random.randint(0, self.num_actions)
            confidence = probs[action].item()
        else:
            # Exploit: best action
            action = torch.argmax(probs).item()
            confidence = probs[action].item()
        
        return action, confidence

class MetaController:
    """
    Phase 2/3 Meta-Controller: Learns when and how to self-improve.
    Enhanced in Phase 3 to read and analyze its own past decisions.
    """
    
    def __init__(self, device: torch.device, learning_rate: float = 1e-3, memory_db=None):
        self.device = device
        self.net = MetaControllerNet().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        # Phase 3: Memory access for reading past decisions
        self.memory_db = memory_db
        
        # Experience buffer for training
        self.experience_buffer = []
        self.max_buffer_size = 2000  # Larger buffer for complex decisions
        
        # Training state
        self.total_steps = 0
        self.last_loss = 0.0
        
        # Action tracking
        self.action_history = []
        self.reward_history = []
        
        # Phase 3: Decision analysis cache
        self.decision_analysis_cache = {}
        self.recent_decision_outcomes = []
        
        # Calculate parameters
        params = sum(p.numel() for p in self.net.parameters())
        print(f"EnhancedMetaController initialized with {params:,} parameters ({params/1e6:.1f}M)")
        print(f"   Input features: 40")
        print(f"   Hidden dimensions: 512")
        print(f"   Actions: 11")
        print(f"   Architecture: 6-layer deep network")
    
    def observe(self, metrics: Dict[str, float], steps_since_improvement: int, 
                steps_since_last_update: int) -> MetaObservation:
        """
        Create rich observation from current agent state including advanced analysis.
        """
        # Calculate trends from recent history
        val_loss_trend = 0.0
        train_loss_trend = 0.0
        gradient_trend = 0.0
        confidence_trend = 0.0
        
        if len(self.reward_history) >= 5:
            recent_losses = [r for r in self.reward_history[-10:]]
            if len(recent_losses) >= 3:
                val_loss_trend = recent_losses[-1] - recent_losses[0]
                train_loss_trend = recent_losses[-1] - recent_losses[-3]
                gradient_trend = recent_losses[-1] - recent_losses[-2]
                confidence_trend = recent_losses[-1] - recent_losses[-5]
        
        # Advanced analysis
        last_action_reward = self.reward_history[-1] if self.reward_history else 0.0
        recent_success_rate = self._calculate_recent_success_rate()
        decision_embedding = self._create_decision_history_embedding()
        
        # Calculate variances
        loss_variance = metrics.get('loss_variance', 0.1)
        gradient_variance = metrics.get('gradient_variance', 0.1)
        
        return MetaObservation(
            # Core metrics
            val_loss=metrics.get('val_loss', 0.5),
            train_loss=metrics.get('train_loss', 0.5),
            gradient_norm=metrics.get('gradient_norm', 1.0),
            confidence=metrics.get('val_confidence', 0.5),
            
            # Trend analysis
            val_loss_trend=val_loss_trend,
            train_loss_trend=train_loss_trend,
            gradient_trend=gradient_trend,
            confidence_trend=confidence_trend,
            
            # Training dynamics
            steps_since_improvement=steps_since_improvement,
            steps_since_last_update=steps_since_last_update,
            memory_usage=metrics.get('memory_usage_gb', 0.1),
            attention_entropy=metrics.get('attention_entropy', 2.0),
            
            # Advanced metrics
            loss_variance=loss_variance,
            gradient_variance=gradient_variance,
            learning_rate=metrics.get('learning_rate', 1e-3),
            batch_size=metrics.get('batch_size', 32),
            
            # Historical context
            last_action_reward=last_action_reward,
            recent_action_success_rate=recent_success_rate,
            decision_history_embedding=decision_embedding,
            
            # Multi-agent coordination
            agent_coordination_score=metrics.get('coordination_score', 0.5),
            federation_health=metrics.get('federation_health', 1.0)
        )
    
    def decide_action(self, observation: MetaObservation, 
                     epsilon: float = 0.1) -> Tuple[MetaAction, float, str]:
        """
        Make a meta-level decision based on current state.
        Returns (action, confidence, reasoning)
        """
        obs_tensor = observation.to_tensor().to(self.device)
        
        with torch.no_grad():
            action_id, confidence = self.net.select_action(obs_tensor, epsilon)
        
        action = MetaAction(action_id)
        
        # Generate reasoning based on observation
        reasoning = self._generate_reasoning(observation, action, confidence)
        
        # Store for training
        self.action_history.append((observation, action_id, confidence))
        
        return action, confidence, reasoning
    
    def _generate_reasoning(self, obs: MetaObservation, action: MetaAction, 
                          confidence: float) -> str:
        """Generate sophisticated human-readable reasoning for the decision."""
        reasons = []
        
        # Analyze multiple factors
        if obs.val_loss > 0.7:
            reasons.append("high validation loss")
        if obs.train_loss > 0.8:
            reasons.append("high training loss")
        if obs.steps_since_improvement > 10:
            reasons.append(f"no improvement for {obs.steps_since_improvement} steps")
        if obs.confidence < 0.6:
            reasons.append("low model confidence")
        if obs.gradient_norm < 0.1:
            reasons.append("very small gradients")
        if obs.gradient_norm > 3.0:
            reasons.append("large gradients")
        if obs.loss_variance > 0.5:
            reasons.append("high loss variance")
        if obs.memory_usage > 0.8:
            reasons.append("high memory usage")
        if obs.agent_coordination_score < 0.3:
            reasons.append("poor agent coordination")
        if obs.federation_health < 0.7:
            reasons.append("federation health concerns")
        
        reason_text = ", ".join(reasons) if reasons else "optimal training conditions"
        
        return f"{action.name} (conf: {confidence:.3f}) due to {reason_text}"
    
    def receive_reward(self, reward: float):
        """
        Receive reward for the last action taken.
        Positive reward = good decision, negative = bad decision.
        """
        if self.action_history:
            # Store experience for training
            last_obs, last_action, _ = self.action_history[-1]
            experience = {
                'observation': last_obs.to_tensor(),
                'action': last_action,
                'reward': reward,
                'step': self.total_steps
            }
            
            self.experience_buffer.append(experience)
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer.pop(0)
            
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:  # Larger history
                self.reward_history.pop(0)
    
    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """
        Train the enhanced meta-controller on recent experience.
        Uses advanced policy gradient with larger batches.
        """
        if len(self.experience_buffer) < batch_size // 2:
            return None
        
        # Sample larger batch for complex learning
        effective_batch = min(batch_size, len(self.experience_buffer))
        indices = np.random.choice(len(self.experience_buffer), effective_batch, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Prepare tensors
        observations = torch.stack([exp['observation'] for exp in batch]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        
        # Advanced reward normalization
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Forward pass
        logits = self.net(observations)
        log_probs = torch.log_softmax(logits, dim=1)
        
        # Policy gradient loss
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -(action_log_probs * rewards).mean()
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.total_steps += 1
        self.last_loss = float(loss.item())
        
        return self.last_loss
    
    def _calculate_recent_success_rate(self) -> float:
        """Phase 3: Calculate success rate of recent decisions."""
        if len(self.reward_history) < 3:
            return 0.5  # Neutral assumption
        
        recent_rewards = self.reward_history[-10:]  # Last 10 decisions
        successful_decisions = sum(1 for r in recent_rewards if r > 0.1)
        return successful_decisions / len(recent_rewards)
    
    def _create_decision_history_embedding(self) -> List[float]:
        """Create embedding of recent decision history."""
        if len(self.action_history) < 5:
            return [0.0] * 20
        
        # Simple embedding: recent action frequencies and reward trends
        recent_actions = [a[1] for a in self.action_history[-20:]]
        recent_rewards = self.reward_history[-20:] if self.reward_history else [0.0] * 20
        
        # Action frequency features (11 actions)
        action_counts = [recent_actions.count(i) for i in range(11)]
        
        # Reward trend features (9 features)
        reward_features = [
            np.mean(recent_rewards),
            np.std(recent_rewards),
            np.max(recent_rewards),
            np.min(recent_rewards),
            recent_rewards[-1] if recent_rewards else 0.0,
            recent_rewards[-5] if len(recent_rewards) >= 5 else 0.0,
            recent_rewards[-10] if len(recent_rewards) >= 10 else 0.0,
            np.sum(1 for r in recent_rewards if r > 0),
            np.sum(1 for r in recent_rewards if r < 0)
        ]
        
        return action_counts + reward_features
    
    def _analyze_decision_from_memory(self, step_range: int = 20) -> Dict[str, Any]:
        """Phase 3: Analyze past decisions from memory database."""
        if not self.memory_db:
            return {}
        
        try:
            cursor = self.memory_db.conn.cursor()
            cursor.execute("""
                SELECT * FROM meta_events 
                WHERE event_type = 'meta_decision' 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (step_range,))
            
            recent_decisions = cursor.fetchall()
            
            if not recent_decisions:
                return {}
            
            # Analyze decision patterns
            import json
            decision_analysis = {
                'total_decisions': len(recent_decisions),
                'action_distribution': {},
                'avg_confidence': 0.0,
                'decision_effectiveness': []
            }
            
            for decision in recent_decisions:
                info = json.loads(decision['info_json'])
                action = info.get('action', 'UNKNOWN')
                confidence = info.get('confidence', 0.0)
                
                # Count actions
                decision_analysis['action_distribution'][action] = \
                    decision_analysis['action_distribution'].get(action, 0) + 1
                
                # Average confidence
                decision_analysis['avg_confidence'] += confidence
            
            if len(recent_decisions) > 0:
                decision_analysis['avg_confidence'] /= len(recent_decisions)
            
            return decision_analysis
            
        except Exception as e:
            print(f"Error analyzing decisions from memory: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        recent_rewards = self.reward_history[-10:] if self.reward_history else []
        avg_recent = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        reward_trend = float(np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])) if len(recent_rewards) >= 5 else 0.0
        return {
            'total_steps': int(self.total_steps),
            'buffer_size': int(len(self.experience_buffer)),
            'avg_recent_reward': avg_recent,
            'reward_trend': reward_trend,
            'experience_count': int(len(self.experience_buffer)),
            'last_loss': float(self.last_loss) if isinstance(self.last_loss, (int, float)) else 0.0,
        }
    
    def explain_policy(self, observation: MetaObservation) -> Dict[str, float]:
        """
        Explain what the current policy would do in this situation.
        Returns action probabilities for interpretability.
        """
        obs_tensor = observation.to_tensor().to(self.device)
        
        with torch.no_grad():
            probs = self.net.get_action_probs(obs_tensor)
        
        action_names = [action.name for action in MetaAction]
        return {name: prob.item() for name, prob in zip(action_names, probs)}

def test_meta_controller():
    """Test the MetaController implementation."""
    print("Testing MetaController...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_controller = MetaController(device)
    
    # Test observation creation
    test_metrics = {
        'val_loss': 0.8,
        'train_loss': 0.7,
        'gradient_norm': 1.2,
        'val_confidence': 0.4,
        'attention_entropy': 2.5,
        'memory_usage_gb': 0.1
    }
    
    obs = meta_controller.observe(test_metrics, steps_since_improvement=3, steps_since_last_update=10)
    print(f"Observation created: val_loss={obs.val_loss}, confidence={obs.confidence}")
    
    # Test decision making
    action, confidence, reasoning = meta_controller.decide_action(obs)
    print(f"Decision: {action.name} (confidence: {confidence:.3f})")
    print(f"Reasoning: {reasoning}")
    
    # Test reward and training
    meta_controller.receive_reward(0.2)  # Slightly positive reward
    
    # Generate some experience
    for i in range(50):
        fake_obs = meta_controller.observe(
            {k: v + np.random.normal(0, 0.1) for k, v in test_metrics.items()},
            steps_since_improvement=np.random.randint(0, 10),
            steps_since_last_update=np.random.randint(5, 20)
        )
        action, _, _ = meta_controller.decide_action(fake_obs)
        reward = np.random.normal(0, 0.3)  # Random reward
        meta_controller.receive_reward(reward)
    
    # Test training
    loss = meta_controller.train_step()
    print(f"Training loss: {loss:.4f}" if loss else "Not enough data for training")
    
    # Test policy explanation
    policy_probs = meta_controller.explain_policy(obs)
    print("Current policy probabilities:")
    for action_name, prob in policy_probs.items():
        print(f"  {action_name}: {prob:.3f}")
    
    # Test stats
    stats = meta_controller.get_stats()
    print(f"Stats: {stats}")
    
    print("MetaController test completed!")

if __name__ == "__main__":
    test_meta_controller()