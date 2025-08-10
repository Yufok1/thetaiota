#!/usr/bin/env python3
"""
Phase 3: Self-Aware AI Agent - Complete integrated system with reflection capabilities.
Combines Phase 2 capabilities with advanced self-awareness and interactive introspection.
"""

import torch
import time
import os
from typing import Dict, List, Any, Optional

# Import Phase 2 components
from phase2_agent import Phase2Agent
from memory_db import MemoryDB

# Import Phase 3 components
from reflection_explainer import ReflectionExplainer
from memory_summarizer import MemorySummarizer, ReflectionArtifact
from phase3_shell import AgentShell
from human_feedback_system import HumanFeedbackSystem, FeedbackType, FeedbackSentiment

class Phase3Agent(Phase2Agent):
    """
    Phase 3: Self-Aware AI Agent
    Features:
    - All Phase 2 capabilities (meta-learning, task spawning, curriculum)
    - Advanced decision explanation and memory analysis
    - Automatic reflection creation and summarization
    - Interactive introspection interface
    - Self-awareness through persistent memory summaries
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize Phase 2 base
        super().__init__(config)
        
        print(f"Initializing Phase 3 Self-Awareness components...")
        
        # Initialize Phase 3 components
        self._init_reflection_system()
        self._init_human_feedback_system()
        
        # Phase 3 configuration
        self.reflection_frequency = config.get('reflection_frequency', 50)
        self.auto_summarization = config.get('auto_summarization', True)
        self.human_feedback_enabled = config.get('human_feedback_enabled', True)
        
        print(f"Phase 3 Agent initialized - Enhanced with self-awareness capabilities")
    
    def _init_reflection_system(self):
        """Initialize the reflection and introspection system."""
        self.explainer = ReflectionExplainer(self.memory)
        self.summarizer = MemorySummarizer(self.memory)
        
        # Track reflection creation
        self.last_reflection_step = 0
        self.reflection_artifacts = []
    
    def _init_human_feedback_system(self):
        """Initialize the human feedback integration system."""
        self.feedback_system = HumanFeedbackSystem(self.memory)
        self.feedback_interface = self.feedback_system.create_feedback_interface()
        
        # Feedback integration state
        self.last_feedback_processing_step = 0
        self.feedback_processing_frequency = 25  # Process feedback every 25 steps
    
    def enhanced_meta_decision_cycle(self, current_metrics: Dict[str, float]) -> bool:
        """
        Phase 3: Enhanced meta-decision making with self-reflection.
        """
        # Standard Phase 2 meta-decision making
        meta_action_taken = super().meta_decision_cycle(current_metrics)
        
        # Phase 3: Add self-reflection after significant decisions
        if meta_action_taken:
            try:
                # Explain the decision that was just made
                explanation = self.explainer.explain_last_action(context_window=15)
                
                # Log enhanced decision explanation
                self.memory.log_meta_event(
                    event_type="enhanced_decision_explanation",
                    info={
                        "step": self.global_step,
                        "decision_type": explanation.decision_type,
                        "action": explanation.action_taken,
                        "confidence": explanation.confidence,
                        "reasoning": explanation.reasoning,
                        "context_summary": len(explanation.context_factors),
                        "outcome_assessment": explanation.outcome_metrics.get('overall_assessment', 'unknown')
                    }
                )
                
                print(f"    Self-reflection: {explanation.reasoning}")
                if explanation.outcome_metrics:
                    assessment = explanation.outcome_metrics.get('overall_assessment', 'neutral')
                    print(f"    Outcome assessment: {assessment}")
                
            except Exception as e:
                print(f"    Self-reflection error: {e}")
        
        return meta_action_taken
    
    def periodic_reflection_cycle(self) -> List[ReflectionArtifact]:
        """
        Phase 3: Create periodic reflections and summaries.
        """
        created_reflections = []
        
        if self.global_step - self.last_reflection_step >= self.reflection_frequency:
            print(f"  Creating periodic reflections (step {self.global_step})...")
            
            try:
                # Check for automatic summary creation opportunities
                auto_summaries = self.summarizer.check_and_create_summaries()
                
                if auto_summaries:
                    print(f"    Created {len(auto_summaries)} automatic summaries")
                    for summary in auto_summaries:
                        print(f"      - {summary.summary_type}: {summary.trigger}")
                        if summary.insights:
                            print(f"        Key insight: {summary.insights[0]}")
                    
                    created_reflections.extend(auto_summaries)
                    self.reflection_artifacts.extend(auto_summaries)
                
                self.last_reflection_step = self.global_step
                
                # Create a comprehensive reflection report periodically
                if self.global_step % (self.reflection_frequency * 2) == 0:
                    reflection_report = self.summarizer.generate_reflection_report()
                    print(f"    Generated reflection report ({len(reflection_report.split())} words)")
                
            except Exception as e:
                print(f"    Reflection error: {e}")
        
        return created_reflections
    
    def process_human_feedback_cycle(self) -> Dict[str, Any]:
        """
        Phase 3: Process accumulated human feedback and integrate into learning.
        """
        if not self.human_feedback_enabled:
            return {}
        
        if self.global_step - self.last_feedback_processing_step < self.feedback_processing_frequency:
            return {}
        
        # Only log when there is actual feedback to process
        # (silences noisy logs when queue is empty)
        # Will print details below if processed_count > 0
        
        try:
            # Process unprocessed feedback
            processing_results = self.feedback_system.process_feedback_for_learning(self.meta_controller)
            
            if processing_results.get("processed_count", 0) > 0:
                print(f"  Processing human feedback (step {self.global_step})...")
                print(f"    Processed {processing_results['processed_count']} feedback entries")
                print(f"    Average sentiment: {processing_results['avg_sentiment']:.2f}")
                
                # Apply feedback-based reward adjustments
                for adjustment in processing_results["reward_adjustments"]:
                    reward = adjustment["reward"]
                    feedback_type = adjustment["type"]
                    
                    # Integrate feedback reward into meta-controller learning
                    if hasattr(self.meta_controller, 'reward_history'):
                        self.meta_controller.reward_history.append(reward * 0.5)  # Scale down
                    
                    print(f"      Applied {feedback_type} reward: {reward:.3f}")
                
                # Log suggestions for future improvement
                if processing_results["suggested_actions"]:
                    print(f"    Received {len(processing_results['suggested_actions'])} action suggestions")
            
            self.last_feedback_processing_step = self.global_step
            return processing_results
            
        except Exception as e:
            print(f"    Human feedback processing error: {e}")
            return {}
    
    def request_feedback_for_decision(self, decision_details: Dict[str, Any]) -> str:
        """
        Request human feedback for a specific decision.
        """
        if not self.human_feedback_enabled:
            return ""
        
        context = {
            "step": self.global_step,
            "action": decision_details.get("action", "unknown"),
            "reasoning": decision_details.get("reasoning", ""),
            "confidence": decision_details.get("confidence", 0.0),
            "outcome_assessment": decision_details.get("outcome_assessment", "unknown")
        }
        
        request_id = self.feedback_system.request_feedback(
            request_type="decision_review",
            context=context,
            target_step=self.global_step
        )
        
        return request_id
    
    def simulate_human_feedback(self, step: int, feedback_type: str = "positive"):
        """
        Simulate human feedback for testing purposes.
        """
        if not self.human_feedback_enabled:
            return
        
        feedback_map = {
            "positive": (FeedbackType.DECISION_APPROVAL, FeedbackSentiment.POSITIVE, 
                        "Good decision, this improved performance", 4.0),
            "negative": (FeedbackType.DECISION_APPROVAL, FeedbackSentiment.NEGATIVE,
                        "Poor decision, this hurt performance", 2.0),
            "neutral": (FeedbackType.PERFORMANCE_RATING, FeedbackSentiment.NEUTRAL,
                       "Performance is adequate, room for improvement", 3.0),
            "correction": (FeedbackType.BEHAVIOR_CORRECTION, FeedbackSentiment.NEGATIVE,
                          "Should have spawned more tasks", 2.5)
        }
        
        if feedback_type not in feedback_map:
            feedback_type = "positive"
        
        fb_type, sentiment, content, rating = feedback_map[feedback_type]
        
        return self.feedback_system.submit_feedback(
            feedback_type=fb_type,
            sentiment=sentiment,
            content=content,
            rating=rating,
            target_step=step
        )
    
    def crisis_detection_and_response(self, current_metrics: Dict[str, float]) -> bool:
        """
        Phase 3: Detect crisis conditions and create crisis reflections.
        """
        crisis_indicators = {}
        
        # Detect various crisis conditions
        if current_metrics.get('val_loss', 0) > 1.0:
            crisis_indicators['high_validation_loss'] = current_metrics['val_loss']
        
        if current_metrics.get('val_confidence', 1.0) < 0.3:
            crisis_indicators['low_confidence'] = current_metrics['val_confidence']
        
        if self.steps_since_improvement > 15:
            crisis_indicators['performance_plateau'] = self.steps_since_improvement
        
        if current_metrics.get('gradient_norm', 0) > 5.0:
            crisis_indicators['gradient_explosion'] = current_metrics['gradient_norm']
        
        # If crisis detected, create crisis reflection
        if crisis_indicators:
            print(f"  Crisis detected: {list(crisis_indicators.keys())}")
            
            try:
                crisis_reflection = self.summarizer.create_crisis_summary(crisis_indicators)
                
                print(f"    Created crisis reflection with {len(crisis_reflection.action_recommendations)} recommendations")
                for i, action in enumerate(crisis_reflection.action_recommendations[:2]):
                    print(f"      {i+1}. {action}")
                
                self.reflection_artifacts.append(crisis_reflection)
                return True
                
            except Exception as e:
                print(f"    Crisis reflection error: {e}")
        
        return False
    
    def train(self, num_epochs: int = 5):
        """
        Phase 3: Enhanced training loop with self-awareness features.
        """
        print(f"Starting Phase 3 self-aware training for {num_epochs} epochs...")
        print(f"Features: reflection every {self.reflection_frequency} steps, crisis detection, auto-summarization")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} (Level: {self.curriculum.current_level.name})")
            
            epoch_train_loss = 0.0
            num_batches = 0
            
            # Training loop
            for batch_x, batch_y in self.train_loader:
                batch_metrics = self.train_batch(batch_x, batch_y)
                epoch_train_loss += batch_metrics['train_loss']
                num_batches += 1
                self.global_step += 1
                
                # Periodic validation and enhanced meta-decisions
                if self.global_step % 25 == 0:
                    val_metrics = self.validate()
                    
                    # Combine metrics for decision making
                    combined_metrics = {**batch_metrics, **val_metrics}
                    
                    print(f"  Step {self.global_step}: "
                          f"train_loss={batch_metrics['train_loss']:.4f}, "
                          f"val_loss={val_metrics['val_loss']:.4f}, "
                          f"val_acc={val_metrics['val_accuracy']:.3f}")
                    
                    # Phase 3: Enhanced meta-decision making with self-reflection
                    meta_action_taken = self.enhanced_meta_decision_cycle(combined_metrics)
                    
                    # Phase 3: Crisis detection and response
                    crisis_detected = self.crisis_detection_and_response(combined_metrics)
                    
                    # Phase 3: Process human feedback
                    feedback_results = self.process_human_feedback_cycle()
                    
                    # Phase 2: Task spawning and execution (from parent class)
                    self._spawn_and_execute_tasks()
                    
                    # Phase 2: Curriculum advancement check (from parent class)
                    self._check_curriculum_advancement(val_metrics['val_accuracy'])
                    
                    # Phase 3: Periodic reflection cycle
                    if self.auto_summarization:
                        self.periodic_reflection_cycle()
            
            # End of epoch summary with reflection
            avg_train_loss = epoch_train_loss / num_batches
            final_val_metrics = self.validate()
            
            print(f"  Epoch summary: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={final_val_metrics['val_loss']:.4f}, "
                  f"val_acc={final_val_metrics['val_accuracy']:.3f}")
            # Save checkpoint via Phase 2 helper
            try:
                self._save_checkpoint()
            except Exception:
                pass
            
            # Create epoch milestone reflection
            epoch_metrics = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "final_val_accuracy": final_val_metrics['val_accuracy']
            }
            
            try:
                milestone_reflection = self.summarizer.create_milestone_summary(
                    f"epoch_{epoch+1}", epoch_metrics
                )
                if milestone_reflection:
                    print(f"    Created epoch milestone reflection")
            except Exception as e:
                print(f"    Milestone reflection error: {e}")
        
        # Final comprehensive analysis
        self._print_phase3_training_summary()
    
    def _print_phase3_training_summary(self):
        """Print comprehensive Phase 3 training summary."""
        print(f"\n=== PHASE 3 SELF-AWARE TRAINING SUMMARY ===")
        
        # Phase 2 summary (from parent class)
        super()._print_training_summary()
        
        # Phase 3 specific summary
        print(f"Self-Awareness Features:")
        print(f"  Reflection artifacts created: {len(self.reflection_artifacts)}")
        
        if self.reflection_artifacts:
            type_counts = {}
            for artifact in self.reflection_artifacts:
                type_counts[artifact.summary_type] = type_counts.get(artifact.summary_type, 0) + 1
            
            print(f"  Reflection types: {dict(type_counts)}")
            
            # Show most recent insights
            if self.reflection_artifacts:
                latest = self.reflection_artifacts[-1]
                print(f"  Latest reflection ({latest.summary_type}): {latest.trigger}")
                if latest.insights:
                    print(f"    Key insight: {latest.insights[0]}")
        
        # Human feedback summary
        if self.human_feedback_enabled:
            try:
                feedback_summary = self.feedback_system.get_feedback_summary()
                print(f"Human Feedback Integration:")
                print(f"  Total feedback received: {feedback_summary['total_feedback']}")
                print(f"  Pending feedback requests: {feedback_summary['pending_requests']}")
                if feedback_summary['recent_sentiment']:
                    print(f"  Recent sentiment: {dict(feedback_summary['recent_sentiment'])}")
            except Exception as e:
                print(f"  Human feedback summary error: {e}")
        
        # Generate final comprehensive reflection report
        try:
            final_report = self.summarizer.generate_reflection_report()
            print(f"\nComprehensive Reflection Report:")
            print("=" * 50)
            # Show first few lines of the report
            report_lines = final_report.split("\n")
            for line in report_lines[:15]:  # Show first 15 lines
                print(line)
            if len(report_lines) > 15:
                print(f"... ({len(report_lines) - 15} more lines)")
        except Exception as e:
            print(f"Final report generation error: {e}")
        
        print("=== Phase 3 Complete! ===")
    
    def start_interactive_session(self):
        """Start an interactive introspection session."""
        print(f"\nStarting interactive introspection session...")
        print(f"You can query the agent's memory, decisions, and self-reflections.")
        
        shell = AgentShell(self.memory.db_path)
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print("\nInteractive session ended")

def main():
    """Run Phase 3 demonstration."""
    print("=== PHASE 3: SELF-AWARE AI AGENT ===\n")
    
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
        'db_path': 'phase3_demo.db',
        'reflection_frequency': 40,  # Create reflections every 40 steps
        'auto_summarization': True
    }
    
    # Create and train Phase 3 agent
    agent = Phase3Agent(config)
    agent.train(num_epochs=3)
    
    # Optionally start interactive session
    print(f"\nTraining complete! You can now:")
    print(f"1. Run 'python phase3_shell.py' to start interactive introspection")
    print(f"2. Examine the reflection artifacts in the database")
    print(f"3. Continue training or advance to Phase 4")

if __name__ == "__main__":
    main()