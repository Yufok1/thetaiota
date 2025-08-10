#!/usr/bin/env python3
"""
Phase 3: Reflection Explainer - Advanced decision explanation and memory analysis.
Makes the agent's reasoning process transparent and queryable.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from memory_db import MemoryDB

@dataclass
class DecisionExplanation:
    """Structured explanation of an agent decision."""
    step: int
    timestamp: float
    decision_type: str
    action_taken: str
    confidence: float
    reasoning: str
    context_factors: Dict[str, Any]
    outcome_metrics: Optional[Dict[str, Any]] = None
    related_events: List[Dict] = None

@dataclass
class ReflectionSummary:
    """Summary of agent behavior over a time period."""
    start_step: int
    end_step: int
    duration_steps: int
    decisions_made: int
    tasks_spawned: int
    performance_trend: str
    key_insights: List[str]
    weakness_patterns: Dict[str, int]
    meta_learning_progress: Dict[str, Any]

class ReflectionExplainer:
    """
    Phase 3: Advanced explanation engine for agent decisions and behavior.
    Provides deep introspection into the agent's reasoning process.
    """
    
    def __init__(self, memory_db: MemoryDB):
        self.memory = memory_db
        self.explanation_cache = {}  # Cache for expensive queries
        
    def explain_last_action(self, context_window: int = 10) -> DecisionExplanation:
        """
        Explain the most recent meta-decision made by the agent.
        """
        # Get the most recent meta-decision
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM meta_events 
            WHERE event_type = 'meta_decision' 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        
        decision_event = cursor.fetchone()
        if not decision_event:
            raise ValueError("No meta-decisions found in memory")
        
        decision_info = json.loads(decision_event['info_json'])
        step = decision_info['step']
        
        # Get context around this decision
        context = self._get_decision_context(step, context_window)
        
        # Analyze the outcome
        outcome_metrics = self._analyze_decision_outcome(step)
        
        return DecisionExplanation(
            step=step,
            timestamp=decision_event['timestamp'],
            decision_type="meta_decision",
            action_taken=decision_info['action'],
            confidence=decision_info['confidence'],
            reasoning=decision_info['reasoning'],
            context_factors=context,
            outcome_metrics=outcome_metrics,
            related_events=self._get_related_events(step, context_window)
        )
    
    def explain_decision_at_step(self, step: int, context_window: int = 5) -> DecisionExplanation:
        """
        Explain a specific decision made at a given step.
        """
        # Find meta-decision near this step
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM meta_events 
            WHERE event_type = 'meta_decision' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (step - 5, step + 5))
        
        decision_event = cursor.fetchone()
        if not decision_event:
            return self._create_no_decision_explanation(step)
        
        decision_info = json.loads(decision_event['info_json'])
        actual_step = decision_info['step']
        
        context = self._get_decision_context(actual_step, context_window)
        outcome_metrics = self._analyze_decision_outcome(actual_step)
        
        return DecisionExplanation(
            step=actual_step,
            timestamp=decision_event['timestamp'],
            decision_type="meta_decision",
            action_taken=decision_info['action'],
            confidence=decision_info['confidence'],
            reasoning=decision_info['reasoning'],
            context_factors=context,
            outcome_metrics=outcome_metrics,
            related_events=self._get_related_events(actual_step, context_window)
        )
    
    def _get_decision_context(self, step: int, window: int) -> Dict[str, Any]:
        """Gather context factors that influenced a decision."""
        context = {}
        
        # Get performance metrics around decision time
        metrics_before = self._get_metrics_window(step - window, step)
        metrics_after = self._get_metrics_window(step, step + window)
        
        if metrics_before:
            context['performance_before'] = {
                'avg_val_loss': float(np.nanmean([m['value'] for m in metrics_before if m['type'] == 'val_loss']) if [m['value'] for m in metrics_before if m['type'] == 'val_loss'] else float('nan')),
                'avg_confidence': float(np.nanmean([m['value'] for m in metrics_before if m['type'] == 'val_confidence']) if [m['value'] for m in metrics_before if m['type'] == 'val_confidence'] else float('nan')),
                'avg_gradient_norm': float(np.nanmean([m['value'] for m in metrics_before if m['type'] == 'gradient_norm']) if [m['value'] for m in metrics_before if m['type'] == 'gradient_norm'] else float('nan'))
            }
        
        if metrics_after:
            context['performance_after'] = {
                'avg_val_loss': float(np.nanmean([m['value'] for m in metrics_after if m['type'] == 'val_loss']) if [m['value'] for m in metrics_after if m['type'] == 'val_loss'] else float('nan')),
                'avg_confidence': float(np.nanmean([m['value'] for m in metrics_after if m['type'] == 'val_confidence']) if [m['value'] for m in metrics_after if m['type'] == 'val_confidence'] else float('nan')),
                'avg_gradient_norm': float(np.nanmean([m['value'] for m in metrics_after if m['type'] == 'gradient_norm']) if [m['value'] for m in metrics_after if m['type'] == 'gradient_norm'] else float('nan'))
            }
        
        # Get recent task spawning activity
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM meta_events 
            WHERE event_type = 'tasks_spawned' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (step - window*2, step))
        
        recent_spawning = cursor.fetchall()
        if recent_spawning:
            context['recent_task_spawning'] = len(recent_spawning)
            spawn_info = [json.loads(event['info_json']) for event in recent_spawning]
            all_weaknesses = []
            for info in spawn_info:
                all_weaknesses.extend(info.get('weaknesses', []))
            context['recent_weaknesses'] = list(set(all_weaknesses))
        
        return context
    
    def _get_metrics_window(self, start_step: int, end_step: int) -> List[Dict]:
        """Get all metrics within a step range."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM introspection 
            WHERE step BETWEEN ? AND ?
            ORDER BY step, timestamp
        """, (start_step, end_step))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _analyze_decision_outcome(self, decision_step: int, look_ahead: int = 10) -> Dict[str, Any]:
        """Analyze the outcome of a decision by looking at subsequent metrics."""
        # Get metrics before and after the decision
        before_metrics = self._get_metrics_window(decision_step - 5, decision_step)
        after_metrics = self._get_metrics_window(decision_step, decision_step + look_ahead)
        
        if not before_metrics or not after_metrics:
            return {"analysis": "insufficient_data"}
        
        outcome = {}
        
        # Analyze val_loss change
        before_loss = [m['value'] for m in before_metrics if m['type'] == 'val_loss']
        after_loss = [m['value'] for m in after_metrics if m['type'] == 'val_loss']
        
        if before_loss and after_loss:
            loss_improvement = np.mean(before_loss) - np.mean(after_loss)
            outcome['loss_change'] = {
                'improvement': loss_improvement,
                'direction': 'improved' if loss_improvement > 0.01 else 'degraded' if loss_improvement < -0.01 else 'stable'
            }
        
        # Analyze confidence change
        before_conf = [m['value'] for m in before_metrics if m['type'] == 'val_confidence']
        after_conf = [m['value'] for m in after_metrics if m['type'] == 'val_confidence']
        
        if before_conf and after_conf:
            conf_change = np.mean(after_conf) - np.mean(before_conf)
            outcome['confidence_change'] = {
                'change': conf_change,
                'direction': 'increased' if conf_change > 0.05 else 'decreased' if conf_change < -0.05 else 'stable'
            }
        
        # Overall assessment
        if outcome.get('loss_change', {}).get('direction') == 'improved':
            outcome['overall_assessment'] = 'positive'
        elif outcome.get('loss_change', {}).get('direction') == 'degraded':
            outcome['overall_assessment'] = 'negative'
        else:
            outcome['overall_assessment'] = 'neutral'
        
        return outcome
    
    def _get_related_events(self, step: int, window: int) -> List[Dict]:
        """Get related events around a decision."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM meta_events 
            WHERE json_extract(info_json, '$.step') BETWEEN ? AND ?
            AND event_type != 'meta_decision'
            ORDER BY timestamp
        """, (step - window, step + window))
        
        events = []
        for row in cursor.fetchall():
            event_data = dict(row)
            event_data['info'] = json.loads(event_data['info_json'])
            events.append(event_data)
        
        return events
    
    def _create_no_decision_explanation(self, step: int) -> DecisionExplanation:
        """Create explanation for when no decision was made."""
        return DecisionExplanation(
            step=step,
            timestamp=0.0,
            decision_type="no_decision",
            action_taken="CONTINUE_TRAINING",
            confidence=0.0,
            reasoning="No meta-decision was made at this step - agent continued normal training",
            context_factors=self._get_decision_context(step, 5),
            related_events=[]
        )
    
    def create_behavior_summary(self, start_step: int, end_step: int) -> ReflectionSummary:
        """
        Create a comprehensive summary of agent behavior over a step range.
        """
        # Count decisions made
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM meta_events 
            WHERE event_type = 'meta_decision' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (start_step, end_step))
        
        decisions_made = cursor.fetchone()[0]
        
        # Count tasks spawned
        cursor.execute("""
            SELECT COUNT(*) FROM meta_events 
            WHERE event_type = 'tasks_spawned' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (start_step, end_step))
        
        tasks_spawned = cursor.fetchone()[0]
        
        # Analyze performance trend
        performance_trend = self._analyze_performance_trend(start_step, end_step)
        
        # Extract key insights
        key_insights = self._extract_key_insights(start_step, end_step)
        
        # Analyze weakness patterns
        weakness_patterns = self._analyze_weakness_patterns(start_step, end_step)
        
        # Meta-learning progress
        meta_progress = self._analyze_meta_learning_progress(start_step, end_step)
        
        return ReflectionSummary(
            start_step=start_step,
            end_step=end_step,
            duration_steps=end_step - start_step,
            decisions_made=decisions_made,
            tasks_spawned=tasks_spawned,
            performance_trend=performance_trend,
            key_insights=key_insights,
            weakness_patterns=weakness_patterns,
            meta_learning_progress=meta_progress
        )
    
    def _analyze_performance_trend(self, start_step: int, end_step: int) -> str:
        """Analyze overall performance trend over the period."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT value, step FROM introspection 
            WHERE type = 'val_loss' 
            AND step BETWEEN ? AND ?
            ORDER BY step
        """, (start_step, end_step))
        
        losses = [(row[0], row[1]) for row in cursor.fetchall()]
        
        if len(losses) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        early_losses = [loss for loss, step in losses[:len(losses)//3]]
        late_losses = [loss for loss, step in losses[-len(losses)//3:]]
        
        if not early_losses or not late_losses:
            return "insufficient_data"
        
        early_avg = np.mean(early_losses)
        late_avg = np.mean(late_losses)
        
        improvement = early_avg - late_avg
        
        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "degrading"
        else:
            return "stable"
    
    def _extract_key_insights(self, start_step: int, end_step: int) -> List[str]:
        """Extract key behavioral insights from the period."""
        insights = []
        
        # Analyze decision frequency
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT json_extract(info_json, '$.action') as action, COUNT(*) as count
            FROM meta_events 
            WHERE event_type = 'meta_decision' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
            GROUP BY action
            ORDER BY count DESC
        """, (start_step, end_step))
        
        action_counts = cursor.fetchall()
        
        if action_counts:
            most_common_action = action_counts[0][0]
            action_count = action_counts[0][1]
            insights.append(f"Most frequent action: {most_common_action} ({action_count} times)")
        
        # Check for curriculum advancement
        cursor.execute("""
            SELECT COUNT(*) FROM meta_events 
            WHERE event_type = 'curriculum_advanced' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (start_step, end_step))
        
        curriculum_advances = cursor.fetchone()[0]
        if curriculum_advances > 0:
            insights.append(f"Curriculum advanced {curriculum_advances} times")
        
        # Analyze meta-controller confidence trends
        cursor.execute("""
            SELECT AVG(CAST(json_extract(info_json, '$.confidence') AS REAL)) as avg_confidence
            FROM meta_events 
            WHERE event_type = 'meta_decision' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (start_step, end_step))
        
        avg_confidence = cursor.fetchone()[0]
        if avg_confidence is not None:
            if avg_confidence > 0.7:
                insights.append("High meta-controller confidence")
            elif avg_confidence < 0.3:
                insights.append("Low meta-controller confidence - may need more training")
        
        return insights
    
    def _analyze_weakness_patterns(self, start_step: int, end_step: int) -> Dict[str, int]:
        """Analyze patterns in detected weaknesses."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT info_json FROM meta_events 
            WHERE event_type = 'tasks_spawned' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
        """, (start_step, end_step))
        
        weakness_counts = {}
        for row in cursor.fetchall():
            info = json.loads(row[0])
            weaknesses = info.get('weaknesses', [])
            for weakness in weaknesses:
                weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        return weakness_counts
    
    def _analyze_meta_learning_progress(self, start_step: int, end_step: int) -> Dict[str, Any]:
        """Analyze progress in meta-learning capabilities."""
        # This would analyze how the meta-controller's decisions are improving over time
        # For now, return basic stats
        
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT json_extract(info_json, '$.confidence') as confidence
            FROM meta_events 
            WHERE event_type = 'meta_decision' 
            AND json_extract(info_json, '$.step') BETWEEN ? AND ?
            ORDER BY json_extract(info_json, '$.step')
        """, (start_step, end_step))
        
        confidences = [float(row[0]) for row in cursor.fetchall() if row[0] is not None]
        
        if not confidences:
            return {"status": "no_data"}
        
        # Check if confidence is trending upward
        if len(confidences) >= 4:
            early_conf = np.mean(confidences[:len(confidences)//2])
            late_conf = np.mean(confidences[len(confidences)//2:])
            
            return {
                "confidence_trend": "improving" if late_conf > early_conf + 0.05 else "stable",
                "avg_confidence": np.mean(confidences),
                "confidence_range": [min(confidences), max(confidences)],
                "decision_count": len(confidences)
            }
        
        return {
            "avg_confidence": np.mean(confidences),
            "decision_count": len(confidences)
        }
    
    def query_agent_memory(self, query: str) -> Dict[str, Any]:
        """
        Natural language-style queries about agent memory.
        This is a simple pattern-matching approach for Phase 3.
        """
        query_lower = query.lower()
        
        if "last decision" in query_lower or "recent decision" in query_lower:
            explanation = self.explain_last_action()
            return {
                "query_type": "last_decision",
                "result": explanation
            }
        
        elif "task" in query_lower and "spawn" in query_lower:
            cursor = self.memory.conn.cursor()
            cursor.execute("""
                SELECT * FROM meta_events 
                WHERE event_type = 'tasks_spawned' 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            recent_spawning = [dict(row) for row in cursor.fetchall()]
            return {
                "query_type": "task_spawning",
                "result": recent_spawning
            }
        
        elif "performance" in query_lower or "how am i doing" in query_lower:
            # Get recent performance summary
            cursor = self.memory.conn.cursor()
            cursor.execute("SELECT MAX(step) FROM introspection")
            max_step = cursor.fetchone()[0] or 0
            
            if max_step > 50:
                summary = self.create_behavior_summary(max_step - 50, max_step)
                return {
                    "query_type": "performance_summary",
                    "result": summary
                }
        
        elif "weakness" in query_lower:
            weakness_patterns = self._analyze_weakness_patterns(0, 999999)  # All time
            return {
                "query_type": "weakness_analysis",
                "result": weakness_patterns
            }
        
        return {
            "query_type": "unknown",
            "result": "I don't understand that query yet. Try asking about 'last decision', 'performance', 'tasks', or 'weaknesses'."
        }

def test_reflection_explainer():
    """Test the reflection explainer with Phase 2 data."""
    print("Testing Reflection Explainer...")
    
    # Use existing Phase 2 database
    try:
        with MemoryDB("phase2_demo.db") as memory:
            explainer = ReflectionExplainer(memory)
            
            print("\\n1. Testing last action explanation:")
            try:
                last_explanation = explainer.explain_last_action()
                print(f"   Action: {last_explanation.action_taken}")
                print(f"   Reasoning: {last_explanation.reasoning}")
                print(f"   Outcome: {last_explanation.outcome_metrics}")
            except ValueError as e:
                print(f"   {e}")
            
            print("\\n2. Testing behavior summary:")
            try:
                summary = explainer.create_behavior_summary(0, 100)
                print(f"   Decisions made: {summary.decisions_made}")
                print(f"   Tasks spawned: {summary.tasks_spawned}")
                print(f"   Performance trend: {summary.performance_trend}")
                print(f"   Key insights: {summary.key_insights}")
            except Exception as e:
                print(f"   Error: {e}")
            
            print("\\n3. Testing memory queries:")
            queries = [
                "What was my last decision?",
                "How is my performance?",
                "What tasks have I spawned?",
                "What are my weaknesses?"
            ]
            
            for query in queries:
                try:
                    result = explainer.query_agent_memory(query)
                    print(f"   Q: {query}")
                    print(f"   A: {result['query_type']} -> {type(result['result']).__name__}")
                except Exception as e:
                    print(f"   Q: {query} -> Error: {e}")
            
    except Exception as e:
        print(f"Could not access Phase 2 database: {e}")
        print("Run phase2_agent.py first to generate data")
    
    print("Reflection Explainer test completed!")

if __name__ == "__main__":
    test_reflection_explainer()