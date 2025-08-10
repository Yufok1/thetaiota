#!/usr/bin/env python3
"""
Phase 3: Memory Summarizer - Creates time-windowed summaries and reflection artifacts.
Extends the database schema with a reflections table for persistent summaries.
"""

import sqlite3
import json
import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from memory_db import MemoryDB
from reflection_explainer import ReflectionExplainer, ReflectionSummary

@dataclass
class ReflectionArtifact:
    """A saved reflection summary with metadata."""
    id: str
    start_step: int
    end_step: int
    summary_type: str  # "periodic", "milestone", "crisis", "success"
    created_at: float
    trigger: str  # What caused this reflection to be created
    summary_data: Dict[str, Any]
    insights: List[str]
    action_recommendations: List[str]

class MemorySummarizer:
    """
    Phase 3: Creates and manages reflective summaries of agent behavior.
    Implements the reflections database schema and time-windowed analysis.
    """
    
    def __init__(self, memory_db: MemoryDB):
        self.memory = memory_db
        self.explainer = ReflectionExplainer(memory_db)
        
        # Initialize reflections table
        self._init_reflections_schema()
        
        # Configuration
        self.summary_window_size = 50  # Steps per summary
        self.milestone_thresholds = {
            'steps': [100, 250, 500, 1000],
            'decisions': [10, 25, 50, 100],
            'accuracy_improvement': 0.1
        }
        
    def _init_reflections_schema(self):
        """Add reflections table to existing schema."""
        cursor = self.memory.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id              TEXT PRIMARY KEY,
                start_step      INTEGER NOT NULL,
                end_step        INTEGER NOT NULL,
                summary_type    TEXT NOT NULL,
                created_at      REAL NOT NULL,
                trigger         TEXT NOT NULL,
                summary_json    TEXT NOT NULL,
                insights_json   TEXT NOT NULL,
                actions_json    TEXT NOT NULL
            );
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_step_range ON reflections(start_step, end_step);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(summary_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);")
        
        self.memory.conn.commit()
        
        # Log schema update
        self.memory.log_meta_event(
            event_type="schema_updated",
            info={"table": "reflections", "purpose": "persistent_reflection_summaries"}
        )
    
    def create_periodic_summary(self, force_creation: bool = False) -> Optional[ReflectionArtifact]:
        """
        Create a periodic summary if enough new activity has occurred.
        """
        # Get current progress
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT MAX(step) FROM introspection")
        current_step = cursor.fetchone()[0] or 0
        
        if current_step < 20 and not force_creation:
            return None
        
        # Check if we need a new summary
        cursor.execute("""
            SELECT MAX(end_step) FROM reflections 
            WHERE summary_type = 'periodic'
        """)
        
        last_summary_step = cursor.fetchone()[0] or 0
        
        if current_step - last_summary_step < self.summary_window_size and not force_creation:
            return None
        
        # Create summary
        start_step = last_summary_step
        end_step = current_step
        
        behavioral_summary = self.explainer.create_behavior_summary(start_step, end_step)
        
        # Generate insights
        insights = self._generate_insights(behavioral_summary, start_step, end_step)
        
        # Generate action recommendations
        actions = self._generate_action_recommendations(behavioral_summary, insights)
        
        # Create reflection artifact
        artifact = ReflectionArtifact(
            id=str(uuid.uuid4()),
            start_step=start_step,
            end_step=end_step,
            summary_type="periodic",
            created_at=time.time(),
            trigger=f"periodic_summary_at_step_{current_step}",
            summary_data=asdict(behavioral_summary),
            insights=insights,
            action_recommendations=actions
        )
        
        # Save to database
        self._save_reflection_artifact(artifact)
        
        return artifact
    
    def create_milestone_summary(self, milestone_type: str, current_metrics: Dict[str, Any]) -> Optional[ReflectionArtifact]:
        """
        Create a summary triggered by reaching a milestone.
        """
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT MAX(step) FROM introspection")
        current_step = cursor.fetchone()[0] or 0
        
        # Check if this milestone was already recorded
        cursor.execute("""
            SELECT COUNT(*) FROM reflections 
            WHERE summary_type = 'milestone' 
            AND trigger LIKE ?
        """, (f"%{milestone_type}%",))
        
        existing_count = cursor.fetchone()[0]
        
        # Create milestone summary
        start_step = max(0, current_step - 100)  # Look back 100 steps
        end_step = current_step
        
        behavioral_summary = self.explainer.create_behavior_summary(start_step, end_step)
        
        # Milestone-specific insights
        insights = [f"Reached milestone: {milestone_type}"]
        insights.extend(self._generate_insights(behavioral_summary, start_step, end_step))
        
        # Add milestone-specific analysis
        if milestone_type == "accuracy_improvement":
            insights.append(f"Significant accuracy improvement detected: {current_metrics.get('improvement', 0):.3f}")
        elif milestone_type == "steps":
            insights.append(f"Training milestone: {current_step} steps completed")
        elif milestone_type == "decisions":
            insights.append(f"Decision-making milestone: {behavioral_summary.decisions_made} decisions in recent period")
        
        actions = self._generate_action_recommendations(behavioral_summary, insights)
        
        artifact = ReflectionArtifact(
            id=str(uuid.uuid4()),
            start_step=start_step,
            end_step=end_step,
            summary_type="milestone",
            created_at=time.time(),
            trigger=f"milestone_{milestone_type}_at_step_{current_step}",
            summary_data=asdict(behavioral_summary),
            insights=insights,
            action_recommendations=actions
        )
        
        self._save_reflection_artifact(artifact)
        
        return artifact
    
    def create_crisis_summary(self, crisis_indicators: Dict[str, Any]) -> ReflectionArtifact:
        """
        Create a summary when crisis indicators are detected.
        """
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT MAX(step) FROM introspection")
        current_step = cursor.fetchone()[0] or 0
        
        # Crisis summaries look at recent activity
        start_step = max(0, current_step - 30)
        end_step = current_step
        
        behavioral_summary = self.explainer.create_behavior_summary(start_step, end_step)
        
        # Crisis-specific insights
        insights = ["Crisis indicators detected:"]
        for indicator, value in crisis_indicators.items():
            insights.append(f"  • {indicator}: {value}")
        
        insights.extend(self._generate_insights(behavioral_summary, start_step, end_step))
        
        # Crisis-specific recommendations
        actions = ["Immediate actions recommended:"]
        if "performance_degradation" in crisis_indicators:
            actions.append("  • Revert to previous checkpoint if available")
            actions.append("  • Increase meta-controller learning rate")
            actions.append("  • Spawn additional fine-tuning tasks")
        
        if "repeated_failures" in crisis_indicators:
            actions.append("  • Analyze error patterns more deeply")
            actions.append("  • Consider curriculum regression to easier examples")
        
        actions.extend(self._generate_action_recommendations(behavioral_summary, insights))
        
        artifact = ReflectionArtifact(
            id=str(uuid.uuid4()),
            start_step=start_step,
            end_step=end_step,
            summary_type="crisis",
            created_at=time.time(),
            trigger=f"crisis_detected_step_{current_step}",
            summary_data=asdict(behavioral_summary),
            insights=insights,
            action_recommendations=actions
        )
        
        self._save_reflection_artifact(artifact)
        
        return artifact
    
    def _generate_insights(self, summary: ReflectionSummary, start_step: int, end_step: int) -> List[str]:
        """Generate insights from a behavioral summary."""
        insights = []
        
        # Performance insights
        if summary.performance_trend == "improving":
            insights.append("Performance is trending upward - current strategy is effective")
        elif summary.performance_trend == "degrading":
            insights.append("Performance is declining - intervention may be needed")
        else:
            insights.append("Performance is stable - consider advancing difficulty or changing approach")
        
        # Decision-making insights
        if summary.decisions_made == 0:
            insights.append("No meta-decisions made - meta-controller may be too conservative")
        elif summary.decisions_made > (end_step - start_step) / 10:
            insights.append("High decision frequency - meta-controller may be too reactive")
        
        # Task spawning insights
        if summary.tasks_spawned == 0:
            insights.append("No tasks spawned - weakness detection may need tuning")
        elif summary.tasks_spawned > 5:
            insights.append("High task spawning rate - may indicate systematic issues")
        
        # Weakness pattern insights
        if summary.weakness_patterns:
            most_common = max(summary.weakness_patterns, key=summary.weakness_patterns.get)
            count = summary.weakness_patterns[most_common]
            insights.append(f"Primary weakness pattern: {most_common} ({count} occurrences)")
            
            if count > 3:
                insights.append(f"Persistent {most_common} issues suggest deeper architectural problems")
        
        # Meta-learning insights
        meta_progress = summary.meta_learning_progress
        if meta_progress.get('confidence_trend') == 'improving':
            insights.append("Meta-controller confidence is improving - learning from experience")
        elif meta_progress.get('avg_confidence', 0) < 0.3:
            insights.append("Low meta-controller confidence - may need more training data or adjustment")
        
        return insights
    
    def _generate_action_recommendations(self, summary: ReflectionSummary, insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on summary and insights."""
        actions = []
        
        # Performance-based actions
        if summary.performance_trend == "degrading":
            actions.append("Consider reverting recent changes or reducing learning rate")
            actions.append("Spawn additional fine-tuning tasks with diverse examples")
        elif summary.performance_trend == "stable" and summary.decisions_made < 2:
            actions.append("Increase meta-controller sensitivity to trigger more adaptive behavior")
        
        # Task-based actions
        if summary.tasks_spawned == 0:
            actions.append("Lower weakness detection thresholds to increase task spawning")
        
        # Weakness-specific actions
        if "overfitting" in str(summary.weakness_patterns):
            actions.append("Increase regularization or expand dataset diversity")
        
        if "low_confidence" in str(summary.weakness_patterns):
            actions.append("Gather more training examples for difficult cases")
        
        if "gradient" in str(summary.weakness_patterns):
            actions.append("Adjust learning rate or optimizer parameters")
        
        # Meta-learning actions
        meta_progress = summary.meta_learning_progress
        if meta_progress.get('avg_confidence', 0) < 0.4:
            actions.append("Increase meta-controller training frequency")
            actions.append("Provide more explicit reward signals for meta-decisions")
        
        # Curriculum actions
        if summary.performance_trend == "improving" and summary.decisions_made > 0:
            actions.append("Consider advancing curriculum difficulty level")
        
        return actions
    
    def _save_reflection_artifact(self, artifact: ReflectionArtifact):
        """Save a reflection artifact to the database."""
        cursor = self.memory.conn.cursor()
        
        # Idempotent insert: skip if a reflection with same type+trigger already exists
        cursor.execute("""
            SELECT COUNT(*) FROM reflections WHERE summary_type = ? AND trigger = ?
        """, (artifact.summary_type, artifact.trigger))
        if (cursor.fetchone()[0] or 0) > 0:
            return
        cursor.execute(
            """
            INSERT INTO reflections 
            (id, start_step, end_step, summary_type, created_at, trigger, 
             summary_json, insights_json, actions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.start_step,
                artifact.end_step,
                artifact.summary_type,
                artifact.created_at,
                artifact.trigger,
                json.dumps(artifact.summary_data),
                json.dumps(artifact.insights),
                json.dumps(artifact.action_recommendations),
            ),
        )
        
        self.memory.conn.commit()
        
        # Log the reflection creation
        self.memory.log_meta_event(
            event_type="reflection_created",
            info={
                "reflection_id": artifact.id,
                "summary_type": artifact.summary_type,
                "step_range": [artifact.start_step, artifact.end_step],
                "insights_count": len(artifact.insights),
                "actions_count": len(artifact.action_recommendations)
            }
        )
    
    def get_all_reflections(self, summary_type: Optional[str] = None) -> List[ReflectionArtifact]:
        """Retrieve all reflection artifacts, optionally filtered by type."""
        cursor = self.memory.conn.cursor()
        
        if summary_type:
            cursor.execute("""
                SELECT * FROM reflections 
                WHERE summary_type = ?
                ORDER BY created_at DESC
            """, (summary_type,))
        else:
            cursor.execute("""
                SELECT * FROM reflections 
                ORDER BY created_at DESC
            """)
        
        artifacts = []
        for row in cursor.fetchall():
            artifacts.append(self._row_to_artifact(dict(row)))
        
        return artifacts
    
    def get_recent_reflections(self, limit: int = 5) -> List[ReflectionArtifact]:
        """Get the most recent reflection artifacts."""
        cursor = self.memory.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM reflections 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        artifacts = []
        for row in cursor.fetchall():
            artifacts.append(self._row_to_artifact(dict(row)))
        
        return artifacts
    
    def _row_to_artifact(self, row: Dict) -> ReflectionArtifact:
        """Convert database row to ReflectionArtifact."""
        return ReflectionArtifact(
            id=row['id'],
            start_step=row['start_step'],
            end_step=row['end_step'],
            summary_type=row['summary_type'],
            created_at=row['created_at'],
            trigger=row['trigger'],
            summary_data=json.loads(row['summary_json']),
            insights=json.loads(row['insights_json']),
            action_recommendations=json.loads(row['actions_json'])
        )
    
    def check_and_create_summaries(self) -> List[ReflectionArtifact]:
        """
        Check for conditions that should trigger summary creation.
        This is called periodically by the main agent.
        """
        created_summaries = []
        
        # Check for periodic summary
        periodic = self.create_periodic_summary()
        if periodic:
            created_summaries.append(periodic)
        
        # Check for milestones
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT MAX(step) FROM introspection")
        current_step = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM meta_events WHERE event_type = 'meta_decision'")
        total_decisions = cursor.fetchone()[0] or 0
        
        # Check step milestones
        for milestone_step in self.milestone_thresholds['steps']:
            if current_step >= milestone_step:
                cursor.execute("""
                    SELECT COUNT(*) FROM reflections 
                    WHERE summary_type = 'milestone' 
                    AND trigger LIKE ?
                """, (f"%steps_{milestone_step}%",))
                
                if cursor.fetchone()[0] == 0:  # Milestone not yet recorded
                    milestone = self.create_milestone_summary("steps", {"step": milestone_step})
                    if milestone:
                        created_summaries.append(milestone)
        
        # Check decision milestones
        for milestone_decisions in self.milestone_thresholds['decisions']:
            if total_decisions >= milestone_decisions:
                cursor.execute("""
                    SELECT COUNT(*) FROM reflections 
                    WHERE summary_type = 'milestone' 
                    AND trigger LIKE ?
                """, (f"%decisions_{milestone_decisions}%",))
                
                if cursor.fetchone()[0] == 0:
                    milestone = self.create_milestone_summary("decisions", {"decisions": milestone_decisions})
                    if milestone:
                        created_summaries.append(milestone)
        
        return created_summaries
    
    def generate_reflection_report(self) -> str:
        """Generate a comprehensive reflection report."""
        reflections = self.get_all_reflections()
        
        if not reflections:
            return "No reflections available yet."
        
        report = ["=== AGENT REFLECTION REPORT ===\\n"]
        
        # Summary by type
        type_counts = {}
        for r in reflections:
            type_counts[r.summary_type] = type_counts.get(r.summary_type, 0) + 1
        
        report.append("Reflection Summary:")
        for summary_type, count in type_counts.items():
            report.append(f"  • {summary_type}: {count}")
        
        report.append("\\nRecent Reflections:")
        
        for reflection in reflections[:5]:  # Show 5 most recent
            dt = datetime.fromtimestamp(reflection.created_at)
            report.append(f"\\n{dt.strftime('%Y-%m-%d %H:%M')} - {reflection.summary_type.upper()}")
            report.append(f"  Steps {reflection.start_step}-{reflection.end_step}")
            report.append(f"  Trigger: {reflection.trigger}")
            
            if reflection.insights:
                report.append("  Key Insights:")
                for insight in reflection.insights[:3]:  # Show top 3
                    report.append(f"    • {insight}")
            
            if reflection.action_recommendations:
                report.append("  Recommended Actions:")
                for action in reflection.action_recommendations[:2]:  # Show top 2
                    report.append(f"    • {action}")
        
        return "\\n".join(report)

def test_memory_summarizer():
    """Test memory summarizer with Phase 2 data."""
    print("Testing Memory Summarizer...")
    
    try:
        with MemoryDB("phase2_demo.db") as memory:
            summarizer = MemorySummarizer(memory)
            
            print("\\n1. Creating periodic summary...")
            periodic = summarizer.create_periodic_summary(force_creation=True)
            if periodic:
                print(f"   Created periodic summary: {len(periodic.insights)} insights, {len(periodic.action_recommendations)} actions")
                print(f"   Sample insight: {periodic.insights[0] if periodic.insights else 'None'}")
            
            print("\\n2. Creating milestone summary...")
            milestone = summarizer.create_milestone_summary("steps", {"step": 100})
            if milestone:
                print(f"   Created milestone summary: {milestone.trigger}")
            
            print("\\n3. Testing automatic summary creation...")
            auto_summaries = summarizer.check_and_create_summaries()
            print(f"   Created {len(auto_summaries)} automatic summaries")
            
            print("\\n4. Retrieving all reflections...")
            all_reflections = summarizer.get_all_reflections()
            print(f"   Found {len(all_reflections)} total reflections")
            
            print("\\n5. Generating reflection report...")
            report = summarizer.generate_reflection_report()
            print("   Report preview:")
            print("\\n".join(report.split("\\n")[:10]))  # Show first 10 lines
            
    except Exception as e:
        print(f"Error testing memory summarizer: {e}")
        print("Make sure phase2_agent.py has been run to generate data")
    
    print("Memory Summarizer test completed!")

if __name__ == "__main__":
    test_memory_summarizer()