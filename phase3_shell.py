#!/usr/bin/env python3
"""
Phase 3: Interactive Agent Shell - CLI/API for real-time agent introspection.
Makes the agent's knowledge and reasoning directly queryable.
"""

import cmd
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

from memory_db import MemoryDB
from reflection_explainer import ReflectionExplainer, DecisionExplanation, ReflectionSummary

class AgentShell(cmd.Cmd):
    """
    Interactive shell for querying the self-reflective AI agent.
    Provides real-time access to agent's memory, reasoning, and decisions.
    """
    
    intro = '''
=== PHASE 3: SELF-AWARE AI AGENT SHELL ===
Welcome to the interactive agent introspection system.
Type 'help' for commands or '?' for a list of topics.
The agent can explain its decisions, analyze its behavior, and reflect on its learning.
    '''
    
    prompt = 'agent> '
    
    def __init__(self, db_path: str = "phase2_demo.db"):
        super().__init__()
        self.db_path = db_path
        self.memory = None
        self.explainer = None
        self._connect_to_agent()
        
        # Shell state
        self.last_query_result = None
        self.current_step_range = None
    
    def _connect_to_agent(self):
        """Connect to the agent's memory database."""
        try:
            self.memory = MemoryDB(self.db_path)
            self.explainer = ReflectionExplainer(self.memory)
            print(f"Connected to agent memory at: {self.db_path}")
            
            # Show basic agent stats
            self._show_agent_status()
            
        except Exception as e:
            print(f"Error connecting to agent: {e}")
            print("Make sure the agent has been trained (run phase2_agent.py)")
    
    def _show_agent_status(self):
        """Show basic agent status information."""
        if not self.memory:
            return
        
        cursor = self.memory.conn.cursor()
        
        # Get training progress
        cursor.execute("SELECT MAX(step) FROM introspection")
        max_step = cursor.fetchone()[0] or 0
        
        # Get decision count
        cursor.execute("SELECT COUNT(*) FROM meta_events WHERE event_type = 'meta_decision'")
        decision_count = cursor.fetchone()[0] or 0
        
        # Get task count
        cursor.execute("SELECT COUNT(*) FROM meta_events WHERE event_type = 'tasks_spawned'")
        task_count = cursor.fetchone()[0] or 0
        
        print(f"Agent Status: {max_step} training steps, {decision_count} decisions, {task_count} task spawning events")
    
    def do_status(self, arg):
        """Show current agent status and recent activity."""
        if not self.explainer:
            print("Not connected to agent memory")
            return
        
        try:
            # Get recent performance
            cursor = self.memory.conn.cursor()
            cursor.execute("SELECT MAX(step) FROM introspection")
            max_step = cursor.fetchone()[0] or 0
            
            if max_step == 0:
                print("No training data found")
                return
            
            # Show recent activity summary
            if max_step > 20:
                summary = self.explainer.create_behavior_summary(max_step - 20, max_step)
                print(f"\\nRecent Activity (last 20 steps):")
                print(f"  Decisions made: {summary.decisions_made}")
                print(f"  Tasks spawned: {summary.tasks_spawned}")
                print(f"  Performance trend: {summary.performance_trend}")
                print(f"  Key insights: {', '.join(summary.key_insights)}")
                
                if summary.weakness_patterns:
                    print(f"  Common weaknesses: {list(summary.weakness_patterns.keys())}")
            
            # Show latest metrics
            cursor.execute("""
                SELECT type, value FROM introspection 
                WHERE step = (SELECT MAX(step) FROM introspection)
            """)
            
            latest_metrics = cursor.fetchall()
            if latest_metrics:
                print(f"\\nLatest Metrics (step {max_step}):")
                for metric_type, value in latest_metrics:
                    print(f"  {metric_type}: {value:.4f}")
                    
        except Exception as e:
            print(f"Error getting status: {e}")
    
    def do_explain(self, arg):
        """
        Explain agent decisions. Usage:
        explain last - explain the last decision
        explain step N - explain decision at step N
        explain recent - explain recent behavior pattern
        """
        if not self.explainer:
            print("Not connected to agent memory")
            return
        
        parts = arg.strip().split()
        
        try:
            if not parts or parts[0] == "last":
                explanation = self.explainer.explain_last_action()
                self._print_decision_explanation(explanation)
                
            elif parts[0] == "step" and len(parts) > 1:
                step = int(parts[1])
                explanation = self.explainer.explain_decision_at_step(step)
                self._print_decision_explanation(explanation)
                
            elif parts[0] == "recent":
                cursor = self.memory.conn.cursor()
                cursor.execute("SELECT MAX(step) FROM introspection")
                max_step = cursor.fetchone()[0] or 0
                
                if max_step > 30:
                    summary = self.explainer.create_behavior_summary(max_step - 30, max_step)
                    self._print_behavior_summary(summary)
                else:
                    print("Not enough data for recent behavior analysis")
                    
            else:
                print("Usage: explain [last|step N|recent]")
                
        except Exception as e:
            print(f"Error explaining: {e}")
    
    def _print_decision_explanation(self, explanation: DecisionExplanation):
        """Pretty print a decision explanation."""
        print(f"\\n=== Decision Explanation (Step {explanation.step}) ===")
        print(f"Action Taken: {explanation.action_taken}")
        print(f"Confidence: {explanation.confidence:.3f}")
        print(f"Reasoning: {explanation.reasoning}")
        
        if explanation.context_factors:
            print(f"\\nContext:")
            for key, value in explanation.context_factors.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.4f}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        if explanation.outcome_metrics:
            print(f"\\nOutcome Assessment:")
            for key, value in explanation.outcome_metrics.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        
        if explanation.related_events:
            print(f"\\nRelated Events ({len(explanation.related_events)}):")
            for event in explanation.related_events[:3]:  # Show max 3
                print(f"  - {event['event_type']}: {event.get('info', {})}")
    
    def _print_behavior_summary(self, summary: ReflectionSummary):
        """Pretty print a behavior summary."""
        print(f"\\n=== Behavior Summary (Steps {summary.start_step}-{summary.end_step}) ===")
        print(f"Duration: {summary.duration_steps} steps")
        print(f"Decisions made: {summary.decisions_made}")
        print(f"Tasks spawned: {summary.tasks_spawned}")
        print(f"Performance trend: {summary.performance_trend}")
        
        if summary.key_insights:
            print(f"\\nKey Insights:")
            for insight in summary.key_insights:
                print(f"  • {insight}")
        
        if summary.weakness_patterns:
            print(f"\\nWeakness Patterns:")
            for weakness, count in summary.weakness_patterns.items():
                print(f"  • {weakness}: {count} occurrences")
        
        if summary.meta_learning_progress:
            print(f"\\nMeta-Learning Progress:")
            for key, value in summary.meta_learning_progress.items():
                print(f"  • {key}: {value}")
    
    def do_query(self, arg):
        """
        Natural language queries about agent behavior.
        Examples:
        query what was my last decision?
        query how is my performance?
        query what tasks have I spawned?
        query what are my weaknesses?
        """
        if not self.explainer:
            print("Not connected to agent memory")
            return
        
        if not arg.strip():
            print("Usage: query <your question>")
            print("Examples:")
            print("  query what was my last decision?")
            print("  query how is my performance?")
            print("  query what tasks have I spawned?")
            return
        
        try:
            result = self.explainer.query_agent_memory(arg)
            self.last_query_result = result
            
            query_type = result.get('query_type', 'unknown')
            data = result.get('result')
            
            print(f"\\nQuery Type: {query_type}")
            
            if query_type == "last_decision" and isinstance(data, DecisionExplanation):
                self._print_decision_explanation(data)
                
            elif query_type == "performance_summary" and isinstance(data, ReflectionSummary):
                self._print_behavior_summary(data)
                
            elif query_type == "task_spawning":
                print(f"Recent Task Spawning Events:")
                for i, event in enumerate(data[:3], 1):
                    info = json.loads(event['info_json'])
                    print(f"  {i}. Step {info.get('step', '?')}: {info.get('num_tasks', 0)} tasks")
                    print(f"     Weaknesses: {', '.join(info.get('weaknesses', []))}")
                    
            elif query_type == "weakness_analysis":
                if data:
                    print(f"Weakness Pattern Analysis:")
                    for weakness, count in sorted(data.items(), key=lambda x: x[1], reverse=True):
                        print(f"  • {weakness}: {count} occurrences")
                else:
                    print("No weaknesses detected yet")
                    
            else:
                print(f"Result: {data}")
                
        except Exception as e:
            print(f"Error processing query: {e}")
    
    def do_memory(self, arg):
        """
        Inspect agent memory directly.
        memory recent - show recent memory entries
        memory step N - show memories around step N
        memory search TEXT - search memory for text
        """
        if not self.memory:
            print("Not connected to agent memory")
            return
        
        parts = arg.strip().split()
        
        try:
            cursor = self.memory.conn.cursor()
            
            if not parts or parts[0] == "recent":
                cursor.execute("""
                    SELECT * FROM memories 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """)
                
                print("\\nRecent Memory Entries:")
                for row in cursor.fetchall():
                    entry = dict(row)
                    payload = json.loads(entry['payload'])
                    print(f"  {entry['type']}: {payload}")
                    
            elif parts[0] == "step" and len(parts) > 1:
                step = int(parts[1])
                cursor.execute("""
                    SELECT * FROM memories 
                    WHERE json_extract(payload, '$.step') BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (step - 5, step + 5))
                
                print(f"\\nMemory entries around step {step}:")
                for row in cursor.fetchall():
                    entry = dict(row)
                    payload = json.loads(entry['payload'])
                    print(f"  {entry['type']}: {payload}")
                    
            elif parts[0] == "search" and len(parts) > 1:
                search_term = " ".join(parts[1:])
                cursor.execute("""
                    SELECT * FROM memories 
                    WHERE payload LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, (f"%{search_term}%",))
                
                print(f"\\nMemory search results for '{search_term}':")
                for row in cursor.fetchall():
                    entry = dict(row)
                    payload = json.loads(entry['payload'])
                    print(f"  {entry['type']}: {payload}")
                    
            else:
                print("Usage: memory [recent|step N|search TEXT]")
                
        except Exception as e:
            print(f"Error accessing memory: {e}")
    
    def do_stats(self, arg):
        """Show detailed agent statistics."""
        if not self.memory:
            print("Not connected to agent memory")
            return
        
        try:
            cursor = self.memory.conn.cursor()
            
            print("\\n=== Agent Statistics ===")
            
            # Training progress
            cursor.execute("SELECT MIN(step), MAX(step), COUNT(DISTINCT step) FROM introspection")
            min_step, max_step, step_count = cursor.fetchone()
            print(f"Training: {step_count} unique steps ({min_step} to {max_step})")
            
            # Decision statistics
            cursor.execute("""
                SELECT json_extract(info_json, '$.action') as action, COUNT(*) as count
                FROM meta_events 
                WHERE event_type = 'meta_decision'
                GROUP BY action
                ORDER BY count DESC
            """)
            
            decision_stats = cursor.fetchall()
            if decision_stats:
                print(f"\\nDecision Distribution:")
                for action, count in decision_stats:
                    print(f"  {action}: {count}")
            
            # Task spawning statistics
            cursor.execute("SELECT COUNT(*) FROM meta_events WHERE event_type = 'tasks_spawned'")
            spawn_events = cursor.fetchone()[0]
            print(f"\\nTask Spawning Events: {spawn_events}")
            
            # Performance metrics
            cursor.execute("""
                SELECT type, AVG(value) as avg_val, MIN(value) as min_val, MAX(value) as max_val
                FROM introspection
                WHERE type IN ('val_loss', 'val_accuracy', 'val_confidence')
                GROUP BY type
            """)
            
            perf_stats = cursor.fetchall()
            if perf_stats:
                print(f"\\nPerformance Metrics:")
                for metric_type, avg_val, min_val, max_val in perf_stats:
                    print(f"  {metric_type}: avg={avg_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def do_timeline(self, arg):
        """Show timeline of agent decisions and events."""
        if not self.memory:
            print("Not connected to agent memory")
            return
        
        try:
            cursor = self.memory.conn.cursor()
            
            # Get all major events
            cursor.execute("""
                SELECT event_type, info_json, timestamp FROM meta_events 
                WHERE event_type IN ('meta_decision', 'tasks_spawned', 'curriculum_advanced')
                ORDER BY timestamp DESC
                LIMIT 15
            """)
            
            events = cursor.fetchall()
            
            print("\\n=== Agent Timeline (Recent Events) ===")
            
            for event_type, info_json, timestamp in events:
                dt = datetime.fromtimestamp(timestamp)
                info = json.loads(info_json)
                
                if event_type == 'meta_decision':
                    step = info.get('step', '?')
                    action = info.get('action', '?')
                    reasoning = info.get('reasoning', '')[:50] + "..." if len(info.get('reasoning', '')) > 50 else info.get('reasoning', '')
                    print(f"  {dt.strftime('%H:%M:%S')} - DECISION @ step {step}: {action}")
                    print(f"    └─ {reasoning}")
                    
                elif event_type == 'tasks_spawned':
                    step = info.get('step', '?')
                    num_tasks = info.get('num_tasks', 0)
                    weaknesses = ", ".join(info.get('weaknesses', []))
                    print(f"  {dt.strftime('%H:%M:%S')} - TASKS @ step {step}: spawned {num_tasks}")
                    print(f"    └─ addressing: {weaknesses}")
                    
                elif event_type == 'curriculum_advanced':
                    step = info.get('step', '?')
                    new_level = info.get('new_level', '?')
                    print(f"  {dt.strftime('%H:%M:%S')} - CURRICULUM @ step {step}: advanced to {new_level}")
            
        except Exception as e:
            print(f"Error showing timeline: {e}")
    
    def do_connect(self, arg):
        """Connect to a different agent database."""
        if not arg.strip():
            print("Usage: connect <database_path>")
            return
        
        db_path = arg.strip()
        
        try:
            if self.memory:
                self.memory.conn.close()
            
            self.db_path = db_path
            self._connect_to_agent()
            
        except Exception as e:
            print(f"Error connecting to {db_path}: {e}")
    
    def do_exit(self, arg):
        """Exit the agent shell."""
        if self.memory:
            self.memory.conn.close()
        print("\\nGoodbye! The agent's memory persists...")
        return True
    
    def do_EOF(self, arg):
        """Handle Ctrl+D to exit."""
        return self.do_exit(arg)
    
    def emptyline(self):
        """Handle empty line input."""
        pass
    
    def default(self, line):
        """Handle unrecognized commands as natural language queries."""
        if line.strip():
            print(f"Unknown command: {line}")
            print("Try 'query {line}' for natural language queries")

def main():
    """Run the interactive agent shell."""
    print("Starting Phase 3 Interactive Agent Shell...")
    
    # Check for existing agent databases
    import os
    possible_dbs = ["phase2_demo.db", "phase2_memory.db", "agent_memory.db"]
    
    existing_db = None
    for db in possible_dbs:
        if os.path.exists(db):
            existing_db = db
            break
    
    if existing_db:
        print(f"Found agent database: {existing_db}")
        shell = AgentShell(existing_db)
    else:
        print("No agent database found. Run phase2_agent.py first to train the agent.")
        print("Creating shell anyway - use 'connect <path>' to connect to an agent.")
        shell = AgentShell()
    
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    except Exception as e:
        print(f"Error in shell: {e}")

if __name__ == "__main__":
    main()