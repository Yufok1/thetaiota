#!/usr/bin/env python3
"""
Memory & Knowledge Base implementation for self-reflective AI agent.
Handles persistent storage of introspection data, tasks, and meta-events.
"""

import sqlite3
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class IntrospectionEntry:
    """Represents a single introspection measurement."""
    step: int
    type: str  # "train_loss", "val_loss", "gradient_norm", etc.
    value: float
    aux_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class Task:
    """Represents a task in the queue."""
    prompt: str
    priority: int = 1  # 0 = highest priority
    objective: Optional[str] = None
    created_by: str = "system"  # "user", "system", "self_spawn"
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    id: Optional[str] = None
    created_at: Optional[float] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = time.time()
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

class MemoryDB:
    """
    Memory & Knowledge Base for the self-reflective AI agent.
    Provides persistent storage and retrieval of introspection data.
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        # Allow use across threads in the service (Phase 4 background thread + API thread)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            # Harden SQLite for concurrent access
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        self._closed = False
        # Ensure chat tables exist for conversational persistence
        self._ensure_chat_tables()

    # ---------------- SQLite helpers ----------------
    def _exec_retry(self, sql: str, args: tuple = (), retries: int = 5, backoff_s: float = 0.05):
        for attempt in range(retries):
            try:
                cur = self.conn.execute(sql, args)
                return cur
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if ("database is locked" in msg or "database locked" in msg) and attempt < retries - 1:
                    time.sleep(backoff_s * (2 ** attempt))
                    continue
                raise
        
    def __enter__(self):
        # Re-open connection if it was previously closed by a context exit
        if getattr(self, "_closed", False) or self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._closed = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Keep the connection open to allow continued use after context
        # (callers often retain this instance). Commit for safety.
        try:
            if self.conn is not None:
                self.conn.commit()
        finally:
            self._closed = False

    def ensure_connection(self):
        """Ensure the SQLite connection is open (reopen if needed)."""
        if self.conn is None or getattr(self, "_closed", False):
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            try:
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA synchronous=NORMAL;")
                self.conn.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass
            self._closed = False

    # ---------------- Chat persistence ----------------
    def _ensure_chat_tables(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,                -- 'user' | 'assistant'
                    text TEXT NOT NULL,
                    mode TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time ON chat_messages (session_id, timestamp);")
            self.conn.commit()
        except Exception:
            pass

    def create_chat_session(self, session_id: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        """Idempotently create a chat session."""
        try:
            cur = self._exec_retry("SELECT id FROM chat_sessions WHERE id = ?;", (session_id,))
            if cur.fetchone():
                return False
            self._exec_retry(
                "INSERT INTO chat_sessions (id, created_at, meta_json) VALUES (?, ?, ?);",
                (session_id, time.time(), json.dumps(meta or {}))
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def log_chat_message(self, session_id: str, role: str, text: str, mode: Optional[str] = None) -> str:
        """Persist a single chat message. Ensures session exists."""
        try:
            self.create_chat_session(session_id)
            msg_id = str(uuid.uuid4())
            self._exec_retry(
                "INSERT INTO chat_messages (id, session_id, role, text, mode, timestamp) VALUES (?, ?, ?, ?, ?, ?);",
                (msg_id, session_id, role, text, mode, time.time())
            )
            self.conn.commit()
            return msg_id
        except Exception:
            return ""

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?;",
                (session_id, int(limit))
            )
            rows = [dict(r) for r in cursor.fetchall()]
            rows.reverse()
            return rows
        except Exception:
            return []
    
    def log_introspection(self, entry: IntrospectionEntry) -> str:
        """Log an introspection measurement."""
        entry_id = str(uuid.uuid4())
        aux_json = json.dumps(entry.aux_data) if entry.aux_data else None
        
        self._exec_retry(
            """
            INSERT INTO introspection (id, step, type, value, aux_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (entry_id, entry.step, entry.type, entry.value, aux_json, entry.timestamp),
        )
        self.conn.commit()
        
        return entry_id
    
    def log_batch_metrics(self, step: int, metrics: Dict[str, float], aux_data: Optional[Dict] = None):
        """Convenience method to log multiple metrics from a training batch."""
        for metric_name, value in metrics.items():
            entry = IntrospectionEntry(
                step=step,
                type=metric_name,
                value=value,
                aux_data=aux_data
            )
            self.log_introspection(entry)
    
    def get_recent_metrics(self, metric_type: str, n: int = 10) -> List[Dict]:
        """Get the N most recent metrics of a specific type."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM introspection 
            WHERE type = ? 
            ORDER BY timestamp DESC 
            LIMIT ?;
        """, (metric_type, n))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def detect_plateau(self, metric_type: str = "val_loss", window: int = 5, threshold: float = 0.001) -> bool:
        """
        Detect if a metric has plateaued (not improving significantly).
        Returns True if the metric hasn't improved by more than threshold in the last window measurements.
        """
        recent = self.get_recent_metrics(metric_type, window)
        if len(recent) < window:
            return False
        
        # Sort by timestamp (oldest first)
        recent.sort(key=lambda x: x['timestamp'])
        
        # Check if improvement is less than threshold
        oldest_value = recent[0]['value']
        newest_value = recent[-1]['value']
        
        # For loss metrics, improvement means decrease
        if metric_type.endswith('_loss'):
            improvement = oldest_value - newest_value
        else:
            improvement = newest_value - oldest_value
            
        return improvement < threshold
    
    def enqueue_task(self, task: Task) -> str:
        """Add a task to the queue."""
        self._exec_retry(
            """
            INSERT INTO tasks (id, priority, prompt, objective, deps_json, meta_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                task.id, task.priority, task.prompt, task.objective,
                json.dumps(task.dependencies), json.dumps(task.metadata),
                task.status, task.created_at,
            ),
        )
        self.conn.commit()
        return task.id
    
    def dequeue_task(self) -> Optional[Task]:
        """Get the highest priority pending task."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks 
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC 
            LIMIT 1;
        """)
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Convert back to Task object
        task = Task(
            id=row['id'],
            priority=row['priority'],
            prompt=row['prompt'],
            objective=row['objective'],
            dependencies=json.loads(row['deps_json']),
            metadata=json.loads(row['meta_json']),
            status=row['status'],
            created_at=row['created_at']
        )
        
        return task
    
    def update_task_status(self, task_id: str, status: str, metadata_update: Optional[Dict] = None):
        """Update the status of a task."""
        cursor = self.conn.cursor()
        
        if metadata_update:
            # Get current metadata and merge
            cursor.execute("SELECT meta_json FROM tasks WHERE id = ?;", (task_id,))
            row = cursor.fetchone()
            if row:
                current_meta = json.loads(row['meta_json'])
                current_meta.update(metadata_update)
                self._exec_retry(
                    """
                    UPDATE tasks SET status = ?, meta_json = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (status, json.dumps(current_meta), time.time(), task_id),
                )
            else:
                # Task not found
                return False
        else:
            self._exec_retry(
                """
                UPDATE tasks SET status = ?, updated_at = ?
                WHERE id = ?;
                """,
                (status, time.time(), task_id),
            )
        
        self.conn.commit()
        return True
    
    def log_meta_event(self, event_type: str, info: Dict[str, Any], task_id: Optional[str] = None) -> str:
        """Log a meta-level event (self-updates, decisions, etc.)."""
        event_id = str(uuid.uuid4())
        
        self._exec_retry(
            """
            INSERT INTO meta_events (id, task_id, event_type, info_json, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """,
            (event_id, task_id, event_type, json.dumps(info), time.time()),
        )
        self.conn.commit()
        
        return event_id
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = ?;", (key,))
        row = cursor.fetchone()
        
        if row:
            try:
                # Try to parse as JSON first
                return json.loads(row['value'])
            except json.JSONDecodeError:
                # Return as string if not JSON
                return row['value']
        
        return default
    
    def explain_decision(self, step_id: int, context_window: int = 10) -> Dict[str, Any]:
        """
        Generate an explanation for what the agent was doing at a specific step.
        Returns introspection data and any meta-events around that time.
        """
        cursor = self.conn.cursor()
        
        # Get introspection data around that step
        cursor.execute("""
            SELECT * FROM introspection 
            WHERE step BETWEEN ? AND ?
            ORDER BY step, timestamp;
        """, (step_id - context_window, step_id + context_window))
        
        introspection_data = [dict(row) for row in cursor.fetchall()]
        
        # Get meta-events around that time
        if introspection_data:
            start_time = introspection_data[0]['timestamp']
            end_time = introspection_data[-1]['timestamp']
            
            cursor.execute("""
                SELECT * FROM meta_events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp;
            """, (start_time, end_time))
            
            meta_events = [dict(row) for row in cursor.fetchall()]
        else:
            meta_events = []
        
        return {
            "step_id": step_id,
            "context_window": context_window,
            "introspection_data": introspection_data,
            "meta_events": meta_events,
            "summary": self._generate_explanation_summary(introspection_data, meta_events)
        }
    
    def _generate_explanation_summary(self, introspection_data: List[Dict], meta_events: List[Dict]) -> str:
        """Generate a human-readable summary of what was happening."""
        if not introspection_data:
            return "No data available for this time period."
        
        summary = []
        
        # Analyze loss trends
        losses = [d for d in introspection_data if 'loss' in d['type']]
        if losses:
            if len(losses) > 1:
                trend = "decreasing" if losses[-1]['value'] < losses[0]['value'] else "increasing"
                summary.append(f"Loss was {trend} (from {losses[0]['value']:.4f} to {losses[-1]['value']:.4f})")
        
        # Analyze meta-events
        if meta_events:
            event_types = [e['event_type'] for e in meta_events]
            summary.append(f"Meta-events occurred: {', '.join(set(event_types))}")
        
        return "; ".join(summary) if summary else "Normal training progression."

def test_memory_db():
    """Test the MemoryDB functionality."""
    print("Testing MemoryDB...")
    
    # Initialize the test database first
    from init_database import create_database
    create_database("test_memory.db")
    
    with MemoryDB("test_memory.db") as memory:
        # Test introspection logging
        entry = IntrospectionEntry(step=1, type="train_loss", value=0.5)
        entry_id = memory.log_introspection(entry)
        print(f"Logged introspection entry: {entry_id}")
        
        # Test batch metrics
        memory.log_batch_metrics(
            step=2,
            metrics={"train_loss": 0.4, "val_loss": 0.45, "gradient_norm": 0.1},
            aux_data={"batch_size": 32}
        )
        print("Logged batch metrics")
        
        # Test recent metrics retrieval
        recent_losses = memory.get_recent_metrics("train_loss", n=5)
        print(f"Recent train losses: {[r['value'] for r in recent_losses]}")
        
        # Test task queue
        task = Task(
            prompt="Fine-tune on hard examples",
            priority=0,
            objective="Improve accuracy on difficult cases",
            created_by="self_spawn"
        )
        task_id = memory.enqueue_task(task)
        print(f"Enqueued task: {task_id}")
        
        # Test task dequeue
        next_task = memory.dequeue_task()
        print(f"Dequeued task: {next_task.prompt if next_task else 'None'}")
        
        # Test meta-event logging
        event_id = memory.log_meta_event(
            event_type="self_update_triggered",
            info={"reason": "plateau_detected", "plateau_steps": 3}
        )
        print(f"Logged meta-event: {event_id}")
        
        print("MemoryDB tests completed successfully!")

if __name__ == "__main__":
    test_memory_db()