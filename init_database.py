#!/usr/bin/env python3
"""
Initialize SQLite database with schema for self-reflective AI agent.
Based on Phase 0 design documents.
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path

def create_database(db_path="agent_memory.db"):
    """Create SQLite database with introspection, tasks, and meta_events tables."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. INTROSPECTION LOGS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS introspection (
        id            TEXT PRIMARY KEY,          -- uuid string
        step          INTEGER      NOT NULL,     -- global training step
        type          TEXT         NOT NULL,     -- e.g. "gradient_norm", "val_loss"
        value         REAL         NOT NULL,     -- numeric metric (float)
        aux_json      TEXT,                     -- optional JSON for extra dims
        timestamp     REAL         NOT NULL      -- Unix epoch (float, seconds)
    );
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_introspection_step ON introspection (step);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_introspection_type ON introspection (type);")
    
    # 2. TASK QUEUE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id           TEXT PRIMARY KEY,
        priority     INTEGER      NOT NULL DEFAULT 1,  -- 0 = highest
        prompt       TEXT         NOT NULL,
        objective    TEXT,
        deps_json    TEXT         NOT NULL DEFAULT '[]',   -- JSON list of task ids
        meta_json    TEXT         NOT NULL DEFAULT '{}',   -- arbitrary key/val
        status       TEXT         NOT NULL DEFAULT 'pending',
        created_at   REAL         NOT NULL,      -- Unix epoch
        updated_at   REAL
    );
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks (priority, created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks (status);")
    
    # 3. META EVENTS (audit-trail for self-updates & controller actions)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS meta_events (
        id            TEXT PRIMARY KEY,
        task_id       TEXT,
        event_type    TEXT         NOT NULL,             -- e.g. "self_update_start"
        info_json     TEXT         NOT NULL DEFAULT '{}',
        timestamp     REAL         NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
    );
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_meta_events_time ON meta_events (timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_meta_events_type ON meta_events (event_type);")
    
    # 4. CONFIG (key-value store)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS config (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    
    # Insert default config values
    cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('abort_if_vram_gt', '2.8');")  # 2.8GB limit
    cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('plateau_threshold', '3');")    # 3 epochs
    cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('confidence_threshold', '0.7');")
    
    # 5. UNIFIED MEMORIES VIEW (Phase-3+)
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS memories AS
    SELECT id,
           'introspection' AS type,
           timestamp,
           json_object('step', step,
                       'metric_type', type,
                       'value', value,
                       'aux', aux_json) AS payload
    FROM introspection
    UNION ALL
    SELECT id,
           'task' AS type,
           created_at AS timestamp,
           json_object('prompt', prompt,
                       'objective', objective,
                       'priority', priority,
                       'deps', deps_json,
                       'meta', meta_json,
                       'status', status) AS payload
    FROM tasks
    UNION ALL
    SELECT id,
           'event' AS type,
           timestamp,
           json_object('task_id', task_id,
                       'event_type', event_type,
                       'info', info_json) AS payload
    FROM meta_events;
    """)
    
    conn.commit()
    print(f"Database initialized: {db_path}")
    print(f"Tables created: introspection, tasks, meta_events, config")
    print(f"View created: memories (Phase-3 compatibility)")
    
    # Test the database with a sample entry
    cursor.execute("""
    INSERT INTO introspection (id, step, type, value, timestamp) 
    VALUES (?, ?, ?, ?, ?);
    """, (str(uuid.uuid4()), 0, "init_test", 1.0, time.time()))
    
    conn.commit()
    conn.close()
    
    return db_path

if __name__ == "__main__":
    db_path = create_database()
    print("Database ready for Phase 1 training!")