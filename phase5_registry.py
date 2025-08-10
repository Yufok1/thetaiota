#!/usr/bin/env python3
"""
Phase 5: Agent Registry & Discovery System - Central coordination for federated agents.
Manages agent registration, health monitoring, and capability discovery.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

class AgentCapability(Enum):
    """Agent capabilities for specialization."""
    GENERAL_LEARNING = "general_learning"
    DECISION_MAKING = "decision_making" 
    TASK_SPAWNING = "task_spawning"
    REFLECTION_ANALYSIS = "reflection_analysis"
    HUMAN_FEEDBACK = "human_feedback"
    MEMORY_QUERYING = "memory_querying"
    CRISIS_DETECTION = "crisis_detection"

@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    name: str
    endpoint: str  # HTTP API endpoint
    ws_endpoint: str  # WebSocket endpoint
    capabilities: List[AgentCapability]
    status: str  # "online", "offline", "busy", "error"
    performance_metrics: Dict[str, float]
    last_heartbeat: float
    registered_at: float
    metadata: Dict[str, Any]

@dataclass
class RegistryEvent:
    """Event from the registry system."""
    event_type: str  # "agent_joined", "agent_left", "status_changed"
    agent_id: str
    timestamp: float
    data: Dict[str, Any]

class AgentRegistry:
    """
    Phase 5: Central registry for federated agent discovery and coordination.
    
    Features:
    - Agent registration and discovery
    - Health monitoring with heartbeat
    - Capability-based agent selection
    - Event broadcasting for registry changes
    - Persistent storage of agent information
    """
    
    def __init__(self, registry_db: str = "phase5_registry.db"):
        self.registry_db = registry_db
        self.agents: Dict[str, AgentInfo] = {}
        self.event_subscribers = []
        self.heartbeat_timeout = 30.0  # seconds
        
        # Initialize database
        self._init_registry_db()
        
        # Start background tasks
        self._cleanup_task = None
        self._running = False
        
        print(f"AgentRegistry initialized with database: {registry_db}")
    
    def _init_registry_db(self):
        """Initialize registry database schema."""
        conn = sqlite3.connect(self.registry_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id        TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                endpoint        TEXT NOT NULL,
                ws_endpoint     TEXT NOT NULL,
                capabilities    TEXT NOT NULL,
                status          TEXT NOT NULL,
                performance_metrics TEXT,
                last_heartbeat  REAL NOT NULL,
                registered_at   REAL NOT NULL,
                metadata        TEXT
            );
        """)
        
        # Registry events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registry_events (
                id              TEXT PRIMARY KEY,
                event_type      TEXT NOT NULL,
                agent_id        TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                data            TEXT NOT NULL
            );
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agents_capabilities ON agents(capabilities);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON registry_events(timestamp);")
        
        conn.commit()
        conn.close()
        
        print("Registry database initialized")
    
    async def start(self):
        """Start the registry service."""
        self._running = True
        
        # Load existing agents from database
        await self._load_agents_from_db()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._heartbeat_monitor())
        
        print("Agent Registry started")
    
    async def stop(self):
        """Stop the registry service."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        print("Agent Registry stopped")
    
    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent in the federation."""
        try:
            # Update timestamps
            agent_info.registered_at = time.time()
            agent_info.last_heartbeat = time.time()
            agent_info.status = "online"
            
            # Store in memory
            self.agents[agent_info.agent_id] = agent_info
            
            # Store in database
            await self._save_agent_to_db(agent_info)
            
            # Broadcast event
            await self._emit_event("agent_joined", agent_info.agent_id, {
                "name": agent_info.name,
                "capabilities": [cap.value for cap in agent_info.capabilities],
                "endpoint": agent_info.endpoint
            })
            
            print(f"Agent registered: {agent_info.agent_id} ({agent_info.name})")
            return True
            
        except Exception as e:
            print(f"Agent registration failed: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the federation."""
        try:
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                
                # Remove from memory
                del self.agents[agent_id]
                
                # Update database status
                await self._update_agent_status_in_db(agent_id, "offline")
                
                # Broadcast event
                await self._emit_event("agent_left", agent_id, {
                    "name": agent_info.name,
                    "reason": "unregistered"
                })
                
                print(f"Agent unregistered: {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Agent unregistration failed: {e}")
            return False
    
    async def update_heartbeat(self, agent_id: str, 
                             performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update agent heartbeat and performance metrics."""
        try:
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                agent_info.last_heartbeat = time.time()
                
                if performance_metrics:
                    agent_info.performance_metrics.update(performance_metrics)
                
                # Update status if was offline
                if agent_info.status == "offline":
                    agent_info.status = "online"
                    await self._emit_event("status_changed", agent_id, {
                        "old_status": "offline",
                        "new_status": "online"
                    })
                
                # Save to database
                await self._save_agent_to_db(agent_info)
                return True
            
            return False
            
        except Exception as e:
            print(f"Heartbeat update failed: {e}")
            return False
    
    async def find_agents_by_capability(self, capability: AgentCapability, 
                                      status_filter: str = "online") -> List[AgentInfo]:
        """Find agents with specific capabilities."""
        matching_agents = []
        
        for agent_info in self.agents.values():
            if (capability in agent_info.capabilities and 
                agent_info.status == status_filter):
                matching_agents.append(agent_info)
        
        return matching_agents
    
    async def get_best_agent_for_task(self, required_capabilities: List[AgentCapability],
                                    performance_metric: str = "recent_val_loss") -> Optional[AgentInfo]:
        """Find the best agent for a specific task based on capabilities and performance."""
        candidates = []
        
        for agent_info in self.agents.values():
            if (agent_info.status == "online" and 
                all(cap in agent_info.capabilities for cap in required_capabilities)):
                candidates.append(agent_info)
        
        if not candidates:
            return None
        
        # Sort by performance metric (lower is better for loss metrics)
        if performance_metric in candidates[0].performance_metrics:
            candidates.sort(key=lambda a: a.performance_metrics.get(performance_metric, float('inf')))
        
        return candidates[0]
    
    async def get_all_agents(self, status_filter: Optional[str] = None) -> List[AgentInfo]:
        """Get all registered agents, optionally filtered by status."""
        if status_filter:
            return [agent for agent in self.agents.values() if agent.status == status_filter]
        return list(self.agents.values())
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get specific agent information."""
        return self.agents.get(agent_id)
    
    async def update_agent_status(self, agent_id: str, new_status: str) -> bool:
        """Update agent status."""
        try:
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                old_status = agent_info.status
                agent_info.status = new_status
                
                # Save to database
                await self._update_agent_status_in_db(agent_id, new_status)
                
                # Broadcast event if status changed
                if old_status != new_status:
                    await self._emit_event("status_changed", agent_id, {
                        "old_status": old_status,
                        "new_status": new_status
                    })
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Status update failed: {e}")
            return False
    
    async def _heartbeat_monitor(self):
        """Background task to monitor agent heartbeats."""
        while self._running:
            try:
                current_time = time.time()
                offline_agents = []
                
                for agent_id, agent_info in self.agents.items():
                    if (current_time - agent_info.last_heartbeat > self.heartbeat_timeout and
                        agent_info.status == "online"):
                        offline_agents.append(agent_id)
                
                # Mark offline agents
                for agent_id in offline_agents:
                    await self.update_agent_status(agent_id, "offline")
                    print(f"Agent marked offline due to missed heartbeat: {agent_id}")
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _emit_event(self, event_type: str, agent_id: str, data: Dict[str, Any]):
        """Emit a registry event."""
        event = RegistryEvent(
            event_type=event_type,
            agent_id=agent_id,
            timestamp=time.time(),
            data=data
        )
        
        # Store in database
        await self._save_event_to_db(event)
        
        # Notify subscribers (for future WebSocket broadcasting)
        for subscriber in self.event_subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                print(f"Event subscriber error: {e}")
    
    async def _save_agent_to_db(self, agent_info: AgentInfo):
        """Save agent information to database."""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agents 
            (agent_id, name, endpoint, ws_endpoint, capabilities, status, 
             performance_metrics, last_heartbeat, registered_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_info.agent_id,
            agent_info.name,
            agent_info.endpoint,
            agent_info.ws_endpoint,
            json.dumps([cap.value for cap in agent_info.capabilities]),
            agent_info.status,
            json.dumps(agent_info.performance_metrics),
            agent_info.last_heartbeat,
            agent_info.registered_at,
            json.dumps(agent_info.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_agent_status_in_db(self, agent_id: str, status: str):
        """Update agent status in database."""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE agents 
            SET status = ?, last_heartbeat = ?
            WHERE agent_id = ?
        """, (status, time.time(), agent_id))
        
        conn.commit()
        conn.close()
    
    async def _save_event_to_db(self, event: RegistryEvent):
        """Save registry event to database."""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO registry_events 
            (id, event_type, agent_id, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            event.event_type,
            event.agent_id,
            event.timestamp,
            json.dumps(event.data)
        ))
        
        conn.commit()
        conn.close()
    
    async def _load_agents_from_db(self):
        """Load existing agents from database on startup."""
        conn = sqlite3.connect(self.registry_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agents")
        
        loaded_count = 0
        for row in cursor.fetchall():
            try:
                capabilities = [AgentCapability(cap) for cap in json.loads(row['capabilities'])]
                
                agent_info = AgentInfo(
                    agent_id=row['agent_id'],
                    name=row['name'],
                    endpoint=row['endpoint'],
                    ws_endpoint=row['ws_endpoint'],
                    capabilities=capabilities,
                    status=row['status'],
                    performance_metrics=json.loads(row['performance_metrics'] or "{}"),
                    last_heartbeat=row['last_heartbeat'],
                    registered_at=row['registered_at'],
                    metadata=json.loads(row['metadata'] or "{}")
                )
                
                self.agents[agent_info.agent_id] = agent_info
                loaded_count += 1
                
            except Exception as e:
                print(f"Failed to load agent {row['agent_id']}: {e}")
        
        conn.close()
        print(f"Loaded {loaded_count} agents from database")

async def test_agent_registry():
    """Test the agent registry system."""
    print("=== PHASE 5: AGENT REGISTRY TEST ===\n")
    
    # Create registry
    registry = AgentRegistry("test_registry.db")
    await registry.start()
    
    # Create test agents
    agent1 = AgentInfo(
        agent_id="agent_001",
        name="Primary Learning Agent",
        endpoint="http://localhost:8081",
        ws_endpoint="ws://localhost:8081/stream",
        capabilities=[
            AgentCapability.GENERAL_LEARNING,
            AgentCapability.DECISION_MAKING,
            AgentCapability.REFLECTION_ANALYSIS
        ],
        status="online",
        performance_metrics={"recent_val_loss": 0.45, "accuracy": 0.89},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={"specialization": "general", "version": "4.0"}
    )
    
    agent2 = AgentInfo(
        agent_id="agent_002", 
        name="Task Specialization Agent",
        endpoint="http://localhost:8082",
        ws_endpoint="ws://localhost:8082/stream",
        capabilities=[
            AgentCapability.TASK_SPAWNING,
            AgentCapability.CRISIS_DETECTION,
            AgentCapability.HUMAN_FEEDBACK
        ],
        status="online",
        performance_metrics={"recent_val_loss": 0.52, "accuracy": 0.83},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={"specialization": "task_management", "version": "4.0"}
    )
    
    # Test registration
    print("1. Testing agent registration...")
    success1 = await registry.register_agent(agent1)
    success2 = await registry.register_agent(agent2)
    print(f"   Agent 1 registered: {success1}")
    print(f"   Agent 2 registered: {success2}")
    
    # Test discovery
    print("\n2. Testing agent discovery...")
    all_agents = await registry.get_all_agents()
    print(f"   Total agents: {len(all_agents)}")
    
    online_agents = await registry.get_all_agents("online")
    print(f"   Online agents: {len(online_agents)}")
    
    # Test capability-based search
    print("\n3. Testing capability search...")
    learning_agents = await registry.find_agents_by_capability(AgentCapability.GENERAL_LEARNING)
    print(f"   Agents with learning capability: {len(learning_agents)}")
    
    task_agents = await registry.find_agents_by_capability(AgentCapability.TASK_SPAWNING)
    print(f"   Agents with task spawning: {len(task_agents)}")
    
    # Test best agent selection
    print("\n4. Testing best agent selection...")
    best_learner = await registry.get_best_agent_for_task(
        [AgentCapability.GENERAL_LEARNING, AgentCapability.DECISION_MAKING],
        "recent_val_loss"
    )
    
    if best_learner:
        print(f"   Best learning agent: {best_learner.name} (loss: {best_learner.performance_metrics['recent_val_loss']})")
    
    # Test heartbeat updates
    print("\n5. Testing heartbeat updates...")
    heartbeat_success = await registry.update_heartbeat("agent_001", {"recent_val_loss": 0.42})
    print(f"   Heartbeat update: {heartbeat_success}")
    
    # Test status changes
    print("\n6. Testing status changes...")
    status_success = await registry.update_agent_status("agent_002", "busy")
    print(f"   Status change: {status_success}")
    
    # Final agent list
    print("\n7. Final registry state...")
    final_agents = await registry.get_all_agents()
    for agent in final_agents:
        print(f"   {agent.agent_id}: {agent.name} ({agent.status}) - {len(agent.capabilities)} capabilities")
    
    # Stop registry
    await registry.stop()
    
    print("\n[OK] AGENT REGISTRY TEST COMPLETED!")
    print("Registry provides:")
    print("- Agent registration and discovery")
    print("- Capability-based agent selection")
    print("- Health monitoring with heartbeats")
    print("- Performance-based task routing")
    print("- Event broadcasting for coordination")

if __name__ == "__main__":
    # Clean up test database
    import os
    if os.path.exists("test_registry.db"):
        os.remove("test_registry.db")
    
    asyncio.run(test_agent_registry())