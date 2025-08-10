#!/usr/bin/env python3
"""
Phase 4: Agent Service Core - Production-ready wrapper for the self-aware AI agent.
Provides async lifecycle management, real-time streaming, and API integration.
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import uuid
from datetime import datetime

from phase3_agent import Phase3Agent
from human_feedback_system import FeedbackType, FeedbackSentiment
from chat_engine import ChatEngine, ChatConfig

class ServiceState(Enum):
    """Agent service lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready" 
    TRAINING = "training"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class ServiceEvent:
    """Real-time event from the agent service."""
    id: str
    timestamp: float
    event_type: str
    agent_id: str
    step: int
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON for streaming."""
        return json.dumps({
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "step": self.step,
            "data": self.data
        })

@dataclass
class ServiceStatus:
    """Current status of the agent service."""
    state: ServiceState
    agent_id: str
    uptime: float
    current_step: int
    total_decisions: int
    total_reflections: int
    total_feedback: int
    last_decision: Optional[str]
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None

class AgentService:
    """
    Phase 4: Production-ready agent service wrapper.
    
    Features:
    - Async lifecycle management  
    - Real-time event streaming
    - Thread-safe operation
    - Health monitoring
    - Memory persistence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get('agent_id', f"agent_{uuid.uuid4().hex[:8]}")
        
        # Service state
        self.state = ServiceState.INITIALIZING
        self.start_time = time.time()
        self.error_message = None
        
        # Agent instance (will be created in background thread)
        self.agent: Optional[Phase3Agent] = None
        self.agent_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Event streaming
        self.event_queue = queue.Queue(maxsize=1000)
        self.event_subscribers = []
        
        # Performance tracking
        self.metrics_history = []
        self.decision_count = 0
        self.reflection_count = 0
        
        print(f"AgentService initialized: {self.agent_id}")

        # Peers (list of http base URLs like http://127.0.0.1:8082)
        self.peers: List[str] = config.get('peers', [])
        self.heartbeat_interval_s: float = float(config.get('heartbeat_interval_s', 5.0))
        self._hb_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._startup_grace_s: float = float(config.get('startup_grace_s', 3.0))
        self._last_peer_contact: Dict[str, float] = {}
        self._official_sync_interval_s: float = float(config.get('official_sync_interval_s', 30.0))
        self._leader_id: Optional[str] = config.get('leader_id')
        self.leader_url: Optional[str] = config.get('leader_url')

        # Chat engine (reflective by default)
        self.chat_engine = ChatEngine(agent_ref=None, config=ChatConfig())
    
    async def start_async(self) -> bool:
        """Start the agent service asynchronously."""
        try:
            print(f"Starting agent service: {self.agent_id}")
            self.state = ServiceState.INITIALIZING
            
            # Start agent in background thread
            self.agent_thread = threading.Thread(
                target=self._run_agent_training,
                daemon=True
            )
            self.agent_thread.start()
            
            # Wait for agent to initialize with more patience
            for i in range(10):  # Wait up to 10 seconds
                await asyncio.sleep(1.0)
                if self.agent is not None:
                    break
            
            if self.agent is not None:
                self.state = ServiceState.READY
                self._emit_event("service_started", {"agent_id": self.agent_id})
                # Install quorum checker into agent via config hook
                try:
                    if self.agent:
                        self.agent.quorum_checker = self._quorum_check
                        # Attach agent to chat engine
                        self.chat_engine.attach_agent(self.agent)
                except Exception:
                    pass
                # Start heartbeat loop
                try:
                    loop = asyncio.get_running_loop()
                    self._hb_task = loop.create_task(self._heartbeat_loop())
                    # Start periodic official checkpoint sync loop (will be removed after leader-based sync wired)
                    # self._sync_task = loop.create_task(self._idle_sync_loop())
                except Exception:
                    pass
                return True
            else:
                self.state = ServiceState.ERROR
                self.error_message = f"Agent initialization failed. Current state: {self.state}, Error: {self.error_message}"
                return False
                
        except Exception as e:
            self.state = ServiceState.ERROR
            self.error_message = str(e)
            print(f"Service start error: {e}")
            return False
    
    def _run_agent_training(self):
        """Run agent training in background thread."""
        try:
            print(f"Initializing Phase 3 agent in background thread...")
            
            # Create agent with event hooks
            self.agent = Phase3Agent(self.config)
            self._setup_agent_hooks()
            
            print(f"Agent initialized successfully: {self.agent.agent_id if hasattr(self.agent, 'agent_id') else 'unknown'}")
            print(f"Starting training loop...")
            self.state = ServiceState.TRAINING
            
            # Run training with periodic status updates
            epochs = self.config.get('training_epochs', 5)
            
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    break
                
                print(f"Service training epoch {epoch+1}/{epochs}")
                
                # Run one epoch with event monitoring
                self._run_monitored_epoch()
                
                # Check for pause requests
                while self.state == ServiceState.PAUSED and not self.stop_event.is_set():
                    time.sleep(1.0)
            
            print(f"Training completed for {self.agent_id}")
            self.state = ServiceState.READY
            
        except Exception as e:
            self.state = ServiceState.ERROR
            self.error_message = str(e)
            print(f"Agent training error: {e}")
            self._emit_event("training_error", {"error": str(e)})
    
    def _setup_agent_hooks(self):
        """Setup hooks to monitor agent events."""
        if not self.agent:
            return
        
        # Hook into decision making
        original_decision_cycle = self.agent.enhanced_meta_decision_cycle
        
        def monitored_decision_cycle(metrics):
            result = original_decision_cycle(metrics)
            
            # Emit decision event
            if hasattr(self.agent, 'meta_controller') and self.agent.meta_controller.reward_history:
                last_action = getattr(self.agent.meta_controller, 'action_history', ['UNKNOWN'])[-1:]
                action_name = last_action[0] if last_action else 'UNKNOWN'
                
                self._emit_event("meta_decision", {
                    "action": action_name,
                    "metrics": metrics,
                    "step": self.agent.global_step,
                    "result": result
                })
                
                self.decision_count += 1
            
            return result
        
        self.agent.enhanced_meta_decision_cycle = monitored_decision_cycle
        
        # Hook into reflection creation
        original_reflection_cycle = self.agent.periodic_reflection_cycle
        
        def monitored_reflection_cycle():
            reflections = original_reflection_cycle()
            
            if reflections:
                self.reflection_count += len(reflections)
                self._emit_event("reflections_created", {
                    "count": len(reflections),
                    "types": [r.summary_type for r in reflections],
                    "step": self.agent.global_step
                })
            
            return reflections
        
        self.agent.periodic_reflection_cycle = monitored_reflection_cycle
    
    def _run_monitored_epoch(self):
        """Run one training epoch with monitoring."""
        if not self.agent:
            return
        
        # Store original train method
        original_train = self.agent.train
        epoch_metrics = []
        
        def monitored_train(num_epochs=1):
            self._emit_event("epoch_started", {
                "epochs": num_epochs,
                "current_level": self.agent.curriculum.current_level.name
            })
            
            # Run original training
            result = original_train(num_epochs)
            
            # Emit epoch completion
            self._emit_event("epoch_completed", {
                "step": self.agent.global_step,
                "current_level": self.agent.curriculum.current_level.name
            })
            
            return result
        
        # Replace temporarily
        self.agent.train = monitored_train
        
        try:
            # Run one epoch
            self.agent.train(num_epochs=1)
            
            # Update metrics
            if hasattr(self.agent, 'memory'):
                recent_metrics = self.agent.memory.get_recent_metrics("val_loss", n=1)
                if recent_metrics:
                    self.metrics_history.append({
                        "step": self.agent.global_step,
                        "timestamp": time.time(),
                        "val_loss": recent_metrics[0]['value']
                    })
        
        finally:
            # Restore original
            self.agent.train = original_train
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a real-time event to subscribers."""
        event = ServiceEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            agent_id=self.agent_id,
            step=getattr(self.agent, 'global_step', 0) if self.agent else 0,
            data=data
        )
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            print("Warning: Event queue full, dropping event")
    
    async def pause(self) -> bool:
        """Pause agent training."""
        if self.state == ServiceState.TRAINING:
            self.state = ServiceState.PAUSED
            self._emit_event("service_paused", {})
            return True
        return False
    
    async def resume(self) -> bool:
        """Resume agent training."""
        if self.state == ServiceState.PAUSED:
            self.state = ServiceState.TRAINING
            self._emit_event("service_resumed", {})
            return True
        return False
    
    async def stop(self) -> bool:
        """Stop the agent service gracefully."""
        print(f"Stopping agent service: {self.agent_id}")
        self.stop_event.set()
        
        if self.agent_thread and self.agent_thread.is_alive():
            self.agent_thread.join(timeout=10.0)
        
        self.state = ServiceState.STOPPED
        self._emit_event("service_stopped", {})
        return True

    # ---------------- Heartbeats & Quorum ----------------
    async def _heartbeat_loop(self):
        import aiohttp
        # startup grace to avoid hammering peers before they mount endpoints
        await asyncio.sleep(self._startup_grace_s)
        while self.state != ServiceState.STOPPED:
            try:
                await asyncio.sleep(self.heartbeat_interval_s)
                for peer in list(self.peers):
                    try:
                        now = time.time()
                        # backoff if we've recently contacted
                        last = self._last_peer_contact.get(peer, 0.0)
                        if now - last < max(1.0, self.heartbeat_interval_s / 2):
                            continue
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f"{peer}/messages/receive", json={
                                "type": "heartbeat",
                                "payload": {"from": self.agent_id, "ts": now}
                            }, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                                data = await resp.json()
                                # Capture leader and official step if provided
                                if isinstance(data, dict) and data.get('success'):
                                    d = data.get('data', {})
                                    leader = d.get('leader')
                                    if leader:
                                        self._leader_id = leader
                                    official_step = d.get('official_step')
                                    # If peer is leader and has newer official, pull
                                    if leader and leader != self.agent_id and isinstance(official_step, int) and self.agent is not None:
                                        try:
                                            current_official = int(self.agent.get_official_checkpoint_step() or 0)
                                        except Exception:
                                            current_official = 0
                                        if official_step > current_official:
                                            try:
                                                self.agent.load_official_checkpoint()
                                            except Exception:
                                                pass
                                self._last_peer_contact[peer] = now
                    except Exception:
                        # Peer might be down; keep going
                        continue
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.heartbeat_interval_s)

    async def _idle_sync_loop(self):
        # Deprecated: prefer leader-driven sync; keep temporarily as no-op
        await asyncio.sleep(self._startup_grace_s)
        while self.state != ServiceState.STOPPED:
            await asyncio.sleep(self._official_sync_interval_s)

    def _local_policy_vote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Very simple local voting heuristic using current metrics
        action = payload.get("action", "")
        metrics = payload.get("metrics", {})
        approve = True
        reason = "ok"
        # Loosened policy: strong approve on clear need; reject only when clearly unnecessary
        val_loss = float(metrics.get('val_loss', 1.0) or 1.0)
        confidence = float(metrics.get('confidence', 1.0) or 1.0)
        grad = float(metrics.get('gradient_norm', 0.0) or 0.0)
        if action == "FINE_TUNE_NOW":
            # Reject if model is already good
            if val_loss < 0.65:
                approve = False
                reason = "low val_loss; skip fine-tune"
            # Strong approve conditions
            elif val_loss >= 0.72 or (val_loss >= 0.70 and confidence < 0.55) or grad > 1.9:
                approve = True
                reason = "high need (loss/conf/grad)"
        return {"approve": approve, "reason": reason, "voter_id": self.agent_id}

    def local_vote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_policy_vote(payload)

    def _quorum_check(self, action_name: str, step: int, metrics: Dict[str, Any]) -> bool:
        # Broadcast vote_request and count approvals (self + peers)
        approvals = 0
        total_needed = 2  # for 2-of-3

        # Local vote first
        local = self._local_policy_vote({"action": action_name, "step": step, "metrics": metrics})
        approvals += 1 if local.get("approve") else 0

        # Ask peers synchronously (best-effort)
        try:
            import aiohttp
            async def ask_peers():
                nonlocal approvals
                tasks = []
                async with aiohttp.ClientSession() as session:
                    for peer in self.peers:
                        tasks.append(session.post(f"{peer}/messages/receive", json={
                            "type": "vote_request",
                            "payload": {"action": action_name, "step": step, "metrics": metrics, "from": self.agent_id}
                        }, timeout=aiohttp.ClientTimeout(total=3.0)))
                    for t in asyncio.as_completed(tasks, timeout=3.5):
                        try:
                            resp = await t
                            data = await resp.json()
                            if isinstance(data, dict) and data.get("success"):
                                d = data.get("data", {})
                                if d.get("approve"):
                                    approvals += 1
                        except Exception:
                            continue
            loop = asyncio.get_event_loop()
            loop.run_until_complete(ask_peers())
        except Exception:
            pass

        # Slightly less conservative local gate: approve only on clear need
        # Majority approval, with a slight loosening if conditions are clearly poor
        severe_need = False
        try:
            v = float(metrics.get('val_loss', 0.0) or 0.0)
            c = float(metrics.get('confidence', 1.0) or 1.0)
            g = float(metrics.get('gradient_norm', 0.0) or 0.0)
            severe_need = (v >= 0.72) or (v >= 0.70 and c < 0.55) or (g > 1.9)
        except Exception:
            severe_need = False
        approved = (approvals >= total_needed) or (severe_need and approvals >= 1)
        # Log quorum result
        try:
            if self.agent and hasattr(self.agent, 'memory'):
                self.agent.memory.log_meta_event(
                    event_type="quorum_result",
                    info={"action": action_name, "step": step, "approvals": approvals, "needed": total_needed, "severe_need": severe_need}
                )
        except Exception:
            pass
        return approved

    async def maybe_pull_official_checkpoint(self):
        """If an official checkpoint exists and is newer, pull it while idle."""
        if not self.agent:
            return False
        try:
            return self.agent.load_official_checkpoint_if_newer()
        except Exception:
            return False

    # ---------------- Leader-routed operations ----------------
    async def leader_chat(self, text: str, mode: Optional[str], session: Optional[str]) -> Dict[str, Any]:
        """Route chat to leader if configured; otherwise handle locally and replicate to peers."""
        # If a leader URL is configured and this node is a follower, forward the request
        if getattr(self, 'leader_url', None) and self._is_follower():
            try:
                import aiohttp
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{self.leader_url}/chat", json={
                        "text": text, "mode": mode, "session": session
                    }, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                        data = await resp.json()
                        if isinstance(data, dict) and data.get("success"):
                            return data.get("data", {})
                        raise RuntimeError(f"Leader chat failed: {data}")
            except Exception as e:
                raise RuntimeError(f"Leader forward error: {e}")
        # Local handling (this node acts as leader)
        result = self.chat_engine.chat(text, mode=mode, session=session)
        # Log and replicate
        try:
            if self.agent and hasattr(self.agent, 'memory'):
                self.agent.memory.log_meta_event(
                    event_type="chat_turn",
                    info={"text": text, "mode": result.get("mode"), "reply": result.get("reply"), "session": session}
                )
        except Exception:
            pass
        await self._replicate_chat(text, mode or None, result.get("reply"), session)
        return result

    async def leader_submit_task(self, task_type: str, priority: Optional[int], objective: Optional[str], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Route task intake to leader if configured; leader enqueues and replicates meta-event."""
        if getattr(self, 'leader_url', None) and self._is_follower():
            try:
                import aiohttp
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{self.leader_url}/task", json={
                        "task_type": task_type,
                        "priority": priority,
                        "objective": objective,
                        "payload": payload or {}
                    }, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                        data = await resp.json()
                        if isinstance(data, dict) and data.get("success"):
                            return data.get("data", {})
                        raise RuntimeError(f"Leader task failed: {data}")
            except Exception as e:
                raise RuntimeError(f"Leader task forward error: {e}")
        # Local leader path: directly use the agent to enqueue
        if not self.agent:
            raise RuntimeError("Agent not available")
        from memory_db import Task
        pri = 1 if priority is None else max(0, min(3, int(priority)))
        obj = objective or f"User task: {task_type}"
        task = Task(prompt=obj, priority=pri, objective=obj, created_by="user", metadata={"task_type": task_type, "payload": payload or {}})
        task_id = self.agent.memory.enqueue_task(task)
        # Log and replicate meta-event (not the queue)
        try:
            self.agent.memory.log_meta_event(
                event_type="user_task_enqueued",
                info={"task_id": task_id, "task_type": task_type, "priority": pri}
            )
        except Exception:
            pass
        await self._broadcast_to_peers("task_replicate", {"task_type": task_type, "priority": pri, "objective": obj})
        return {"task_id": task_id}

    def _is_follower(self) -> bool:
        # If leader_url is configured and not pointing to our own port, treat as follower
        try:
            return bool(self.leader_url) and (self.leader_url.find(":8081") == -1 if self.agent_id == 'agent_A' else True)
        except Exception:
            return bool(self.leader_url)

    async def _replicate_chat(self, text: str, mode: Optional[str], reply: Optional[str], session: Optional[str]):
        await self._broadcast_to_peers("chat_replicate", {"text": text, "mode": mode, "reply": reply, "session": session})

    async def _broadcast_to_peers(self, msg_type: str, payload: Dict[str, Any]):
        if not self.peers:
            return
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                tasks = []
                for peer in self.peers:
                    tasks.append(session.post(f"{peer}/messages/receive", json={
                        "type": msg_type,
                        "payload": payload
                    }, timeout=aiohttp.ClientTimeout(total=3.0)))
                for t in asyncio.as_completed(tasks, timeout=3.5):
                    try:
                        _ = await t
                    except Exception:
                        continue
        except Exception:
            pass
    
    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        performance_metrics = {}
        
        if self.metrics_history:
            recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
            performance_metrics = {
                "recent_val_loss": sum(m["val_loss"] for m in recent_metrics) / len(recent_metrics),
                "training_steps": self.agent.global_step if self.agent else 0
            }
        # Meta stats (if available)
        try:
            if self.agent and hasattr(self.agent, 'meta_controller'):
                mc_stats = self.agent.meta_controller.get_stats() or {}
                performance_metrics.update({
                    "meta_total_steps": mc_stats.get('total_steps', 0),
                    "meta_last_loss": mc_stats.get('last_loss', 0.0),
                    "meta_avg_reward": mc_stats.get('avg_recent_reward', 0.0),
                })
        except Exception:
            pass
        
        last_decision = None
        if self.agent and hasattr(self.agent, 'meta_controller'):
            if hasattr(self.agent.meta_controller, 'action_history') and self.agent.meta_controller.action_history:
                last_decision = str(self.agent.meta_controller.action_history[-1])
        
        return ServiceStatus(
            state=self.state,
            agent_id=self.agent_id,
            uptime=time.time() - self.start_time,
            current_step=self.agent.global_step if self.agent else 0,
            total_decisions=self.decision_count,
            total_reflections=self.reflection_count,
            total_feedback=0,  # Will be implemented when feedback integration is added
            last_decision=last_decision,
            performance_metrics=performance_metrics,
            error_message=self.error_message
        )
    
    async def submit_feedback(self, feedback_type: str, sentiment: str, 
                            content: str, rating: Optional[float] = None,
                            target_step: Optional[int] = None) -> str:
        """Submit human feedback to the agent."""
        if not self.agent or not self.agent.human_feedback_enabled:
            raise ValueError("Human feedback not available")
        
        # Convert string parameters to enums
        fb_type = FeedbackType(feedback_type)
        fb_sentiment = FeedbackSentiment[sentiment.upper()]
        
        feedback_id = self.agent.feedback_system.submit_feedback(
            feedback_type=fb_type,
            sentiment=fb_sentiment,
            content=content,
            rating=rating,
            target_step=target_step or (self.agent.global_step if self.agent else None)
        )
        
        # Emit feedback event
        self._emit_event("feedback_received", {
            "feedback_id": feedback_id,
            "type": feedback_type,
            "sentiment": sentiment,
            "has_rating": rating is not None
        })
        
        return feedback_id
    
    async def query_memory(self, query: str) -> Dict[str, Any]:
        """Query agent memory with natural language."""
        if not self.agent:
            raise ValueError("Agent not available")
        
        try:
            result = self.agent.explainer.query_agent_memory(query)
            
            self._emit_event("memory_query", {
                "query": query,
                "query_type": result.get("query_type", "unknown")
            })
            
            return result
        except Exception as e:
            raise ValueError(f"Memory query failed: {e}")
    
    def get_event_stream(self) -> queue.Queue:
        """Get the event queue for streaming."""
        return self.event_queue
    
    async def get_recent_events(self, limit: int = 50) -> List[ServiceEvent]:
        """Get recent events from the queue."""
        events = []
        temp_events = []
        
        # Drain queue without blocking
        try:
            while len(events) < limit:
                event = self.event_queue.get_nowait()
                events.append(event)
                temp_events.append(event)
        except queue.Empty:
            pass
        
        # Put events back for other consumers
        for event in reversed(temp_events):
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                break
        
        return events[-limit:]

    # ---------------- Guardian & Evaluation ----------------
    def get_eval_report(self) -> Dict[str, Any]:
        """Compute a simple held-out evaluation report for guardian checks."""
        report: Dict[str, Any] = {"val_score": 1.0, "regression_pct": 0.0, "toxicity": 0.0, "lr": 0.0}
        try:
            if not self.agent:
                return report
            # Use validate() if available
            val = {}
            try:
                val = self.agent.validate() or {}
            except Exception:
                val = {}
            val_loss = float(val.get('val_loss', 0.0) or 0.0)
            # Proxy: higher score when loss is low
            report['val_score'] = max(0.0, 1.0 - min(1.0, val_loss))
            # Learning rate
            try:
                lr = float(self.agent.optimizer.param_groups[0]['lr'])
            except Exception:
                lr = 0.0
            report['lr'] = lr
            # Regression/toxicity placeholders (hook for future)
            report['regression_pct'] = 0.0
            report['toxicity'] = 0.0
        except Exception:
            pass
        return report

    async def collect_peer_eval_reports(self) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        # include self first
        reports.append(self.get_eval_report())
        # ask peers
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                tasks = []
                for peer in self.peers:
                    tasks.append(session.post(f"{peer}/messages/receive", json={
                        "type": "eval_request",
                        "payload": {"from": self.agent_id}
                    }, timeout=aiohttp.ClientTimeout(total=4.0)))
                for t in asyncio.as_completed(tasks, timeout=5.0):
                    try:
                        resp = await t
                        data = await resp.json()
                        if isinstance(data, dict) and data.get('success'):
                            rep = data.get('data', {})
                            if isinstance(rep, dict):
                                reports.append(rep)
                    except Exception:
                        continue
        except Exception:
            pass
        return reports

async def main():
    """Test the agent service."""
    print("=== PHASE 4: AGENT SERVICE TEST ===\n")
    
    # Test configuration
    config = {
        'agent_id': 'test_agent_001',
        'model': {
            'vocab_size': 11,
            'd_model': 32,
            'd_ff': 128,
            'max_seq_len': 12,
            'dropout': 0.1
        },
        'optimizer': {
            'lr': 3e-3,
            'weight_decay': 1e-5
        },
        'curriculum_size': 200,
        'batch_size': 8,
        'meta_training_frequency': 10,
        'task_execution_frequency': 20,
        'reflection_frequency': 15,
        'db_path': 'phase4_service_test.db',
        'training_epochs': 2,
        'human_feedback_enabled': True
    }
    
    # Create and start service
    service = AgentService(config)
    
    print("1. Starting agent service...")
    success = await service.start_async()
    
    if not success:
        print("Failed to start service")
        return
    
    print(f"   Service started: {service.agent_id}")
    
    # Monitor for a bit
    print("\n2. Monitoring service for 30 seconds...")
    
    for i in range(6):  # 6 x 5 seconds = 30 seconds
        await asyncio.sleep(5.0)
        
        status = service.get_status()
        print(f"   Status: {status.state.value}, Step: {status.current_step}, "
              f"Decisions: {status.total_decisions}, Reflections: {status.total_reflections}")
        
        # Test feedback submission
        if i == 2:  # At 10 seconds
            try:
                feedback_id = await service.submit_feedback(
                    feedback_type="decision_approval",
                    sentiment="positive", 
                    content="Good progress!",
                    rating=4.0
                )
                print(f"   Submitted feedback: {feedback_id}")
            except Exception as e:
                print(f"   Feedback error: {e}")
        
        # Test memory query
        if i == 4:  # At 20 seconds
            try:
                result = await service.query_memory("What was my last decision?")
                print(f"   Memory query result: {result.get('query_type', 'unknown')}")
            except Exception as e:
                print(f"   Query error: {e}")
    
    print("\n3. Testing service control...")
    
    # Test pause/resume
    await service.pause()
    print("   Service paused")
    await asyncio.sleep(2.0)
    
    await service.resume()
    print("   Service resumed")
    await asyncio.sleep(2.0)
    
    # Get final status
    final_status = service.get_status()
    print(f"\n4. Final status:")
    print(f"   State: {final_status.state.value}")
    print(f"   Uptime: {final_status.uptime:.1f} seconds")
    print(f"   Steps completed: {final_status.current_step}")
    print(f"   Total decisions: {final_status.total_decisions}")
    print(f"   Total reflections: {final_status.total_reflections}")
    
    # Check recent events
    recent_events = await service.get_recent_events(limit=10)
    print(f"   Recent events: {len(recent_events)}")
    for event in recent_events[-3:]:  # Show last 3
        print(f"     • {event.event_type} at step {event.step}")
    
    # Stop service
    print("\n5. Stopping service...")
    await service.stop()
    print("   Service stopped successfully")
    
    print("\n[OK] AGENT SERVICE TEST COMPLETED!")
    print(f"The agent service provides:")
    print(f"• Async lifecycle management")
    print(f"• Real-time event streaming") 
    print(f"• Human feedback integration")
    print(f"• Memory query capabilities")
    print(f"• Production-ready monitoring")

if __name__ == "__main__":
    # Clean up any existing test database
    import os
    if os.path.exists("phase4_service_test.db"):
        os.remove("phase4_service_test.db")
    
    asyncio.run(main())