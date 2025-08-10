#!/usr/bin/env python3
"""
Phase 4: HTTP API Server - RESTful interface for the self-aware AI agent service.
Provides production-ready endpoints for agent control, introspection, and human feedback.
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import traceback

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Using simple HTTP server.")

# Optional metrics
try:
    from starlette_exporter import PrometheusMiddleware, handle_metrics
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

# Optional rate limiting
try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True
except Exception:
    SLOWAPI_AVAILABLE = False

from phase4_agent_service import AgentService, ServiceState, ServiceStatus
from memory_db import Task  # for /task enqueue

# Request/Response Models
class FeedbackRequest(BaseModel):
    feedback_type: str  # "decision_approval", "performance_rating", etc.
    sentiment: str      # "positive", "neutral", "negative"
    content: str
    rating: Optional[float] = None
    target_step: Optional[int] = None

class MemoryQueryRequest(BaseModel):
    query: str
    context_limit: Optional[int] = 10

class AgentControlRequest(BaseModel):
    action: str  # "start", "pause", "resume", "stop"

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float

class Phase4APIServer:
    """
    Phase 4: HTTP API Server for the self-aware agent.
    
    Features:
    - RESTful endpoints for agent control
    - Real-time WebSocket event streaming  
    - Human feedback integration
    - Memory querying and introspection
    - Health monitoring and metrics
    """
    
    def __init__(self, agent_config: Dict[str, Any], port: int = 8080):
        self.agent_config = agent_config
        self.port = port
        
        # Agent service
        self.agent_service: Optional[AgentService] = None
        self.is_running = False
        
        # WebSocket connections
        self.websocket_connections = []
        
        # Create FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints."""
        app = FastAPI(
            title="Phase 4: Self-Aware AI Agent API",
            description="Production-ready API for recursive self-reflective AI agent",
            version="4.0.0"
        )
        
        # CORS middleware for web dashboard integration
        cors_env = os.getenv("RA_CORS")
        if cors_env:
            allow_list = [c.strip() for c in cors_env.split(",") if c.strip()]
        else:
            allow_list = ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_list,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["authorization", "content-type"],
        )

        # Prometheus metrics
        if METRICS_AVAILABLE:
            app.add_middleware(PrometheusMiddleware, app_name="theta-iota")
            app.add_route("/metrics", handle_metrics)

        # Rate limiting
        limiter = None
        if SLOWAPI_AVAILABLE:
            rate = os.getenv("RA_RATE_LIMIT", "20/minute")
            limiter = Limiter(key_func=get_remote_address, default_limits=[rate])
            app.state.limiter = limiter
            app.add_middleware(SlowAPIMiddleware)

            @app.exception_handler(RateLimitExceeded)
            def _rate_limit_handler(request: Request, exc: RateLimitExceeded):  # type: ignore
                return JSONResponse(status_code=429, content={"success": False, "message": "rate limit exceeded"})

        # Simple bearer token auth
        def _require_auth(request: Request):
            secret = os.getenv("RA_AUTH_SECRET")
            if not secret:
                return True  # disabled
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="missing bearer token")
            token = auth[7:].strip()
            if token != secret:
                raise HTTPException(status_code=403, detail="invalid token")
            return True

        # Healthz/readyz
        @app.get("/healthz")
        async def healthz():
            return APIResponse(success=True, message="ok", data={"pid": os.getpid()}, timestamp=time.time())

        @app.get("/readyz")
        async def readyz():
            ok = bool(self.agent_service is not None)
            detail = {"agent_connected": ok}
            try:
                if ok and self.agent_service.agent and hasattr(self.agent_service.agent, 'memory'):
                    cur = self.agent_service.agent.memory.conn.cursor()
                    cur.execute("SELECT 1")
                    _ = cur.fetchone()
                    detail["db"] = True
                else:
                    detail["db"] = False
            except Exception:
                detail["db"] = False
            return APIResponse(success=ok and detail.get("db", False), message="ready" if ok else "not ready", data=detail, timestamp=time.time())
        
        # Agent Control Endpoints
        @app.post("/agent/start")
        async def start_agent(auth_ok: bool = Depends(_require_auth)):
            """Start the agent service."""
            try:
                if self.agent_service is None:
                    self.agent_service = AgentService(self.agent_config)
                
                if self.agent_service.state in [ServiceState.STOPPED, ServiceState.ERROR]:
                    self.agent_service = AgentService(self.agent_config)
                
                success = await self.agent_service.start_async()
                
                if success:
                    # Start event broadcasting
                    asyncio.create_task(self._broadcast_events())
                    
                    return APIResponse(
                        success=True,
                        message=f"Agent {self.agent_service.agent_id} started successfully",
                        data={"agent_id": self.agent_service.agent_id},
                        timestamp=time.time()
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to start agent: {self.agent_service.error_message}"
                    )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/agent/pause")
        async def pause_agent(auth_ok: bool = Depends(_require_auth)):
            """Pause agent training."""
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            success = await self.agent_service.pause()
            return APIResponse(
                success=success,
                message="Agent paused" if success else "Failed to pause agent",
                timestamp=time.time()
            )
        
        @app.post("/agent/resume") 
        async def resume_agent(auth_ok: bool = Depends(_require_auth)):
            """Resume agent training."""
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            success = await self.agent_service.resume()
            return APIResponse(
                success=success,
                message="Agent resumed" if success else "Failed to resume agent",
                timestamp=time.time()
            )
        
        @app.post("/agent/stop")
        async def stop_agent(auth_ok: bool = Depends(_require_auth)):
            """Stop agent service."""
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            success = await self.agent_service.stop()
            return APIResponse(
                success=success,
                message="Agent stopped" if success else "Failed to stop agent",
                timestamp=time.time()
            )
        
        @app.get("/agent/status")
        async def get_agent_status():
            """Get current agent status."""
            if not self.agent_service:
                return APIResponse(
                    success=False,
                    message="Agent service not initialized",
                    data={"state": "not_initialized"},
                    timestamp=time.time()
                )
            
            status = self.agent_service.get_status()
            return APIResponse(
                success=True,
                message="Status retrieved",
                data={
                    "state": status.state.value,
                    "agent_id": status.agent_id,
                    "uptime": status.uptime,
                    "current_step": status.current_step,
                    "total_decisions": status.total_decisions,
                    "total_reflections": status.total_reflections,
                    "total_feedback": status.total_feedback,
                    "last_decision": status.last_decision,
                    "performance_metrics": status.performance_metrics,
                    "error_message": status.error_message
                },
                timestamp=time.time()
            )
        
        # Human Feedback Endpoints
        @app.post("/feedback/submit")
        async def submit_feedback(request: FeedbackRequest, auth_ok: bool = Depends(_require_auth)):
            """Submit human feedback to the agent."""
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            try:
                feedback_id = await self.agent_service.submit_feedback(
                    feedback_type=request.feedback_type,
                    sentiment=request.sentiment,
                    content=request.content,
                    rating=request.rating,
                    target_step=request.target_step
                )
                
                return APIResponse(
                    success=True,
                    message="Feedback submitted successfully",
                    data={"feedback_id": feedback_id},
                    timestamp=time.time()
                )
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/feedback/requests")
        async def get_feedback_requests():
            """Get pending feedback requests."""
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            
            try:
                requests = self.agent_service.agent.feedback_system.get_pending_feedback_requests()
                return APIResponse(
                    success=True,
                    message=f"Found {len(requests)} pending requests",
                    data={"requests": requests},
                    timestamp=time.time()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Memory & Introspection Endpoints
        @app.post("/memory/query")
        async def query_memory(request: MemoryQueryRequest):
            """Query agent memory with natural language."""
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            try:
                result = await self.agent_service.query_memory(request.query)
                return APIResponse(
                    success=True,
                    message="Memory query completed",
                    data=result,
                    timestamp=time.time()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Chat Endpoint (native, no external runtimes)
        class ChatRequest(BaseModel):
            text: str
            mode: Optional[str] = None   # "reflect" (default) or "lm"
            session: Optional[str] = None
            temperature: Optional[float] = None
            top_k: Optional[int] = None

        @app.post("/chat")
        async def chat(req: ChatRequest):
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            try:
                # Apply optional decode controls
                if req.temperature is not None and self.agent_service.chat_engine and self.agent_service.chat_engine.config:
                    self.agent_service.chat_engine.config.temperature = float(req.temperature)
                if req.top_k is not None and self.agent_service.chat_engine and self.agent_service.chat_engine.config:
                    self.agent_service.chat_engine.config.top_k = int(req.top_k)
                # Route via leader-aware handler
                result = await self.agent_service.leader_chat(req.text, req.mode, req.session)
                return APIResponse(success=True, message="ok", data=result, timestamp=time.time())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/memory/decisions")
        async def get_recent_decisions(limit: int = 10):
            """Get recent decision history."""
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            
            try:
                # Get recent decisions from database
                cursor = self.agent_service.agent.memory.conn.cursor()
                cursor.execute("""
                    SELECT * FROM meta_events 
                    WHERE event_type = 'meta_decision' 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                decisions = []
                for row in cursor.fetchall():
                    decision_data = dict(row)
                    decision_data['info'] = json.loads(decision_data['info_json'])
                    del decision_data['info_json']  # Remove raw JSON
                    decisions.append(decision_data)
                
                return APIResponse(
                    success=True,
                    message=f"Retrieved {len(decisions)} recent decisions",
                    data={"decisions": decisions},
                    timestamp=time.time()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/memory/reflections")
        async def get_reflections(limit: int = 10):
            """Get recent reflection artifacts."""
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            
            try:
                reflections = self.agent_service.agent.summarizer.get_recent_reflections(limit)
                reflection_data = []
                
                for reflection in reflections:
                    reflection_data.append({
                        "id": reflection.id,
                        "summary_type": reflection.summary_type,
                        "trigger": reflection.trigger,
                        "step_range": [reflection.start_step, reflection.end_step],
                        "insights": reflection.insights[:3],  # Limit for API response
                        "action_recommendations": reflection.action_recommendations[:3],
                        "created_at": reflection.created_at
                    })
                
                return APIResponse(
                    success=True,
                    message=f"Retrieved {len(reflection_data)} reflections",
                    data={"reflections": reflection_data},
                    timestamp=time.time()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Real-time Event Streaming
        @app.websocket("/stream/events")
        async def websocket_events(websocket: WebSocket):
            """WebSocket endpoint for real-time event streaming."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send initial status
                if self.agent_service:
                    status = self.agent_service.get_status()
                    await websocket.send_text(json.dumps({
                        "event_type": "status_update",
                        "data": asdict(status),
                        "timestamp": time.time()
                    }))
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(1.0)
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
        
        # Health Check
        @app.get("/health")
        async def health_check():
            """API health check."""
            return APIResponse(
                success=True,
                message="API server is healthy",
                data={
                    "server_uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                    "agent_connected": self.agent_service is not None,
                    "agent_state": self.agent_service.state.value if self.agent_service else "disconnected"
                },
                timestamp=time.time()
            )

        # Ops: pull official checkpoint if newer (manual, safe)
        @app.post("/checkpoint/pull_if_newer")
        async def checkpoint_pull_if_newer():
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            ok = self.agent_service.agent.load_official_checkpoint_if_newer()
            return APIResponse(success=ok, message="pulled" if ok else "no newer official", timestamp=time.time())

        # Minimal messaging endpoint for inter-agent communication
        class MessageEnvelope(BaseModel):
            type: str
            payload: Dict[str, Any]

        @app.post("/messages/receive")
        async def messages_receive(msg: MessageEnvelope):
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            try:
                if msg.type == "heartbeat":
                    # Include leader hint and official step for convergence
                    leader = getattr(self.agent_service, '_leader_id', None) or self.agent_service.agent_id
                    official_step = 0
                    try:
                        if self.agent_service.agent:
                            official_step = int(self.agent_service.agent.get_official_checkpoint_step() or 0)
                    except Exception:
                        pass
                    return APIResponse(success=True, message="ok", data={"alive": True, "leader": leader, "official_step": official_step}, timestamp=time.time())
                if msg.type == "vote_request":
                    decision = self.agent_service.local_vote(msg.payload)
                    return APIResponse(success=True, message="ok", data=decision, timestamp=time.time())
                if msg.type == "eval_request":
                    rep = self.agent_service.get_eval_report()
                    return APIResponse(success=True, message="ok", data=rep, timestamp=time.time())
                if msg.type == "chat_replicate":
                    # Optionally store replicated chat turns locally for shared memory view
                    try:
                        p = msg.payload
                        if self.agent_service.agent and hasattr(self.agent_service.agent, 'memory'):
                            self.agent_service.agent.memory.log_meta_event(
                                event_type="chat_turn_replicated",
                                info={"text": p.get("text"), "mode": p.get("mode"), "reply": p.get("reply"), "session": p.get("session")}
                            )
                    except Exception:
                        pass
                    return APIResponse(success=True, message="ok", data={"replicated": True}, timestamp=time.time())
                if msg.type == "task_replicate":
                    # Minimal acknowledgement for task replication
                    return APIResponse(success=True, message="ok", data={"replicated": True}, timestamp=time.time())
                return APIResponse(success=False, message="unknown message type", data={"type": msg.type}, timestamp=time.time())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Task intake: accept structured tasks and enqueue into agent's task queue
        class TaskRequest(BaseModel):
            task_type: str
            payload: Dict[str, Any] | None = None
            priority: int | None = None
            objective: str | None = None

        @app.post("/task")
        async def submit_task(req: TaskRequest, auth_ok: bool = Depends(_require_auth)):
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            try:
                data = await self.agent_service.leader_submit_task(req.task_type, req.priority, req.objective, req.payload)
                return APIResponse(success=True, message="task enqueued", data=data, timestamp=time.time())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Quorum test: trigger a vote round without changing weights
        class QuorumTestRequest(BaseModel):
            action: str = "FINE_TUNE_NOW"
            metrics: Dict[str, Any] = {}

        @app.post("/quorum/test")
        async def quorum_test(req: QuorumTestRequest, auth_ok: bool = Depends(_require_auth)):
            if not self.agent_service:
                raise HTTPException(status_code=404, detail="Agent service not found")
            try:
                step = 0
                if self.agent_service.agent is not None:
                    step = int(getattr(self.agent_service.agent, 'global_step', 0) or 0)
                approved = self.agent_service._quorum_check(req.action, step, req.metrics)
                # Log explicitly
                try:
                    if self.agent_service.agent and hasattr(self.agent_service.agent, 'memory'):
                        self.agent_service.agent.memory.log_meta_event(
                            event_type="quorum_test",
                            info={"action": req.action, "metrics": req.metrics, "approved": approved}
                        )
                except Exception:
                    pass
                return APIResponse(success=True, message="ok", data={"approved": approved}, timestamp=time.time())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Official checkpoint: promote current or pull
        class CheckpointAction(BaseModel):
            op: str  # 'promote' or 'pull'

        @app.post("/checkpoint")
        async def checkpoint_op(req: CheckpointAction, auth_ok: bool = Depends(_require_auth)):
            if not self.agent_service or not self.agent_service.agent:
                raise HTTPException(status_code=404, detail="Agent not available")
            try:
                if req.op == 'promote':
                    # Guardian: collect reports across peers
                    reports = await self.agent_service.collect_peer_eval_reports()
                    ok_guard = True
                    reason = None
                    rules_path = os.getenv("RA_RULES")
                    if rules_path:
                        try:
                            from guardian.validator import validate_report
                            # require 2-of-3 pass
                            passes = 0
                            for rep in reports:
                                good, why = validate_report(rep, rules_path)
                                passes += 1 if good else 0
                                if not good and reason is None:
                                    reason = why
                            ok_guard = (passes >= 2)
                        except Exception as e:
                            ok_guard = False
                            reason = f"guardian_error:{e}"
                    if not ok_guard:
                        return APIResponse(success=False, message=f"guardian_failed:{reason}", timestamp=time.time())
                    path = self.agent_service.agent.save_official_checkpoint_from_current()
                    if not path:
                        raise ValueError("promotion failed")
                    return APIResponse(success=True, message="promoted", data={"path": path}, timestamp=time.time())
                elif req.op == 'pull':
                    ok = self.agent_service.agent.load_official_checkpoint()
                    return APIResponse(success=ok, message="pulled" if ok else "no official found", timestamp=time.time())
                else:
                    raise ValueError("op must be 'promote' or 'pull'")
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        return app
    
    async def _broadcast_events(self):
        """Broadcast agent events to WebSocket connections."""
        if not self.agent_service:
            return
        
        event_queue = self.agent_service.get_event_stream()
        
        while self.agent_service.state != ServiceState.STOPPED:
            try:
                # Get events from queue (non-blocking)
                events = []
                while len(events) < 10:  # Batch up to 10 events
                    try:
                        event = event_queue.get_nowait()
                        events.append(event)
                    except:
                        break
                
                # Broadcast events to connected clients
                if events and self.websocket_connections:
                    for event in events:
                        event_json = event.to_json()
                        
                        # Send to all connected WebSocket clients
                        disconnected = []
                        for websocket in self.websocket_connections:
                            try:
                                await websocket.send_text(event_json)
                            except:
                                disconnected.append(websocket)
                        
                        # Remove disconnected clients
                        for ws in disconnected:
                            if ws in self.websocket_connections:
                                self.websocket_connections.remove(ws)
                
                await asyncio.sleep(0.1)  # Small delay between broadcasts
                
            except Exception as e:
                print(f"Event broadcast error: {e}")
                await asyncio.sleep(1.0)
    
    async def start_server(self):
        """Start the API server."""
        if not FASTAPI_AVAILABLE:
            print("FastAPI not available. Please install: pip install fastapi uvicorn")
            return False
        
        self.start_time = time.time()
        self.is_running = True
        
        print(f"Starting Phase 4 API Server on port {self.port}")
        print(f"Agent config: {self.agent_config.get('agent_id', 'unknown')}")
        
        # Start server with uvicorn
        config = uvicorn.Config(
            app=self.app,
            host="127.0.0.1", 
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_server(self):
        """Stop the API server."""
        self.is_running = False
        
        if self.agent_service:
            await self.agent_service.stop()
        
        # Close WebSocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except:
                pass
        
        self.websocket_connections.clear()

async def main():
    """Test the API server."""
    if not FASTAPI_AVAILABLE:
        print("Please install FastAPI to test the API server:")
        print("pip install fastapi uvicorn")
        return
    
    print("=== PHASE 4: API SERVER TEST ===\n")
    
    # Test configuration
    config = {
        'agent_id': 'api_test_agent',
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
        'db_path': 'phase4_api_test.db',
        'training_epochs': 1,
        'human_feedback_enabled': True
    }
    
    # Create and start API server
    server = Phase4APIServer(config, port=8080)
    
    print("Starting API server on http://127.0.0.1:8080")
    print("Available endpoints:")
    print("  POST /agent/start - Start the agent")
    print("  GET  /agent/status - Get agent status")
    print("  POST /feedback/submit - Submit feedback")
    print("  POST /memory/query - Query agent memory")
    print("  WS   /stream/events - Real-time events")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        await server.stop_server()

if __name__ == "__main__":
    # Clean up test database
    import os
    if os.path.exists("phase4_api_test.db"):
        os.remove("phase4_api_test.db")
    
    asyncio.run(main())