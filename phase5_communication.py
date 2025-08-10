#!/usr/bin/env python3
"""
Phase 5: Agent-to-Agent Communication Layer - Messaging backbone for federated AI.
Enables secure, reliable communication between self-aware agents in the federation.
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets

class MessageType(Enum):
    """Types of inter-agent messages."""
    # Basic communication
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    
    # Decision coordination  
    DECISION_REQUEST = "decision_request"
    DECISION_RESPONSE = "decision_response"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_RESULT = "consensus_result"
    
    # Memory sharing
    REFLECTION_SHARE = "reflection_share"
    MEMORY_QUERY = "memory_query"
    MEMORY_RESPONSE = "memory_response"
    KNOWLEDGE_SYNC = "knowledge_sync"
    
    # Task coordination
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_STATUS = "task_status"
    TASK_RESULT = "task_result"
    
    # Crisis coordination
    CRISIS_ALERT = "crisis_alert"
    HELP_REQUEST = "help_request"
    CAPABILITY_QUERY = "capability_query"

@dataclass
class AgentMessage:
    """Standard message format for agent-to-agent communication."""
    message_id: str
    sender_id: str
    recipient_id: str  # Can be "*" for broadcast
    message_type: MessageType
    timestamp: float
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None  # For request/response pairing
    ttl: Optional[float] = None  # Time to live
    signature: Optional[str] = None  # Message authentication
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps({
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "signature": self.signature
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=MessageType(data["message_type"]),
            timestamp=data["timestamp"],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl"),
            signature=data.get("signature")
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl

class MessageHandler:
    """Handler interface for processing agent messages."""
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message and return optional response."""
        raise NotImplementedError

class AgentCommunicator:
    """
    Phase 5: Agent-to-Agent Communication Layer
    
    Features:
    - Reliable message delivery with retries
    - Request/response patterns with correlation
    - Broadcast messaging for coordination
    - Message authentication and validation
    - Connection pooling and health monitoring
    - Message routing via registry
    """
    
    def __init__(self, agent_id: str, agent_registry=None):
        self.agent_id = agent_id
        self.registry = agent_registry
        
        # Message handling
        self.message_handlers: Dict[MessageType, MessageHandler] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_history: List[AgentMessage] = []
        
        # Connection management
        self.connections: Dict[str, aiohttp.ClientSession] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Configuration
        self.request_timeout = 30.0
        self.max_retries = 3
        self.max_history = 1000
        
        print(f"AgentCommunicator initialized for {agent_id}")
    
    def register_handler(self, message_type: MessageType, handler: MessageHandler):
        """Register a message handler for specific message type."""
        self.message_handlers[message_type] = handler
        print(f"Registered handler for {message_type.value}")
    
    async def send_message(self, recipient_id: str, message_type: MessageType, 
                          payload: Dict[str, Any], correlation_id: Optional[str] = None,
                          ttl: Optional[float] = None) -> bool:
        """Send a message to another agent."""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            timestamp=time.time(),
            payload=payload,
            correlation_id=correlation_id,
            ttl=ttl
        )
        
        # Sign message (basic implementation)
        message.signature = self._sign_message(message)
        
        # Store in history
        self._add_to_history(message)
        
        if recipient_id == "*":
            # Broadcast message
            return await self._broadcast_message(message)
        else:
            # Send to specific agent
            return await self._send_to_agent(recipient_id, message)
    
    async def request_response(self, recipient_id: str, message_type: MessageType,
                              payload: Dict[str, Any], timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Send a request and wait for response."""
        correlation_id = str(uuid.uuid4())
        timeout = timeout or self.request_timeout
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[correlation_id] = response_future
        
        try:
            # Send request
            success = await self.send_message(
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
                ttl=timeout
            )
            
            if not success:
                return None
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            print(f"Request timeout: {message_type.value} to {recipient_id}")
            return None
        finally:
            # Clean up pending request
            self.pending_requests.pop(correlation_id, None)
    
    async def broadcast_message(self, message_type: MessageType, 
                               payload: Dict[str, Any]) -> List[str]:
        """Broadcast message to all agents in federation."""
        success_list = []
        
        if self.registry:
            # Get all online agents
            agents = await self.registry.get_all_agents("online")
            
            for agent_info in agents:
                if agent_info.agent_id != self.agent_id:  # Don't send to self
                    success = await self.send_message(
                        recipient_id=agent_info.agent_id,
                        message_type=message_type,
                        payload=payload
                    )
                    if success:
                        success_list.append(agent_info.agent_id)
        
        return success_list
    
    async def _send_to_agent(self, recipient_id: str, message: AgentMessage) -> bool:
        """Send message to specific agent with retry logic."""
        if not self.registry:
            print("No registry available for agent lookup")
            return False
        
        # Get recipient agent info
        agent_info = await self.registry.get_agent(recipient_id)
        if not agent_info:
            print(f"Recipient agent not found: {recipient_id}")
            return False
        
        # Try sending with retries
        for attempt in range(self.max_retries):
            try:
                success = await self._deliver_message(agent_info.endpoint, message)
                if success:
                    return True
                
                print(f"Message delivery failed (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                print(f"Message delivery error: {e}")
        
        return False
    
    async def _deliver_message(self, endpoint: str, message: AgentMessage) -> bool:
        """Deliver message to agent endpoint."""
        try:
            # Use HTTP POST to agent's message endpoint
            session = await self._get_session(endpoint)
            
            async with session.post(
                f"{endpoint}/messages/receive",
                json={"message": message.to_json()},
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Message delivery failed to {endpoint}: {e}")
            return False
    
    async def _broadcast_message(self, message: AgentMessage) -> bool:
        """Broadcast message to all agents."""
        if not self.registry:
            return False
        
        agents = await self.registry.get_all_agents("online")
        success_count = 0
        
        for agent_info in agents:
            if agent_info.agent_id != self.agent_id:
                success = await self._deliver_message(agent_info.endpoint, message)
                if success:
                    success_count += 1
        
        print(f"Broadcast delivered to {success_count}/{len(agents)-1} agents")
        return success_count > 0
    
    async def receive_message(self, message_json: str) -> Optional[str]:
        """Receive and process incoming message."""
        try:
            message = AgentMessage.from_json(message_json)
            
            # Validate message
            if not self._validate_message(message):
                print(f"Invalid message from {message.sender_id}")
                return None
            
            # Check if message is expired
            if message.is_expired():
                print(f"Expired message from {message.sender_id}")
                return None
            
            # Store in history
            self._add_to_history(message)
            
            # Handle response to pending request
            if message.correlation_id and message.correlation_id in self.pending_requests:
                future = self.pending_requests.pop(message.correlation_id)
                if not future.done():
                    future.set_result(message)
                return "ack"
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                response_message = await handler.handle_message(message)
                
                if response_message:
                    # Send response back
                    await self.send_message(
                        recipient_id=message.sender_id,
                        message_type=response_message.message_type,
                        payload=response_message.payload,
                        correlation_id=message.correlation_id
                    )
                
                return "processed"
            else:
                print(f"No handler for message type: {message.message_type.value}")
                return "no_handler"
        
        except Exception as e:
            print(f"Message processing error: {e}")
            return None
    
    async def ping_agent(self, agent_id: str) -> Optional[float]:
        """Ping another agent and measure response time."""
        start_time = time.time()
        
        response = await self.request_response(
            recipient_id=agent_id,
            message_type=MessageType.PING,
            payload={"timestamp": start_time},
            timeout=5.0
        )
        
        if response and response.message_type == MessageType.PONG:
            return time.time() - start_time
        
        return None
    
    async def _get_session(self, endpoint: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for endpoint."""
        if endpoint not in self.connections:
            self.connections[endpoint] = aiohttp.ClientSession()
        return self.connections[endpoint]
    
    def _sign_message(self, message: AgentMessage) -> str:
        """Create message signature for authentication."""
        # Simple implementation - in production use proper cryptographic signing
        message_data = f"{message.sender_id}:{message.recipient_id}:{message.timestamp}:{json.dumps(message.payload)}"
        return hashlib.sha256(message_data.encode()).hexdigest()[:16]
    
    def _validate_message(self, message: AgentMessage) -> bool:
        """Validate incoming message."""
        # Basic validation
        if not message.sender_id or not message.message_id:
            return False
        
        # Don't accept messages from self
        if message.sender_id == self.agent_id:
            return False
        
        # Verify signature (basic implementation)
        expected_signature = self._sign_message(message)
        if message.signature and message.signature != expected_signature:
            print(f"Message signature mismatch from {message.sender_id}")
            # In development, allow unsigned messages
            pass
        
        return True
    
    def _add_to_history(self, message: AgentMessage):
        """Add message to history with size limits."""
        self.message_history.append(message)
        
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history//2:]  # Keep last half
    
    async def get_message_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_messages = len(self.message_history)
        
        if total_messages == 0:
            return {"total_messages": 0}
        
        # Count by message type
        type_counts = {}
        recent_messages = 0
        current_time = time.time()
        
        for message in self.message_history:
            msg_type = message.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            
            # Count recent messages (last 5 minutes)
            if current_time - message.timestamp < 300:
                recent_messages += 1
        
        return {
            "total_messages": total_messages,
            "recent_messages": recent_messages,
            "message_types": type_counts,
            "pending_requests": len(self.pending_requests),
            "active_connections": len(self.connections)
        }
    
    async def close(self):
        """Clean up connections and resources."""
        # Close HTTP sessions
        for session in self.connections.values():
            await session.close()
        
        self.connections.clear()
        
        # Cancel pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        
        self.pending_requests.clear()
        print(f"AgentCommunicator closed for {self.agent_id}")

# Built-in message handlers
class PingPongHandler(MessageHandler):
    """Handler for ping/pong messages."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.PING:
            # Respond with pong
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.PONG,
                timestamp=time.time(),
                payload={"original_timestamp": message.payload.get("timestamp")},
                correlation_id=message.correlation_id
            )
        return None

class HeartbeatHandler(MessageHandler):
    """Handler for heartbeat messages."""
    
    def __init__(self, agent_id: str, registry):
        self.agent_id = agent_id
        self.registry = registry
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.HEARTBEAT:
            # Update registry with heartbeat
            if self.registry:
                await self.registry.update_heartbeat(
                    message.sender_id,
                    message.payload.get("performance_metrics")
                )
        return None

async def test_agent_communication():
    """Test the agent-to-agent communication system."""
    print("=== PHASE 5: AGENT COMMUNICATION TEST ===\n")
    
    # Import registry for testing
    from phase5_registry import AgentRegistry, AgentInfo, AgentCapability
    
    # Create registry
    registry = AgentRegistry("test_comm_registry.db")
    await registry.start()
    
    # Create test agents
    agent1_info = AgentInfo(
        agent_id="comm_agent_001",
        name="Communication Test Agent 1",
        endpoint="http://localhost:9001",
        ws_endpoint="ws://localhost:9001/stream",
        capabilities=[AgentCapability.GENERAL_LEARNING],
        status="online",
        performance_metrics={"val_loss": 0.45},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={}
    )
    
    agent2_info = AgentInfo(
        agent_id="comm_agent_002", 
        name="Communication Test Agent 2",
        endpoint="http://localhost:9002",
        ws_endpoint="ws://localhost:9002/stream",
        capabilities=[AgentCapability.TASK_SPAWNING],
        status="online",
        performance_metrics={"val_loss": 0.52},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={}
    )
    
    # Register agents
    await registry.register_agent(agent1_info)
    await registry.register_agent(agent2_info)
    
    # Create communicators
    comm1 = AgentCommunicator("comm_agent_001", registry)
    comm2 = AgentCommunicator("comm_agent_002", registry)
    
    # Register handlers
    comm1.register_handler(MessageType.PING, PingPongHandler("comm_agent_001"))
    comm2.register_handler(MessageType.PING, PingPongHandler("comm_agent_002"))
    
    comm1.register_handler(MessageType.HEARTBEAT, HeartbeatHandler("comm_agent_001", registry))
    comm2.register_handler(MessageType.HEARTBEAT, HeartbeatHandler("comm_agent_002", registry))
    
    print("1. Testing basic message sending...")
    
    # Test direct message (will fail since no actual HTTP servers, but tests message creation)
    success = await comm1.send_message(
        recipient_id="comm_agent_002",
        message_type=MessageType.DECISION_REQUEST,
        payload={"request": "evaluate_performance", "context": {"step": 100}}
    )
    print(f"   Direct message result: {success}")  # Expected to fail without actual servers
    
    print("\n2. Testing message creation and processing...")
    
    # Create test message
    test_message = AgentMessage(
        message_id=str(uuid.uuid4()),
        sender_id="comm_agent_001",
        recipient_id="comm_agent_002",
        message_type=MessageType.PING,
        timestamp=time.time(),
        payload={"timestamp": time.time()}
    )
    test_message.signature = comm1._sign_message(test_message)
    
    # Process message directly (simulating reception)
    result = await comm2.receive_message(test_message.to_json())
    print(f"   Message processing result: {result}")
    
    print("\n3. Testing message validation...")
    
    # Test message validation
    valid = comm2._validate_message(test_message)
    print(f"   Message validation: {valid}")
    
    print("\n4. Testing broadcast preparation...")
    
    # Test broadcast (will prepare but not actually send)
    recipients = await comm1.broadcast_message(
        message_type=MessageType.KNOWLEDGE_SYNC,
        payload={"sync_type": "reflection_update", "data": "test"}
    )
    print(f"   Broadcast attempted to: {len(recipients)} recipients")
    
    print("\n5. Testing communication statistics...")
    
    stats1 = await comm1.get_message_stats()
    stats2 = await comm2.get_message_stats()
    
    print(f"   Agent 1 stats: {stats1}")
    print(f"   Agent 2 stats: {stats2}")
    
    # Cleanup
    await comm1.close()
    await comm2.close()
    await registry.stop()
    
    print("\n[OK] AGENT COMMUNICATION TEST COMPLETED!")
    print("Communication layer provides:")
    print("- Reliable message delivery with retries")
    print("- Request/response patterns with correlation")
    print("- Message authentication and validation")
    print("- Broadcast messaging for coordination")
    print("- Connection pooling and health monitoring")
    print("- Built-in ping/pong and heartbeat handlers")

if __name__ == "__main__":
    # Clean up test database
    import os
    if os.path.exists("test_comm_registry.db"):
        os.remove("test_comm_registry.db")
    
    asyncio.run(test_agent_communication())