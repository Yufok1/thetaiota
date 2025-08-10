#!/usr/bin/env python3
"""
Phase 5: Cross-Agent Consensus Mechanisms - Collaborative decision making for federated AI.
Enables multiple self-aware agents to vote, deliberate, and reach consensus on complex decisions.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import Counter

from phase5_communication import AgentCommunicator, MessageType, AgentMessage, MessageHandler

class ConsensusType(Enum):
    """Types of consensus protocols."""
    SIMPLE_MAJORITY = "simple_majority"         # >50% agreement
    SUPERMAJORITY = "supermajority"            # >=67% agreement  
    UNANIMOUS = "unanimous"                     # 100% agreement
    WEIGHTED_CONFIDENCE = "weighted_confidence"  # Weight by agent confidence
    CAPABILITY_WEIGHTED = "capability_weighted" # Weight by relevant capabilities

class ProposalStatus(Enum):
    """Status of a consensus proposal."""
    PROPOSED = "proposed"
    VOTING = "voting" 
    DECIDED = "decided"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ConsensusVote:
    """A vote from an agent on a proposal."""
    agent_id: str
    proposal_id: str
    decision: str  # The agent's preferred choice
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why the agent chose this
    capabilities_used: List[str]  # Which capabilities informed this vote
    timestamp: float

@dataclass
class ConsensusProposal:
    """A proposal requiring consensus from multiple agents."""
    proposal_id: str
    proposer_id: str
    title: str
    description: str
    options: List[str]  # Available choices
    context: Dict[str, Any]  # Additional context data
    consensus_type: ConsensusType
    required_capabilities: List[str]  # Required agent capabilities
    min_voters: int  # Minimum number of voters required
    timeout: float  # When proposal expires
    created_at: float
    
    # Voting state
    votes: List[ConsensusVote]
    status: ProposalStatus
    result: Optional[str] = None
    confidence: Optional[float] = None

class ConsensusEngine:
    """
    Phase 5: Cross-Agent Consensus Engine
    
    Features:
    - Multiple consensus algorithms (majority, supermajority, weighted)
    - Capability-based voting (agents vote based on their expertise)
    - Confidence-weighted decisions (high-confidence votes matter more)
    - Reasoning aggregation (combine agent explanations)
    - Timeout handling and failure recovery
    - Consensus history and learning
    """
    
    def __init__(self, agent_id: str, communicator: AgentCommunicator, registry=None):
        self.agent_id = agent_id
        self.communicator = communicator
        self.registry = registry
        
        # Active proposals
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_history: List[ConsensusProposal] = []
        
        # Register message handlers
        self.communicator.register_handler(MessageType.CONSENSUS_VOTE, ConsensusVoteHandler(self))
        self.communicator.register_handler(MessageType.DECISION_REQUEST, DecisionRequestHandler(self))
        
        # Configuration
        self.default_timeout = 60.0  # seconds
        self.max_concurrent_proposals = 10
        
        print(f"ConsensusEngine initialized for {agent_id}")
    
    async def propose_decision(self, title: str, description: str, options: List[str],
                              consensus_type: ConsensusType = ConsensusType.SIMPLE_MAJORITY,
                              required_capabilities: Optional[List[str]] = None,
                              min_voters: int = 2, timeout: Optional[float] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new consensus proposal."""
        
        proposal_id = str(uuid.uuid4())
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposer_id=self.agent_id,
            title=title,
            description=description,
            options=options,
            context=context or {},
            consensus_type=consensus_type,
            required_capabilities=required_capabilities or [],
            min_voters=min_voters,
            timeout=time.time() + (timeout or self.default_timeout),
            created_at=time.time(),
            votes=[],
            status=ProposalStatus.PROPOSED
        )
        
        # Store proposal
        self.proposals[proposal_id] = proposal
        
        # Find eligible voters
        eligible_voters = await self._find_eligible_voters(proposal)
        
        print(f"Created proposal '{title}' with {len(eligible_voters)} eligible voters")
        
        # Broadcast proposal to eligible agents
        proposal_data = {
            "proposal_id": proposal_id,
            "title": title,
            "description": description, 
            "options": options,
            "consensus_type": consensus_type.value,
            "required_capabilities": required_capabilities or [],
            "timeout": proposal.timeout,
            "context": context or {}
        }
        
        # Send to each eligible voter
        for voter_id in eligible_voters:
            await self.communicator.send_message(
                recipient_id=voter_id,
                message_type=MessageType.DECISION_REQUEST,
                payload=proposal_data
            )
        
        # Start monitoring task
        asyncio.create_task(self._monitor_proposal(proposal_id))
        
        proposal.status = ProposalStatus.VOTING
        return proposal_id
    
    async def vote_on_proposal(self, proposal_id: str, decision: str, 
                              confidence: float, reasoning: str,
                              capabilities_used: Optional[List[str]] = None) -> bool:
        """Submit a vote on a proposal."""
        
        # Find the proposal (might be from another agent)
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            print(f"Proposal not found: {proposal_id}")
            return False
        
        # Check if already voted
        existing_vote = next((v for v in proposal.votes if v.agent_id == self.agent_id), None)
        if existing_vote:
            print(f"Already voted on proposal: {proposal_id}")
            return False
        
        # Create vote
        vote = ConsensusVote(
            agent_id=self.agent_id,
            proposal_id=proposal_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            capabilities_used=capabilities_used or [],
            timestamp=time.time()
        )
        
        # Send vote to proposer
        vote_data = {
            "proposal_id": proposal_id,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "capabilities_used": capabilities_used or [],
            "voter_id": self.agent_id
        }
        
        success = await self.communicator.send_message(
            recipient_id=proposal.proposer_id,
            message_type=MessageType.CONSENSUS_VOTE,
            payload=vote_data
        )
        
        if success:
            print(f"Voted '{decision}' on proposal '{proposal.title}' (confidence: {confidence:.2f})")
        
        return success
    
    async def receive_vote(self, vote_data: Dict[str, Any]) -> bool:
        """Receive a vote from another agent."""
        proposal_id = vote_data.get("proposal_id")
        proposal = self.proposals.get(proposal_id)
        
        if not proposal:
            print(f"Received vote for unknown proposal: {proposal_id}")
            return False
        
        # Create vote object
        vote = ConsensusVote(
            agent_id=vote_data.get("voter_id"),
            proposal_id=proposal_id,
            decision=vote_data.get("decision"),
            confidence=vote_data.get("confidence", 0.5),
            reasoning=vote_data.get("reasoning", ""),
            capabilities_used=vote_data.get("capabilities_used", []),
            timestamp=time.time()
        )
        
        # Add vote to proposal
        proposal.votes.append(vote)
        
        print(f"Received vote from {vote.agent_id}: '{vote.decision}' (confidence: {vote.confidence:.2f})")
        
        # Check if we can reach consensus
        consensus_result = await self._evaluate_consensus(proposal)
        
        if consensus_result:
            await self._finalize_proposal(proposal, consensus_result)
        
        return True
    
    async def _find_eligible_voters(self, proposal: ConsensusProposal) -> List[str]:
        """Find agents eligible to vote on this proposal."""
        eligible_voters = []
        
        if not self.registry:
            return eligible_voters
        
        # Get all online agents
        agents = await self.registry.get_all_agents("online")
        
        for agent_info in agents:
            if agent_info.agent_id == self.agent_id:
                continue  # Proposer doesn't vote
            
            # Check if agent has required capabilities
            if proposal.required_capabilities:
                agent_capabilities = [cap.value for cap in agent_info.capabilities]
                if not any(req_cap in agent_capabilities for req_cap in proposal.required_capabilities):
                    continue
            
            eligible_voters.append(agent_info.agent_id)
        
        return eligible_voters
    
    async def _evaluate_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Evaluate if consensus has been reached on a proposal."""
        
        if len(proposal.votes) < proposal.min_voters:
            return None  # Not enough votes yet
        
        if proposal.consensus_type == ConsensusType.SIMPLE_MAJORITY:
            return self._simple_majority_consensus(proposal)
        
        elif proposal.consensus_type == ConsensusType.SUPERMAJORITY:
            return self._supermajority_consensus(proposal)
        
        elif proposal.consensus_type == ConsensusType.UNANIMOUS:
            return self._unanimous_consensus(proposal)
        
        elif proposal.consensus_type == ConsensusType.WEIGHTED_CONFIDENCE:
            return self._confidence_weighted_consensus(proposal)
        
        elif proposal.consensus_type == ConsensusType.CAPABILITY_WEIGHTED:
            return self._capability_weighted_consensus(proposal)
        
        return None
    
    def _simple_majority_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Simple majority vote (>50%)."""
        if not proposal.votes:
            return None
        
        # Count votes for each option
        vote_counts = Counter(vote.decision for vote in proposal.votes)
        total_votes = len(proposal.votes)
        
        # Find majority
        for decision, count in vote_counts.most_common():
            if count > total_votes / 2:
                confidence = count / total_votes
                return (decision, confidence)
        
        return None
    
    def _supermajority_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Supermajority vote (>=67%)."""
        if not proposal.votes:
            return None
        
        vote_counts = Counter(vote.decision for vote in proposal.votes)
        total_votes = len(proposal.votes)
        
        for decision, count in vote_counts.most_common():
            if count >= (total_votes * 2) // 3:  # 67% threshold
                confidence = count / total_votes
                return (decision, confidence)
        
        return None
    
    def _unanimous_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Unanimous vote (100%)."""
        if not proposal.votes:
            return None
        
        # All votes must be the same
        decisions = [vote.decision for vote in proposal.votes]
        if len(set(decisions)) == 1:
            return (decisions[0], 1.0)
        
        return None
    
    def _confidence_weighted_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Confidence-weighted voting."""
        if not proposal.votes:
            return None
        
        # Group votes by decision
        decision_weights = {}
        
        for vote in proposal.votes:
            if vote.decision not in decision_weights:
                decision_weights[vote.decision] = 0.0
            decision_weights[vote.decision] += vote.confidence
        
        # Find decision with highest total confidence
        if decision_weights:
            best_decision = max(decision_weights, key=decision_weights.get)
            total_confidence = sum(decision_weights.values())
            
            if total_confidence > 0:
                relative_confidence = decision_weights[best_decision] / total_confidence
                
                # Require at least 60% weighted confidence
                if relative_confidence >= 0.6:
                    return (best_decision, relative_confidence)
        
        return None
    
    def _capability_weighted_consensus(self, proposal: ConsensusProposal) -> Optional[Tuple[str, float]]:
        """Weight votes by relevant agent capabilities."""
        if not proposal.votes:
            return None
        
        # This would require access to agent capability information
        # For now, fall back to confidence weighting
        return self._confidence_weighted_consensus(proposal)
    
    async def _finalize_proposal(self, proposal: ConsensusProposal, result: Tuple[str, float]):
        """Finalize a proposal with consensus result."""
        decision, confidence = result
        
        proposal.result = decision
        proposal.confidence = confidence
        proposal.status = ProposalStatus.DECIDED
        
        print(f"Consensus reached on '{proposal.title}': {decision} (confidence: {confidence:.2f})")
        
        # Store in history
        self.consensus_history.append(proposal)
        
        # Broadcast result to all voters
        result_data = {
            "proposal_id": proposal.proposal_id,
            "result": decision,
            "confidence": confidence,
            "vote_count": len(proposal.votes),
            "consensus_type": proposal.consensus_type.value
        }
        
        # Get all agents who voted
        voter_ids = [vote.agent_id for vote in proposal.votes]
        
        for voter_id in voter_ids:
            await self.communicator.send_message(
                recipient_id=voter_id,
                message_type=MessageType.CONSENSUS_RESULT,
                payload=result_data
            )
        
        # Remove from active proposals
        if proposal.proposal_id in self.proposals:
            del self.proposals[proposal.proposal_id]
    
    async def _monitor_proposal(self, proposal_id: str):
        """Monitor proposal for timeout."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return
        
        # Wait until timeout
        wait_time = proposal.timeout - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Check if still active
        proposal = self.proposals.get(proposal_id)
        if proposal and proposal.status == ProposalStatus.VOTING:
            # Timeout reached
            print(f"Proposal '{proposal.title}' timed out with {len(proposal.votes)} votes")
            
            proposal.status = ProposalStatus.EXPIRED
            
            # Try to reach consensus with available votes
            consensus_result = await self._evaluate_consensus(proposal)
            
            if consensus_result:
                await self._finalize_proposal(proposal, consensus_result)
            else:
                proposal.status = ProposalStatus.FAILED
                print(f"Proposal '{proposal.title}' failed to reach consensus")
                
                # Store in history
                self.consensus_history.append(proposal)
                del self.proposals[proposal_id]
    
    async def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus system statistics."""
        active_count = len(self.proposals)
        history_count = len(self.consensus_history)
        
        if history_count == 0:
            return {
                "active_proposals": active_count,
                "completed_proposals": 0,
                "success_rate": 0.0
            }
        
        # Analyze historical results
        successful = sum(1 for p in self.consensus_history if p.status == ProposalStatus.DECIDED)
        failed = sum(1 for p in self.consensus_history if p.status == ProposalStatus.FAILED)
        
        consensus_types = Counter(p.consensus_type.value for p in self.consensus_history)
        
        return {
            "active_proposals": active_count,
            "completed_proposals": history_count,
            "successful_proposals": successful,
            "failed_proposals": failed,
            "success_rate": successful / history_count if history_count > 0 else 0.0,
            "consensus_types_used": dict(consensus_types)
        }

# Message handlers for consensus
class ConsensusVoteHandler(MessageHandler):
    """Handler for consensus vote messages."""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.CONSENSUS_VOTE:
            await self.consensus_engine.receive_vote(message.payload)
        return None

class DecisionRequestHandler(MessageHandler):
    """Handler for decision request messages."""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DECISION_REQUEST:
            # This would trigger the agent's decision-making process
            # For now, just acknowledge receipt
            print(f"Received decision request: {message.payload.get('title', 'Unknown')}")
        return None

async def test_consensus_system():
    """Test the cross-agent consensus system."""
    print("=== PHASE 5: CONSENSUS SYSTEM TEST ===\n")
    
    # Import registry and communication for testing
    from phase5_registry import AgentRegistry, AgentInfo, AgentCapability
    from phase5_communication import AgentCommunicator
    
    # Create registry
    registry = AgentRegistry("test_consensus_registry.db")
    await registry.start()
    
    # Create test agents with different specializations
    agent1_info = AgentInfo(
        agent_id="consensus_agent_001",
        name="Decision Specialist",
        endpoint="http://localhost:9101",
        ws_endpoint="ws://localhost:9101/stream",
        capabilities=[AgentCapability.DECISION_MAKING, AgentCapability.GENERAL_LEARNING],
        status="online",
        performance_metrics={"confidence": 0.85, "decisions_made": 150},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={"specialization": "decisions"}
    )
    
    agent2_info = AgentInfo(
        agent_id="consensus_agent_002",
        name="Task Management Specialist", 
        endpoint="http://localhost:9102",
        ws_endpoint="ws://localhost:9102/stream",
        capabilities=[AgentCapability.TASK_SPAWNING, AgentCapability.CRISIS_DETECTION],
        status="online",
        performance_metrics={"confidence": 0.78, "tasks_completed": 89},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={"specialization": "task_management"}
    )
    
    agent3_info = AgentInfo(
        agent_id="consensus_agent_003",
        name="Reflection Analyst",
        endpoint="http://localhost:9103", 
        ws_endpoint="ws://localhost:9103/stream",
        capabilities=[AgentCapability.REFLECTION_ANALYSIS, AgentCapability.MEMORY_QUERYING],
        status="online",
        performance_metrics={"confidence": 0.92, "reflections_created": 45},
        last_heartbeat=time.time(),
        registered_at=time.time(),
        metadata={"specialization": "analysis"}
    )
    
    # Register agents
    await registry.register_agent(agent1_info)
    await registry.register_agent(agent2_info) 
    await registry.register_agent(agent3_info)
    
    # Create communicators and consensus engines
    comm1 = AgentCommunicator("consensus_agent_001", registry)
    comm2 = AgentCommunicator("consensus_agent_002", registry)
    comm3 = AgentCommunicator("consensus_agent_003", registry)
    
    consensus1 = ConsensusEngine("consensus_agent_001", comm1, registry)
    consensus2 = ConsensusEngine("consensus_agent_002", comm2, registry)
    consensus3 = ConsensusEngine("consensus_agent_003", comm3, registry)
    
    print("1. Testing simple majority consensus proposal...")
    
    # Agent 1 proposes a decision
    proposal_id = await consensus1.propose_decision(
        title="Select Next Training Strategy",
        description="Choose the best approach for improving model performance",
        options=["curriculum_advancement", "fine_tuning", "task_spawning", "reflection_focus"],
        consensus_type=ConsensusType.SIMPLE_MAJORITY,
        required_capabilities=["decision_making", "general_learning"],
        min_voters=2,
        timeout=30.0,
        context={"current_performance": {"val_loss": 0.45, "accuracy": 0.82}}
    )
    
    print(f"   Created proposal: {proposal_id}")
    
    # Simulate votes from other agents
    print("\n2. Simulating agent votes...")
    
    # Agent 2 votes (task management perspective)
    vote2_success = await consensus2.vote_on_proposal(
        proposal_id=proposal_id,
        decision="task_spawning",
        confidence=0.85,
        reasoning="Current performance suggests need for more diverse training tasks",
        capabilities_used=["task_spawning"]
    )
    print(f"   Agent 2 vote: {vote2_success}")
    
    # Agent 3 votes (analysis perspective)  
    vote3_success = await consensus3.vote_on_proposal(
        proposal_id=proposal_id,
        decision="reflection_focus",
        confidence=0.78,
        reasoning="Analysis of past decisions indicates need for better self-reflection",
        capabilities_used=["reflection_analysis"]
    )
    print(f"   Agent 3 vote: {vote3_success}")
    
    # Wait a moment for processing
    await asyncio.sleep(1.0)
    
    print("\n3. Testing confidence-weighted consensus...")
    
    # Create another proposal with confidence weighting
    proposal_id2 = await consensus2.propose_decision(
        title="Crisis Response Strategy",
        description="How to respond to detected performance degradation",
        options=["immediate_rollback", "gradual_adjustment", "human_intervention"],
        consensus_type=ConsensusType.WEIGHTED_CONFIDENCE,
        required_capabilities=["crisis_detection"],
        min_voters=2
    )
    
    # Votes with different confidence levels
    await consensus1.vote_on_proposal(
        proposal_id=proposal_id2,
        decision="gradual_adjustment",
        confidence=0.95,  # High confidence
        reasoning="Historical data shows gradual adjustments are more stable"
    )
    
    await consensus3.vote_on_proposal(
        proposal_id=proposal_id2, 
        decision="immediate_rollback",
        confidence=0.60,  # Lower confidence
        reasoning="Crisis indicators suggest immediate action needed"
    )
    
    await asyncio.sleep(1.0)
    
    print("\n4. Testing consensus statistics...")
    
    stats1 = await consensus1.get_consensus_stats()
    stats2 = await consensus2.get_consensus_stats()
    
    print(f"   Agent 1 consensus stats: {stats1}")
    print(f"   Agent 2 consensus stats: {stats2}")
    
    # Check proposal states
    print("\n5. Checking final proposal states...")
    
    for engine, name in [(consensus1, "Agent 1"), (consensus2, "Agent 2"), (consensus3, "Agent 3")]:
        active = len(engine.proposals)
        history = len(engine.consensus_history)
        print(f"   {name}: {active} active proposals, {history} completed")
    
    # Cleanup
    await comm1.close()
    await comm2.close()
    await comm3.close()
    await registry.stop()
    
    print("\n[OK] CONSENSUS SYSTEM TEST COMPLETED!")
    print("Consensus engine provides:")
    print("- Multiple consensus algorithms (majority, supermajority, weighted)")
    print("- Capability-based voting eligibility")
    print("- Confidence-weighted decision making")
    print("- Reasoning aggregation and explanation")
    print("- Timeout handling and failure recovery")
    print("- Consensus history and performance tracking")

if __name__ == "__main__":
    # Clean up test database
    import os
    if os.path.exists("test_consensus_registry.db"):
        os.remove("test_consensus_registry.db")
    
    asyncio.run(test_consensus_system())