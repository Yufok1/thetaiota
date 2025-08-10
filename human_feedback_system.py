#!/usr/bin/env python3
"""
Phase 3: Human Feedback Integration - System for incorporating human feedback
into the agent's learning and decision-making process.
"""

import sqlite3
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from memory_db import MemoryDB

class FeedbackType(Enum):
    """Types of human feedback."""
    DECISION_APPROVAL = "decision_approval"    # Approve/disapprove of a decision
    PERFORMANCE_RATING = "performance_rating"  # Rate overall performance 
    BEHAVIOR_CORRECTION = "behavior_correction"  # Suggest behavior changes
    GOAL_ALIGNMENT = "goal_alignment"          # Feedback on goal alignment
    GENERAL_GUIDANCE = "general_guidance"      # General advice or direction

class FeedbackSentiment(Enum):
    """Sentiment of the feedback."""
    POSITIVE = 1
    NEUTRAL = 0  
    NEGATIVE = -1

@dataclass
class HumanFeedback:
    """Structured human feedback entry."""
    id: str
    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    target_step: Optional[int]  # Step the feedback refers to (if applicable)
    target_decision_id: Optional[str]  # Specific decision ID (if applicable)
    content: str  # Human feedback text
    rating: Optional[float]  # Numerical rating (1-5 scale if applicable)
    suggested_action: Optional[str]  # Suggested alternative action
    timestamp: float
    processed: bool = False  # Whether feedback has been integrated
    
    def to_reward_signal(self) -> float:
        """Convert feedback to a numerical reward signal for learning."""
        base_reward = float(self.sentiment.value)
        
        # Scale by rating if available
        if self.rating is not None:
            # Convert 1-5 rating to -1 to 1 scale
            rating_normalized = (self.rating - 3.0) / 2.0
            base_reward = rating_normalized
        
        # Scale by feedback importance
        importance_weights = {
            FeedbackType.DECISION_APPROVAL: 1.0,
            FeedbackType.PERFORMANCE_RATING: 0.8,
            FeedbackType.BEHAVIOR_CORRECTION: 1.2,
            FeedbackType.GOAL_ALIGNMENT: 1.5,
            FeedbackType.GENERAL_GUIDANCE: 0.6
        }
        
        return base_reward * importance_weights.get(self.feedback_type, 1.0)

class HumanFeedbackSystem:
    """
    Phase 3: System for collecting and integrating human feedback.
    Provides hooks for human oversight and guidance of agent behavior.
    """
    
    def __init__(self, memory_db: MemoryDB):
        self.memory = memory_db
        self._init_feedback_schema()
        
        # Feedback processing state
        self.pending_feedback_requests = []
        self.feedback_integration_mode = True
        
        # Feedback learning parameters
        self.feedback_learning_rate = 0.1
        self.min_feedback_confidence = 0.3
        
    def _init_feedback_schema(self):
        """Initialize human feedback database schema."""
        cursor = self.memory.conn.cursor()
        
        # Human feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_feedback (
                id              TEXT PRIMARY KEY,
                feedback_type   TEXT NOT NULL,
                sentiment       INTEGER NOT NULL,
                target_step     INTEGER,
                target_decision_id TEXT,
                content         TEXT NOT NULL,
                rating          REAL,
                suggested_action TEXT,
                timestamp       REAL NOT NULL,
                processed       BOOLEAN DEFAULT FALSE,
                reward_signal   REAL
            );
        """)
        
        # Feedback requests table (for async feedback collection)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_requests (
                id              TEXT PRIMARY KEY,
                request_type    TEXT NOT NULL,
                target_step     INTEGER,
                target_decision_id TEXT,
                context         TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                fulfilled       BOOLEAN DEFAULT FALSE
            );
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_target_step ON human_feedback(target_step);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_processed ON human_feedback(processed);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_fulfilled ON feedback_requests(fulfilled);")
        
        self.memory.conn.commit()
        
        # Log schema initialization
        self.memory.log_meta_event(
            event_type="human_feedback_system_init",
            info={"schema_tables": ["human_feedback", "feedback_requests"]}
        )
    
    def request_feedback(self, request_type: str, context: Dict[str, Any], 
                        target_step: Optional[int] = None, 
                        target_decision_id: Optional[str] = None) -> str:
        """
        Request human feedback for a specific situation.
        Returns request ID for tracking.
        """
        request_id = str(uuid.uuid4())
        
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            INSERT INTO feedback_requests 
            (id, request_type, target_step, target_decision_id, context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request_id, request_type, target_step, target_decision_id,
            json.dumps(context), time.time()
        ))
        
        self.memory.conn.commit()
        
        self.pending_feedback_requests.append({
            'id': request_id,
            'type': request_type,
            'context': context,
            'target_step': target_step,
            'target_decision_id': target_decision_id
        })
        
        return request_id
    
    def submit_feedback(self, feedback_type: FeedbackType, sentiment: FeedbackSentiment,
                       content: str, rating: Optional[float] = None,
                       suggested_action: Optional[str] = None,
                       target_step: Optional[int] = None,
                       target_decision_id: Optional[str] = None) -> str:
        """
        Submit human feedback to the system.
        Returns feedback ID.
        """
        feedback = HumanFeedback(
            id=str(uuid.uuid4()),
            feedback_type=feedback_type,
            sentiment=sentiment,
            target_step=target_step,
            target_decision_id=target_decision_id,
            content=content,
            rating=rating,
            suggested_action=suggested_action,
            timestamp=time.time()
        )
        
        # Calculate reward signal
        reward_signal = feedback.to_reward_signal()
        
        # Store in database
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            INSERT INTO human_feedback 
            (id, feedback_type, sentiment, target_step, target_decision_id, 
             content, rating, suggested_action, timestamp, reward_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.id, feedback.feedback_type.value, feedback.sentiment.value,
            feedback.target_step, feedback.target_decision_id, feedback.content,
            feedback.rating, feedback.suggested_action, feedback.timestamp, reward_signal
        ))
        
        self.memory.conn.commit()
        
        # Log feedback submission
        self.memory.log_meta_event(
            event_type="human_feedback_submitted",
            info={
                "feedback_id": feedback.id,
                "type": feedback.feedback_type.value,
                "sentiment": feedback.sentiment.value,
                "reward_signal": reward_signal,
                "has_rating": rating is not None,
                "has_suggestion": suggested_action is not None
            }
        )
        
        print(f"Human feedback submitted: {feedback.feedback_type.value} ({sentiment.name}) - Reward: {reward_signal:.3f}")
        
        return feedback.id
    
    def get_pending_feedback_requests(self) -> List[Dict[str, Any]]:
        """Get all pending feedback requests."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM feedback_requests 
            WHERE fulfilled = FALSE 
            ORDER BY timestamp DESC
        """)
        
        requests = []
        for row in cursor.fetchall():
            request_data = dict(row)
            request_data['context'] = json.loads(request_data['context'])
            requests.append(request_data)
        
        return requests
    
    def get_feedback_for_step(self, step: int) -> List[HumanFeedback]:
        """Get all human feedback for a specific training step."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM human_feedback 
            WHERE target_step = ? 
            ORDER BY timestamp DESC
        """, (step,))
        
        feedback_list = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            feedback = HumanFeedback(
                id=row_dict['id'],
                feedback_type=FeedbackType(row_dict['feedback_type']),
                sentiment=FeedbackSentiment(row_dict['sentiment']),
                target_step=row_dict['target_step'],
                target_decision_id=row_dict['target_decision_id'],
                content=row_dict['content'],
                rating=row_dict['rating'],
                suggested_action=row_dict['suggested_action'],
                timestamp=row_dict['timestamp'],
                processed=bool(row_dict['processed'])
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def get_unprocessed_feedback(self) -> List[HumanFeedback]:
        """Get all unprocessed human feedback."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM human_feedback 
            WHERE processed = FALSE 
            ORDER BY timestamp ASC
        """)
        
        feedback_list = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            feedback = HumanFeedback(
                id=row_dict['id'],
                feedback_type=FeedbackType(row_dict['feedback_type']),
                sentiment=FeedbackSentiment(row_dict['sentiment']),
                target_step=row_dict['target_step'],
                target_decision_id=row_dict['target_decision_id'],
                content=row_dict['content'],
                rating=row_dict['rating'],
                suggested_action=row_dict['suggested_action'],
                timestamp=row_dict['timestamp'],
                processed=bool(row_dict['processed'])
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def process_feedback_for_learning(self, meta_controller) -> Dict[str, Any]:
        """
        Process unprocessed feedback and integrate into meta-controller learning.
        Returns summary of processing results.
        """
        unprocessed = self.get_unprocessed_feedback()
        
        if not unprocessed:
            return {"processed_count": 0, "reward_adjustments": [], "suggested_actions": [], "avg_sentiment": 0.0}
        
        processing_results = {
            "processed_count": len(unprocessed),
            "reward_adjustments": [],
            "suggested_actions": [],
            "avg_sentiment": 0.0
        }
        
        total_sentiment = 0.0
        
        for feedback in unprocessed:
            # Apply feedback as additional reward signal to meta-controller
            reward_adjustment = feedback.to_reward_signal()
            
            # If feedback targets a specific step, apply retroactive learning
            if feedback.target_step is not None and hasattr(meta_controller, 'apply_retroactive_reward'):
                meta_controller.apply_retroactive_reward(feedback.target_step, reward_adjustment)
            
            # Store suggested actions for policy adjustment
            if feedback.suggested_action:
                processing_results["suggested_actions"].append({
                    "feedback_id": feedback.id,
                    "suggestion": feedback.suggested_action,
                    "context_step": feedback.target_step
                })
            
            processing_results["reward_adjustments"].append({
                "feedback_id": feedback.id,
                "reward": reward_adjustment,
                "type": feedback.feedback_type.value
            })
            
            total_sentiment += feedback.sentiment.value
            
            # Mark feedback as processed
            self._mark_feedback_processed(feedback.id)
        
        processing_results["avg_sentiment"] = total_sentiment / len(unprocessed)
        
        # Log processing results
        self.memory.log_meta_event(
            event_type="feedback_batch_processed",
            info=processing_results
        )
        
        return processing_results
    
    def _mark_feedback_processed(self, feedback_id: str):
        """Mark a feedback entry as processed."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            UPDATE human_feedback 
            SET processed = TRUE 
            WHERE id = ?
        """, (feedback_id,))
        self.memory.conn.commit()
    
    def create_feedback_interface(self) -> 'FeedbackInterface':
        """Create an interactive feedback interface."""
        return FeedbackInterface(self)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary statistics about human feedback."""
        cursor = self.memory.conn.cursor()
        
        # Total feedback count by type
        cursor.execute("""
            SELECT feedback_type, COUNT(*) as count, AVG(reward_signal) as avg_reward
            FROM human_feedback 
            GROUP BY feedback_type
        """)
        
        feedback_by_type = {}
        for row in cursor.fetchall():
            feedback_by_type[row[0]] = {
                "count": row[1],
                "avg_reward": row[2]
            }
        
        # Recent feedback sentiment trend
        cursor.execute("""
            SELECT sentiment, COUNT(*) as count 
            FROM human_feedback 
            WHERE timestamp > ? 
            GROUP BY sentiment
        """, (time.time() - 86400,))  # Last 24 hours
        
        recent_sentiment = {}
        for row in cursor.fetchall():
            sentiment_name = {1: "positive", 0: "neutral", -1: "negative"}[row[0]]
            recent_sentiment[sentiment_name] = row[1]
        
        return {
            "total_feedback": sum(data["count"] for data in feedback_by_type.values()),
            "feedback_by_type": feedback_by_type,
            "recent_sentiment": recent_sentiment,
            "pending_requests": len(self.get_pending_feedback_requests())
        }

class FeedbackInterface:
    """
    Interactive interface for humans to provide feedback to the agent.
    Can be used in CLI or extended for web interfaces.
    """
    
    def __init__(self, feedback_system: HumanFeedbackSystem):
        self.feedback_system = feedback_system
    
    def show_pending_requests(self):
        """Display pending feedback requests."""
        requests = self.feedback_system.get_pending_feedback_requests()
        
        if not requests:
            print("No pending feedback requests.")
            return
        
        print(f"\n=== {len(requests)} PENDING FEEDBACK REQUESTS ===")
        
        for i, request in enumerate(requests, 1):
            print(f"\n{i}. {request['request_type'].upper()}")
            print(f"   Step: {request['target_step']}")
            print(f"   Context: {request['context']}")
            
            # Show decision details if available
            if request['target_decision_id']:
                print(f"   Decision ID: {request['target_decision_id']}")
    
    def collect_decision_feedback(self, step: int, decision_details: Dict[str, Any]) -> Optional[str]:
        """
        Collect human feedback on a specific decision.
        Returns feedback ID if provided, None if skipped.
        """
        print(f"\n=== DECISION FEEDBACK REQUEST (Step {step}) ===")
        print(f"Action: {decision_details.get('action', 'Unknown')}")
        print(f"Reasoning: {decision_details.get('reasoning', 'No reasoning provided')}")
        print(f"Confidence: {decision_details.get('confidence', 0.0):.3f}")
        
        # Simple CLI feedback collection
        print("\nOptions:")
        print("1. Approve decision (+)")  
        print("2. Neutral/No opinion (0)")
        print("3. Disapprove decision (-)")
        print("4. Skip feedback")
        
        try:
            choice = input("Your choice (1-4): ").strip()
            
            if choice == "4":
                return None
            
            sentiment_map = {
                "1": FeedbackSentiment.POSITIVE,
                "2": FeedbackSentiment.NEUTRAL, 
                "3": FeedbackSentiment.NEGATIVE
            }
            
            if choice not in sentiment_map:
                print("Invalid choice, skipping feedback.")
                return None
            
            sentiment = sentiment_map[choice]
            
            # Get optional comment
            comment = input("Additional comment (optional): ").strip()
            if not comment:
                comment = f"Decision {sentiment.name.lower()} feedback"
            
            # Get optional rating
            rating_input = input("Rating 1-5 (optional): ").strip()
            rating = None
            if rating_input and rating_input.isdigit():
                rating = float(rating_input)
                rating = max(1.0, min(5.0, rating))  # Clamp to 1-5
            
            # Submit feedback
            feedback_id = self.feedback_system.submit_feedback(
                feedback_type=FeedbackType.DECISION_APPROVAL,
                sentiment=sentiment,
                content=comment,
                rating=rating,
                target_step=step
            )
            
            return feedback_id
            
        except KeyboardInterrupt:
            print("\nFeedback cancelled.")
            return None
        except Exception as e:
            print(f"Error collecting feedback: {e}")
            return None
    
    def collect_performance_feedback(self, performance_summary: Dict[str, Any]) -> Optional[str]:
        """Collect feedback on overall performance."""
        print(f"\n=== PERFORMANCE FEEDBACK REQUEST ===")
        print(f"Recent performance summary:")
        for key, value in performance_summary.items():
            print(f"  {key}: {value}")
        
        try:
            rating = input("Rate performance 1-5 (1=poor, 5=excellent): ").strip()
            if not rating.isdigit():
                return None
            
            rating = float(rating)
            rating = max(1.0, min(5.0, rating))
            
            comment = input("Performance feedback (optional): ").strip()
            if not comment:
                comment = f"Performance rated {rating}/5"
            
            # Convert rating to sentiment
            if rating >= 4:
                sentiment = FeedbackSentiment.POSITIVE
            elif rating <= 2:
                sentiment = FeedbackSentiment.NEGATIVE
            else:
                sentiment = FeedbackSentiment.NEUTRAL
            
            feedback_id = self.feedback_system.submit_feedback(
                feedback_type=FeedbackType.PERFORMANCE_RATING,
                sentiment=sentiment,
                content=comment,
                rating=rating
            )
            
            return feedback_id
            
        except KeyboardInterrupt:
            print("\nFeedback cancelled.")
            return None
        except Exception as e:
            print(f"Error collecting performance feedback: {e}")
            return None

def test_human_feedback_system():
    """Test the human feedback system."""
    print("Testing Human Feedback System...")
    
    # Create test database connection
    from init_database import create_database
    create_database("test_feedback.db")
    
    with MemoryDB("test_feedback.db") as memory:
        feedback_system = HumanFeedbackSystem(memory)
        
        print("\n1. Testing feedback submission...")
        
        # Submit test feedback
        feedback_id1 = feedback_system.submit_feedback(
            feedback_type=FeedbackType.DECISION_APPROVAL,
            sentiment=FeedbackSentiment.POSITIVE,
            content="Great decision, improved performance significantly",
            rating=4.5,
            target_step=25
        )
        
        feedback_id2 = feedback_system.submit_feedback(
            feedback_type=FeedbackType.BEHAVIOR_CORRECTION,
            sentiment=FeedbackSentiment.NEGATIVE,
            content="Should have spawned more tasks to address the plateau",
            suggested_action="SPAWN_DATA_COLLECTION",
            target_step=30
        )
        
        print(f"   Submitted feedback: {feedback_id1}, {feedback_id2}")
        
        print("\n2. Testing feedback request...")
        request_id = feedback_system.request_feedback(
            request_type="decision_review",
            context={"action": "FINE_TUNE_NOW", "confidence": 0.8},
            target_step=35
        )
        print(f"   Created feedback request: {request_id}")
        
        print("\n3. Testing feedback retrieval...")
        step_feedback = feedback_system.get_feedback_for_step(25)
        print(f"   Found {len(step_feedback)} feedback entries for step 25")
        
        unprocessed = feedback_system.get_unprocessed_feedback()
        print(f"   Found {len(unprocessed)} unprocessed feedback entries")
        
        print("\n4. Testing feedback summary...")
        summary = feedback_system.get_feedback_summary()
        print(f"   Summary: {summary}")
        
        print("\n5. Testing feedback interface...")
        interface = feedback_system.create_feedback_interface()
        interface.show_pending_requests()
    
    print("\nHuman Feedback System test completed!")

if __name__ == "__main__":
    test_human_feedback_system()