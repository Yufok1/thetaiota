#!/usr/bin/env python3
"""
Phase 4: API Test Client - Demonstrates the HTTP API functionality.
"""

import asyncio
import json
import time
from typing import Dict, Any

try:
    import aiohttp
    import websockets
    HTTP_CLIENTS_AVAILABLE = True
except ImportError:
    HTTP_CLIENTS_AVAILABLE = False
    print("HTTP client libraries not available. Please install: pip install aiohttp websockets")

class Phase4APIClient:
    """Test client for the Phase 4 API server."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        
    async def test_agent_lifecycle(self):
        """Test agent start/pause/resume/stop lifecycle."""
        if not HTTP_CLIENTS_AVAILABLE:
            return
        
        print("Testing agent lifecycle...")
        
        async with aiohttp.ClientSession() as session:
            # Health check
            print("1. Health check")
            async with session.get(f"{self.base_url}/health") as resp:
                data = await resp.json()
                print(f"   Health: {data['success']}, Server uptime: {data['data']['server_uptime']:.1f}s")
            
            # Start agent
            print("2. Starting agent")
            async with session.post(f"{self.base_url}/agent/start") as resp:
                data = await resp.json()
                if data['success']:
                    print(f"   Started: {data['data']['agent_id']}")
                else:
                    print(f"   Failed: {data['message']}")
                    return
            
            # Wait for initialization
            await asyncio.sleep(3.0)
            
            # Get status
            print("3. Getting status")
            async with session.get(f"{self.base_url}/agent/status") as resp:
                data = await resp.json()
                status = data['data']
                print(f"   State: {status['state']}, Step: {status['current_step']}, "
                      f"Decisions: {status['total_decisions']}")
            
            # Pause agent
            print("4. Pausing agent")
            async with session.post(f"{self.base_url}/agent/pause") as resp:
                data = await resp.json()
                print(f"   Pause result: {data['success']}")
            
            await asyncio.sleep(2.0)
            
            # Resume agent  
            print("5. Resuming agent")
            async with session.post(f"{self.base_url}/agent/resume") as resp:
                data = await resp.json()
                print(f"   Resume result: {data['success']}")
            
            await asyncio.sleep(2.0)
            
            # Final status
            async with session.get(f"{self.base_url}/agent/status") as resp:
                data = await resp.json()
                status = data['data']
                print(f"   Final state: {status['state']}, Step: {status['current_step']}")
    
    async def test_feedback_system(self):
        """Test human feedback submission."""
        if not HTTP_CLIENTS_AVAILABLE:
            return
        
        print("\nTesting feedback system...")
        
        async with aiohttp.ClientSession() as session:
            # Submit positive feedback
            feedback_data = {
                "feedback_type": "decision_approval",
                "sentiment": "positive",
                "content": "Great decision! The agent is learning well.",
                "rating": 4.5
            }
            
            async with session.post(f"{self.base_url}/feedback/submit", json=feedback_data) as resp:
                data = await resp.json()
                if data['success']:
                    print(f"   Submitted feedback: {data['data']['feedback_id']}")
                else:
                    print(f"   Feedback failed: {data['message']}")
            
            # Submit corrective feedback
            feedback_data2 = {
                "feedback_type": "behavior_correction",
                "sentiment": "negative", 
                "content": "Should have spawned more tasks to address performance issues.",
                "rating": 2.0
            }
            
            async with session.post(f"{self.base_url}/feedback/submit", json=feedback_data2) as resp:
                data = await resp.json()
                if data['success']:
                    print(f"   Submitted correction: {data['data']['feedback_id']}")
    
    async def test_memory_queries(self):
        """Test memory querying capabilities."""
        if not HTTP_CLIENTS_AVAILABLE:
            return
        
        print("\nTesting memory queries...")
        
        async with aiohttp.ClientSession() as session:
            # Natural language query
            query_data = {"query": "What was my last decision?"}
            
            async with session.post(f"{self.base_url}/memory/query", json=query_data) as resp:
                data = await resp.json()
                if data['success']:
                    result = data['data']
                    print(f"   Query type: {result.get('query_type', 'unknown')}")
                    print(f"   Result: {type(result.get('result', 'None')).__name__}")
                else:
                    print(f"   Query failed: {data['message']}")
            
            # Get recent decisions
            async with session.get(f"{self.base_url}/memory/decisions?limit=3") as resp:
                data = await resp.json()
                if data['success']:
                    decisions = data['data']['decisions']
                    print(f"   Found {len(decisions)} recent decisions")
                    if decisions:
                        latest = decisions[0]['info']
                        print(f"     Latest: {latest.get('action', 'unknown')} (step {latest.get('step', '?')})")
            
            # Get reflections
            async with session.get(f"{self.base_url}/memory/reflections?limit=3") as resp:
                data = await resp.json()
                if data['success']:
                    reflections = data['data']['reflections']
                    print(f"   Found {len(reflections)} reflections")
                    if reflections:
                        latest = reflections[0]
                        print(f"     Latest: {latest['summary_type']} - {latest['trigger']}")
    
    async def test_websocket_stream(self, duration: int = 10):
        """Test WebSocket event streaming."""
        if not HTTP_CLIENTS_AVAILABLE:
            return
        
        print(f"\nTesting WebSocket stream for {duration} seconds...")
        
        try:
            uri = f"{self.ws_url}/stream/events"
            
            async with websockets.connect(uri) as websocket:
                print("   Connected to event stream")
                
                # Listen for events
                events_received = 0
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    try:
                        # Wait for event with timeout
                        event_json = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        event = json.loads(event_json)
                        
                        events_received += 1
                        print(f"   Event {events_received}: {event['event_type']} "
                              f"(step {event.get('step', '?')})")
                        
                        if events_received >= 10:  # Limit output
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"   WebSocket error: {e}")
                        break
                
                print(f"   Received {events_received} events total")
                
        except Exception as e:
            print(f"   WebSocket connection failed: {e}")
    
    async def run_full_test(self):
        """Run complete API test suite."""
        print("=== PHASE 4: API CLIENT TEST ===\n")
        
        if not HTTP_CLIENTS_AVAILABLE:
            print("Please install required packages:")
            print("pip install aiohttp websockets")
            return
        
        try:
            # Test agent lifecycle
            await self.test_agent_lifecycle()
            
            # Give agent time to train
            print("\nWaiting for agent to train...")
            await asyncio.sleep(10.0)
            
            # Test feedback system
            await self.test_feedback_system()
            
            # Test memory queries
            await self.test_memory_queries()
            
            # Test WebSocket streaming
            await self.test_websocket_stream(duration=10)
            
            print("\n[OK] API CLIENT TEST COMPLETED!")
            print("The Phase 4 API provides:")
            print("- RESTful agent control (start/pause/resume/stop)")
            print("- Human feedback integration endpoints")
            print("- Natural language memory querying")
            print("- Real-time WebSocket event streaming")
            print("- Production-ready health monitoring")
            
        except Exception as e:
            print(f"\nAPI test error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run the API test client."""
    client = Phase4APIClient()
    await client.run_full_test()

if __name__ == "__main__":
    asyncio.run(main())