#!/usr/bin/env python3
"""
Simple canary evaluation system to prevent catastrophic forgetting.
"""

import json
import os
from typing import List, Dict, Tuple
import torch
from chat_engine import ChatEngine, ChatConfig


class CanaryEvaluator:
    """Simple regression test for conversational quality."""
    
    def __init__(self, canary_path: str = "eval/canary_prompts.jsonl"):
        self.canary_path = canary_path
        self.prompts = self._load_canary_prompts()
        
    def _load_canary_prompts(self) -> List[Dict]:
        """Load canary prompts from JSONL file."""
        prompts = []
        if os.path.exists(self.canary_path):
            with open(self.canary_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(json.loads(line))
        return prompts
    
    def evaluate_checkpoint(self, agent_ref=None) -> Dict[str, float]:
        """
        Evaluate conversational quality on canary prompts.
        
        Returns:
            dict with 'mean_score', 'min_score', 'max_score', 'total_prompts'
        """
        if not self.prompts:
            return {"mean_score": 0.0, "min_score": 0.0, "max_score": 0.0, "total_prompts": 0}
        
        # Set up chat engine for evaluation
        config = ChatConfig(mode="lm", temperature=0.8, top_k=20)
        chat_engine = ChatEngine(agent_ref=agent_ref, config=config)
        
        scores = []
        
        for prompt_data in self.prompts:
            prompt = prompt_data["prompt"]
            expected_quality = prompt_data.get("expected_quality", 3.0)
            
            try:
                # Generate response
                response = chat_engine.generate_lm_response("default", prompt)
                reply = response.get("reply", "")
                
                # Simple quality scoring
                score = self._score_response(reply, expected_quality)
                scores.append(score)
                
            except Exception as e:
                # Failed generation gets lowest score
                scores.append(0.0)
                print(f"Canary evaluation failed for '{prompt[:30]}...': {e}")
        
        if not scores:
            return {"mean_score": 0.0, "min_score": 0.0, "max_score": 0.0, "total_prompts": 0}
        
        return {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "total_prompts": len(scores)
        }
    
    def _score_response(self, response: str, expected_quality: float) -> float:
        """
        Simple heuristic scoring of response quality.
        
        Args:
            response: Generated response text
            expected_quality: Expected quality level (1-5)
            
        Returns:
            Quality score (0-5)
        """
        if not response or response == "(thinking…)":
            return 0.0
        
        response = response.strip()
        
        # Basic quality heuristics
        score = 1.0  # Base score for non-empty response
        
        # Length reasonableness (not too short, not repetitive garbage)
        if len(response) > 10:
            score += 0.5
        if len(response) > 30:
            score += 0.5
            
        # Not just gibberish (contains some common words)
        common_words = {"the", "a", "an", "is", "are", "and", "or", "but", "to", "of", "in", "on", "for", "with", "by", "I", "you", "we", "they", "this", "that"}
        words = response.lower().split()
        if any(word in common_words for word in words):
            score += 1.0
            
        # Not excessively repetitive
        if len(set(words)) > len(words) * 0.3:  # Reasonable word diversity
            score += 0.5
            
        # Reasonable punctuation/capitalization
        if response[0].isupper() and any(p in response for p in '.!?'):
            score += 0.5
        
        # Cap at reasonable maximum
        return min(score, 5.0)


def test_canary_system():
    """Test the canary evaluation system."""
    evaluator = CanaryEvaluator()
    
    if not evaluator.prompts:
        print("No canary prompts found - create eval/canary_prompts.jsonl")
        return
    
    print(f"Loaded {len(evaluator.prompts)} canary prompts")
    print("Testing canary evaluation (without model)...")
    
    # Mock some responses for testing
    test_responses = [
        "Hello! I'm doing well, thank you for asking.",
        "gibberish random words nonsense",
        "(thinking…)",
        "",
        "My purpose is to help and assist users with various tasks and questions."
    ]
    
    evaluator = CanaryEvaluator()
    for i, response in enumerate(test_responses):
        score = evaluator._score_response(response, 3.0)
        print(f"Response {i+1}: '{response[:50]}...' -> Score: {score:.1f}")


if __name__ == "__main__":
    test_canary_system()