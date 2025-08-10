# Phase 0: Design Document - Self-Reflective AI Agent

## 1. Agent Scope & Self-Awareness Definition

### What "Self-Awareness" Means Here
- **Introspection Capability**: Agent monitors its own training dynamics (loss curves, gradient norms, parameter drift)
- **Decision Transparency**: Agent can explain why it triggered self-updates or spawned sub-tasks
- **Performance Recognition**: Agent detects when it's struggling, plateauing, or improving
- **Meta-Cognitive Logging**: Agent maintains a record of its own "thoughts" and decisions for later analysis

### Introspection Targets (What We Monitor)
| Signal | Frequency | Purpose |
|--------|-----------|---------|
| Training/Validation Loss | Every batch | Detect learning plateau, convergence |
| Gradient L2 Norm | Every optimizer step | Identify exploding/vanishing gradients |
| Parameter Statistics (mean/std) | Every N steps | Monitor weight drift, training stability |
| Token Confidence Scores | Per output | Detect uncertainty, potential errors |
| GPU Memory Usage | Every batch | Stay within hardware limits (3GB GTX 1060) |

## 2. Task-Summoning Semantics

### Task Representation
```python
@dataclass(order=True)
class Task:
    priority: int              # 0 = highest priority
    prompt: str                # Description of what to do
    objective: str             # Success criteria (e.g., "reduce val_loss by 0.1")
    created_by: str            # "user" or "self_spawn"
    dependencies: List[str]    # Task IDs that must complete first
    metadata: Dict[str, Any]   # Flexible additional data
```

### Sub-Task Generation Rules
- **Low Confidence Trigger**: If output confidence < 0.7, spawn "gather_more_examples" task
- **Plateau Trigger**: If val_loss stagnant for 3+ epochs, spawn "hyperparameter_tune" task  
- **Error Pattern Trigger**: If same mistake repeated 3+ times, spawn "focused_fine_tune" task
- **Resource Limit Trigger**: If GPU memory > 2.8GB, spawn "optimize_memory_usage" task

## 3. Meta-Learning Objectives

### What The Agent Can Adjust About Itself
1. **Learning Rate**: Multiply by 0.5-2.0x based on gradient behavior
2. **Training Mode**: Switch between normal training, fine-tuning, or data collection
3. **Attention Focus**: Prioritize certain types of examples or tasks
4. **Memory Management**: Decide when to checkpoint, when to clear cache

### Success Metrics
- Primary: Validation loss improvement
- Secondary: Training stability (gradient norm variance)
- Tertiary: Human feedback scores (Phase 3+)

## 4. Interaction Modality

### Phase 1: Local Script Interface
- CLI-based interaction for development and debugging
- Text file logging for introspection review
- Manual feedback through simple rating system

### Phase 4+: API Interface  
- HTTP endpoints for prompts, status queries, feedback
- Web dashboard for real-time introspection visualization
- Multi-instance coordination capabilities

## 5. Success Criteria for Phase 0

✅ This design document completed
⏳ Architectural blueprint created
⏳ Resource constraints documented
⏳ All Phase 0 deliverables verified before Phase 1 coding begins

---

*Phase 0 Tag: phase0.design_complete*