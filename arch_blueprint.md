# Architectural Blueprint - Self-Reflective AI Agent

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL INTERFACE LAYER                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ CLI Handler │  │ File Logger │  │ Feedback    │             │
│  │             │  │             │  │ Interface   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    META-CONTROLLER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Introspect  │  │ Task Queue  │  │ Self-Update │             │
│  │ Monitor     │  │ Manager     │  │ Trigger     │             │
│  │             │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    BASE LEARNER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ 1-Layer     │  │ Training    │  │ Forward     │             │
│  │ Transformer │  │ Loop        │  │ Pass        │             │
│  │             │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 MEMORY & KNOWLEDGE BASE                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ SQLite DB   │  │ Introspect  │  │ Task        │             │
│  │             │  │ Logs        │  │ History     │             │
│  │             │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Base Learner (Core Model)
**Purpose**: The central AI model that performs the primary learning task

**Components**:
- **1-Layer Transformer**: Minimal transformer with self-attention and FFN
- **Training Loop**: Manual PyTorch training with introspection hooks  
- **Forward Pass**: Generates outputs with confidence scores

**Inputs**: Training data batches, learning parameters
**Outputs**: Predictions, loss values, confidence scores, gradient norms

### 2. Meta-Controller (Executive Function)
**Purpose**: Monitors Base Learner performance and makes self-improvement decisions

**Components**:
- **Introspect Monitor**: Collects and analyzes performance metrics
- **Task Queue Manager**: Handles prioritized task scheduling  
- **Self-Update Trigger**: Decides when/how to modify training

**Decision Flow**:
```
Monitor Metrics → Detect Patterns → Generate Tasks → Execute Actions
     ↓                  ↓               ↓              ↓
   Loss/Grads      Plateau/Error    Tune Params    Update Model
```

### 3. Memory & Knowledge Base (Persistent Storage)
**Purpose**: Stores agent's experience and introspective data

**Storage Schema**:
- **introspection**: Metrics logged during training (loss, gradients, etc.)
- **tasks**: Queue of pending and completed tasks with priorities
- **meta_events**: Audit trail of self-update decisions and outcomes
- **config**: System configuration and thresholds

### 4. External Interface Layer (I/O Management)
**Purpose**: Handles all interaction with the outside world

**Phase 1 Components**:
- **CLI Handler**: Basic command-line interaction
- **File Logger**: Outputs training logs and visualizations
- **Feedback Interface**: Simple rating system for human input

## Data Flow Architecture

### Training Cycle
1. **Base Learner** processes batch → generates loss/metrics
2. **Meta-Controller** receives metrics → analyzes for patterns
3. **Memory** logs all metrics and decisions
4. **Meta-Controller** spawns tasks if needed → queues actions
5. **Base Learner** executes queued self-updates
6. Loop repeats

### Self-Reflection Cycle
1. **Introspect Monitor** analyzes recent performance trends
2. Compares current metrics to historical patterns in **Memory**
3. Generates explanation of current state and recent decisions
4. Stores reflection in **Memory** for future reference

## Hardware Constraints Integration

**Memory Management**:
- All components designed for <3GB GPU memory (GTX 1060)
- Gradient checkpointing in Base Learner
- Batch size limits enforced by Meta-Controller
- Memory usage monitoring in External Interface

**Processing Flow**:
- CPU-based database operations (Memory component)
- GPU-based model training (Base Learner core)
- Mixed CPU/GPU meta-analysis (Meta-Controller)

---

*Phase 0 Tag: phase0.architecture_complete*