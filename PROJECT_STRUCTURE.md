# ThetaIota Project Structure

This document provides a comprehensive overview of the ThetaIota codebase organization and architecture.

## Directory Structure

```
thetaiota/
├── 📁 Core System
│   ├── phase1_agent.py              # Base learner with self-update capabilities
│   ├── phase2_agent.py              # Meta-controller and task spawning
│   ├── phase3_agent.py              # Self-awareness and reflection
│   ├── phase4_agent_service.py      # Production service wrapper
│   └── phase4_api_server.py         # FastAPI REST server
│
├── 📁 Neural Models
│   ├── chat_engine.py               # Conversational LM (150M params)
│   ├── transformer_model.py         # Base transformer architecture
│   └── train_tiny_lm.py             # Training script for LM
│
├── 📁 Memory & Persistence
│   ├── memory_db.py                 # SQLite-based memory system
│   ├── reflection_explainer.py      # Decision explanation
│   ├── memory_summarizer.py         # Memory analysis and summarization
│   └── human_feedback_system.py     # Feedback integration
│
├── 📁 Federation & Communication
│   ├── phase5_communication.py      # Inter-agent messaging
│   ├── phase5_consensus.py          # Consensus mechanisms
│   ├── phase5_registry.py           # Agent registry
│   └── phase5_shared_memory.py      # Shared memory system
│
├── 📁 CLI & Interface
│   ├── cli_control.py               # Main CLI interface
│   ├── djinn.bat                    # Windows CLI wrapper
│   └── start_all.bat                # Federation launcher
│
├── 📁 Training & Development
│   ├── train_conversational_lm_windows.py  # Enhanced training
│   ├── curriculum_dataset.py        # Training data management
│   ├── meta_controller.py           # Meta-learning controller
│   └── task_spawner.py              # Dynamic task generation
│
├── 📁 Evaluation & Testing
│   ├── canary_eval.py               # Model evaluation
│   ├── eval/canary_prompts.jsonl    # Test prompts
│   └── test_*.py                    # Various test files
│
├── 📁 Guardian & Safety
│   └── guardian/validator.py        # Safety validation
│
├── 📁 Documentation
│   ├── README.md                    # Main documentation
│   ├── CHEATSHEET.md                # Quick reference
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   ├── CHANGELOG.md                 # Version history
│   └── PROJECT_STRUCTURE.md         # This file
│
├── 📁 Configuration
│   ├── requirements.txt             # Python dependencies
│   ├── setup.py                     # Package configuration
│   ├── .gitignore                   # Git ignore rules
│   └── LICENSE                      # MIT License
│
├── 📁 GitHub
│   ├── .github/workflows/ci.yml     # CI/CD pipeline
│   ├── .github/ISSUE_TEMPLATE/      # Issue templates
│   └── .github/pull_request_template.md
│
└── 📁 Data & Models
    ├── checkpoints/                 # Model weights
    ├── *.db                         # SQLite databases
    └── *.txt                        # Linguistic data files
```

## Architecture Overview

### 🧠 One Brain, Three Lobes
```
Agent A (Leader) ←→ Agent B (Peer) ←→ Agent C (Peer)
     ↓                    ↓                    ↓
  User Interface    Quorum Voting      Weight Sync
  Chat Surface      Consensus          Checkpoint Pull
  Memory Leader     Heartbeat          Follower Mode
```

### 🔄 System Flow
1. **Initialization**: Three agents start with shared configuration
2. **Leadership**: Agent A becomes user-facing surface
3. **Federation**: B/C provide quorum voting and weight synchronization
4. **Communication**: Heartbeat monitoring and message passing
5. **Consensus**: 2-of-3 approval for sensitive operations
6. **Memory**: Shared introspection and meta-event logging

### 🏗️ Component Relationships

#### Core Agent Stack
```
Phase 4 (Production Service)
    ↓
Phase 3 (Self-Awareness)
    ↓
Phase 2 (Meta-Controller)
    ↓
Phase 1 (Base Learner)
```

#### Memory System
```
MemoryDB (SQLite)
    ↓
Introspection → Reflection → Summarization
    ↓
Human Feedback → Decision Explanation
```

#### Neural Models
```
ChatEngine (150M params)
    ↓
TinyCausalLM (12-layer transformer)
    ↓
ByteTokenizer (UTF-8 encoding)
```

## Key Files Explained

### Core System Files
- **`phase1_agent.py`**: Base learning agent with self-update triggers
- **`phase2_agent.py`**: Meta-controller that spawns tasks and manages learning
- **`phase3_agent.py`**: Self-aware agent with reflection and feedback
- **`phase4_agent_service.py`**: Production wrapper with lifecycle management
- **`phase4_api_server.py`**: FastAPI server with authentication and monitoring

### Neural Model Files
- **`chat_engine.py`**: Conversational interface with 150M parameter LM
- **`transformer_model.py`**: Base transformer architecture with introspection
- **`train_tiny_lm.py`**: Training script for the conversational model

### Memory System Files
- **`memory_db.py`**: SQLite-based persistent storage with WAL mode
- **`reflection_explainer.py`**: Generates explanations for agent decisions
- **`memory_summarizer.py`**: Analyzes and summarizes memory patterns
- **`human_feedback_system.py`**: Integrates human feedback into learning

### Federation Files
- **`phase5_communication.py`**: Inter-agent messaging system
- **`phase5_consensus.py`**: Consensus mechanisms for distributed decisions
- **`phase5_registry.py`**: Agent registration and discovery
- **`phase5_shared_memory.py`**: Shared memory across federation

### Interface Files
- **`cli_control.py`**: Main command-line interface with natural language
- **`djinn.bat`**: Windows wrapper for easy CLI access
- **`start_all.bat`**: Launches the three-agent federation

## Data Flow

### Training Pipeline
```
Project Text → Tokenizer → TinyCausalLM → Training Loop → Checkpoints
```

### Chat Pipeline
```
User Input → ChatEngine → Reflect/LM Mode → Response → Memory Log
```

### Federation Pipeline
```
Agent Decision → Quorum Check → Peer Voting → Consensus → Action
```

### Memory Pipeline
```
Introspection → SQLite Storage → Analysis → Reflection → Summary
```

## Configuration

### Environment Variables
- `RA_BIND_HOST`: Server bind address (default: 127.0.0.1)
- `RA_PORT_BASE`: Base port for federation (default: 8081)
- `RA_REPLICAS`: Number of agents (default: 3)
- `RA_DB_PATH`: Database path per agent
- `RA_CHECKPOINT_DIR`: Model checkpoint directory
- `RA_AUTH_SECRET`: Authentication secret for production
- `RA_RATE_LIMIT`: API rate limiting (default: 20/minute)

### Model Configuration
- **Architecture**: 12-layer transformer
- **Parameters**: ~150M parameters
- **Dimensions**: 1024 d_model, 4096 d_ff
- **Context**: 512 tokens
- **Vocabulary**: 258 tokens (byte-level + specials)

## Development Workflow

### Local Development
1. Clone repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`
5. Start federation: `start_all.bat`
6. Use CLI: `djinn.bat`

### Production Deployment
1. Set environment variables
2. Use `server_main.py` for production launcher
3. Configure authentication and rate limiting
4. Set up monitoring and health checks
5. Deploy with systemd (Linux) or NSSM (Windows)

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock federation for isolated testing
- Memory system validation
- Model training verification

### Integration Tests
- Federation communication
- End-to-end training pipeline
- API endpoint validation
- CLI command testing

### Performance Tests
- Memory usage monitoring
- Training speed benchmarks
- Federation latency measurement
- Model inference timing

This structure provides a solid foundation for a self-reflective AI system with distributed capabilities and production-ready features.
