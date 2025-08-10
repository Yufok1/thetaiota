# Changelog

All notable changes to ThetaIota will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced 150M parameter conversational LM (12-layer transformer)
- Live text generation display with 84 diverse conversation prompts
- Activation checkpointing, gradient accumulation, AMP optimizations
- Production-ready API server with authentication and rate limiting
- Health monitoring endpoints (/healthz, /readyz, /metrics)
- Federation system with quorum voting and heartbeat monitoring
- Memory replication and checkpoint synchronization
- Human feedback integration system
- Self-reflection and introspection capabilities

### Changed
- Unified model configuration to 150M parameters across all components
- Improved training scripts with enhanced logging and monitoring
- Enhanced CLI with natural language prompt interface
- Streamlined deployment process with environment-based configuration

### Fixed
- Model parameter count consistency across documentation and implementation
- Training configuration alignment between scripts and documentation
- Memory management and database connection handling
- Federation communication reliability

## [1.0.0] - 2024-01-XX

### Added
- Initial release of ThetaIota
- Three-agent federation system (A/B/C)
- Self-reflective AI agent with meta-learning capabilities
- Native on-device conversational LM
- SQLite-based memory system with introspection
- FastAPI-based REST API
- Command-line interface with natural language support
- Training pipeline for conversational language models
- Quorum voting system for sensitive operations
- Heartbeat monitoring and checkpoint synchronization
- Human feedback integration
- Production deployment support

### Features
- **One Brain, Three Lobes Architecture**: Leader (A) + two peer agents (B/C)
- **Self-Reflection**: Agent can explain its decisions and learn from feedback
- **Federation**: Distributed consensus and weight synchronization
- **Native LM**: 150M parameter transformer for conversational AI
- **Memory System**: Persistent storage of introspection, tasks, and meta-events
- **Production Ready**: Authentication, rate limiting, health monitoring
- **CLI Interface**: Natural language commands and advanced controls
- **Training Pipeline**: End-to-end training with live monitoring

### Technical Specifications
- **Model**: 12-layer transformer, 1024 d_model, 4096 d_ff
- **Parameters**: ~150M parameters
- **Context**: 512 tokens
- **Federation**: 2-of-3 quorum voting
- **Memory**: SQLite with WAL mode
- **API**: FastAPI with optional authentication
- **Platform**: Python 3.10+, PyTorch, Windows/Linux

---

## Version History

- **1.0.0**: Initial release with complete federation system
- **Unreleased**: Enhanced training, production features, and optimizations

For detailed information about each release, see the [GitHub releases page](https://github.com/Yufok1/thetaiota/releases).
