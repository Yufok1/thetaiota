# Contributing to ThetaIota

Thank you for your interest in contributing to ThetaIota! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

## Development Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch (CPU or CUDA)
- Git

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
flake8 .
mypy .
```

## Project Structure

```
thetaiota/
├── core/                    # Core agent components
│   ├── phase1_agent.py     # Base learner
│   ├── phase2_agent.py     # Meta-controller
│   ├── phase3_agent.py     # Self-awareness
│   └── phase4_agent_service.py  # Production service
├── api/                     # API layer
│   └── phase4_api_server.py
├── models/                  # Neural network models
│   ├── chat_engine.py      # Conversational LM
│   └── transformer_model.py
├── memory/                  # Memory and persistence
│   └── memory_db.py
├── cli/                     # Command line interface
│   └── cli_control.py
└── training/                # Training scripts
    ├── train_tiny_lm.py
    └── train_conversational_lm_windows.py
```

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Test your changes**:
   ```bash
   pytest
   python -m thetaiota.test_phase1  # Run basic tests
   ```

4. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "feat: add new self-reflection capability"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for all functions and classes
- Keep functions focused and small

### Commit Messages
Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### Code Review
- All PRs require review
- Address review comments promptly
- Ensure tests pass
- Update documentation as needed

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test_phase1.py

# Run with coverage
pytest --cov=thetaiota
```

### Writing Tests
- Write tests for new features
- Aim for good test coverage
- Use descriptive test names
- Test both success and failure cases

## Documentation

### Updating Documentation
- Update README.md for user-facing changes
- Update docstrings for API changes
- Update CHEATSHEET.md for new commands
- Add examples where helpful

### Documentation Standards
- Use clear, concise language
- Include code examples
- Keep documentation up to date
- Use proper markdown formatting

## Issues and Bug Reports

### Reporting Bugs
1. Check existing issues first
2. Use the bug report template
3. Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Feature Requests
1. Check existing issues first
2. Use the feature request template
3. Explain the use case and benefits
4. Consider implementation complexity

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Share knowledge and insights
- Follow the project's code of conduct

## Getting Help

- Check the documentation first
- Search existing issues
- Ask questions in discussions
- Join the community chat (if available)

## License

By contributing to ThetaIota, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to ThetaIota! 🧠✨
