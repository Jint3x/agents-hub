# agents-hub

A framework-free multi-agent system built with vanilla Python and direct API calls.

## Overview

agents-hub is designed to be a lightweight, flexible multi-agent system where:

- Agents are implemented as separate files
- Reusable agentic components live in dedicated sub-packages
- No heavy frameworks — just clean Python and direct API calls

## Tech Stack

| Tool        | Purpose                    |
|-------------|----------------------------|
| Python 3.14 | Runtime                    |
| uv          | Package management         |
| hatchling   | Build backend              |
| pytest      | Testing                    |
| ruff        | Linting & formatting       |
| mypy        | Static type checking       |

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run type checker
uv run mypy src
```

## Project Structure

```
agents-hub/
├── src/agents_hub/
│   ├── __init__.py
│   ├── agents/        # Individual agent implementations
│   ├── components/    # Reusable agentic components
│   ├── core/          # Core logic, vanilla API clients
│   └── utils/         # Shared utilities
├── tests/             # Test suite
├── docs/              # Documentation
├── pyproject.toml     # Project configuration
├── README.md          # This file
├── LICENSE            # MIT License
└── AGENTS.md          # Agent working rules
```

## License

MIT — see [LICENSE](LICENSE).
