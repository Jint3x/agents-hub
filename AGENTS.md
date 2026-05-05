# AGENTS.md — agents-hub

## Project Purpose

agents-hub is a **framework-free multi-agent system**.

We build agents as plain Python files that make direct API calls. No LangChain, no CrewAI, no heavy abstractions — just clear, maintainable code.

## Tech Stack & Rationale

| Tool       | Why we use it                                          |
|------------|--------------------------------------------------------|
| Python 3.14| Latest stable, pattern matching, better typing         |
| uv         | Fast, modern package manager and virtual env manager   |
| hatchling  | Clean PEP 518 build backend; works seamlessly with uv  |
| pytest     | Standard, powerful testing                             |
| ruff       | Fast linting and formatting in one tool                |
| mypy       | Strict static typing catches bugs early                |

## Architecture

We use a **src-layout**:

```
src/agents_hub/
├── agents/        # Individual agent implementations — one file per agent
├── components/    # Reusable agentic components (memory, tools, planning, etc.)
├── core/          # Core logic: vanilla HTTP/API clients, retry logic, auth
└── utils/         # Shared utilities: logging helpers, config loaders, etc.
```

### Directory Map

| Package        | Contents                                                           |
|----------------|--------------------------------------------------------------------|
| `agents`       | Concrete agent classes/workflows. One file = one agent.            |
| `components`   | Pluggable pieces agents can share: memory stores, tool registries. |
| `core`         | Low-level infrastructure: API clients, rate-limiting, streaming.   |
| `utils`        | Cross-cutting helpers: env parsing, structured logging.            |

## How to Run

```bash
# Run the test suite
uv run pytest

# Run the linter
uv run ruff check .

# Run the type checker
uv run mypy src
```

## Agent Working Rules

1. **Framework-free always.** Use `httpx` or `urllib` for HTTP. No LangChain, no LlamaIndex, no CrewAI unless explicitly approved.
2. **Tests before commit.** Every behavior change needs a test. Run `uv run pytest` before committing.
3. **Type everything.** Use strict mypy. Annotate public functions, class attributes, and module-level variables.
4. **One agent per file.** Keep agents in `src/agents_hub/agents/`. Name the file after the agent.
5. **Reuse components.** Before writing new logic, check if `components/` already has something close.
6. **No secrets in code.** Load API keys from environment variables or a `.env` file (never commit `.env`).
7. **Keep PRs small.** One logical change per PR. Easier to review, easier to revert.
