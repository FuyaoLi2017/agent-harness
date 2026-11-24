# agent-harness

A small Python agent runtime for synthetic retail order support. The first milestone is a Claude tool loop with a per-run JSONL trace. The roadmap adds replay, regression evals, and crash recovery for actions such as refunds.

## Development setup

Install `uv`, then run these commands from the repository root to sync the Python 3.12 environment. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync --locked
uv run --locked python -c 'import anthropic; print(anthropic.__version__)'
```

## Design

- [Architecture and release roadmap](designs/architecture.md)
- [v0.1 CLI and run trace](designs/v0.1.md)
