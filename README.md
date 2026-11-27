# agent-harness

A small Python agent runtime for synthetic retail order support. The first milestone is a Claude tool loop with a per-run JSONL trace. The roadmap adds replay, regression evals, and crash recovery for actions such as refunds.

## Development setup

Install `uv`, then run these commands from the repository root to sync the Python 3.12 environment. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync --locked
uv run --locked python -c 'import anthropic; print(anthropic.__version__)'
```

## Run the v0.1 demo

In the Claude Console, select **Create key**, enter a name and expiration, set **Linked account** to yourself, and select a workspace to scope the personal key to. Copy the key when it appears. [Claude Console API keys](https://platform.claude.com/settings/keys) · [Claude API key guide](https://platform.claude.com/docs/en/get-api-key)

Live requests use your Console organization’s API billing or available credits. [Claude API billing](https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage)

In the zsh terminal where you’ll run the demo, paste the key at the hidden prompt and press Enter. You can also have your credential manager set `ANTHROPIC_API_KEY` for that shell.

```zsh
read -rs 'ANTHROPIC_API_KEY?Anthropic API key: '
printf '\n'
export ANTHROPIC_API_KEY
```

From the repository root, start the interactive agent.

```sh
uv run --locked python loop.py
```

Try this exchange. Claude’s wording may vary.

```text
Synthetic order support. Type exit or press Ctrl-D to quit.
You> Please refund the mug from order ORD-1001.
Agent> I issued a simulated $25.00 refund for the mug.
You> Please refund it again.
Agent> The mug has already been refunded, so no additional refund was issued.
You> exit
```

Each task prompt gets up to eight Claude requests and its own JSONL trace under `runs/`. The CLI prints the answer to stdout and the trace path to stderr. The session remembers completed conversation turns and synthetic order changes, so the second request can refer to the mug and the store rules prevent a duplicate refund. Each new process starts with fresh synthetic orders.

Type `exit` or press Ctrl-D to close the session. Ctrl-C interrupts it. A quoted task runs once. Both modes default to Claude Haiku 5.5 (`claude-haiku-5-5`) and accept a `--model` override.

```sh
uv run --locked python loop.py 'Please refund the mug from order ORD-1001.'
```

## Read the code

- Follow the CLI from argument parsing through model requests, tool results, and JSONL events. [loop.py](loop.py)
- See the tool schemas and the store rules for order lookup, cancellation, and refunds. [tools.py](tools.py)
- Inspect the synthetic orders, customer identity, and reference date for the refund window. [orders.json](orders.json)

## Design

- [Architecture and release roadmap](designs/architecture.md)
- [v0.1 CLI and run trace](designs/v0.1.md)
