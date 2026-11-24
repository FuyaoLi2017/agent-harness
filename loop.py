"""CLI entry point for the synthetic order support harness."""

import argparse
import os
import sys

from anthropic import Anthropic

from tools import OrderFixture, load_orders


def run(client: Anthropic, *, task: str, model: str, orders: OrderFixture) -> str:
    """Run one support task and return the final answer."""
    # TODO: Send the task with GET_ORDER_TOOL and a 1,024-token response limit.
    # TODO: Call get_order for tool requests and batch results into one user message.
    # TODO: Enforce the eight-request limit and record the JSONL trace.
    raise NotImplementedError("Implement the Messages loop in run().")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic order support task.")
    parser.add_argument("task", help="A synthetic customer request.")
    parser.add_argument(
        "--model", default="claude-opus-5-5", help="Claude model (default: %(default)s)."
    )
    args = parser.parse_args()

    if not args.task.strip():
        parser.error("task must not be empty")
    if not args.model.strip():
        parser.error("--model must not be empty")

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        parser.error("set ANTHROPIC_API_KEY before running a task")

    try:
        orders = load_orders()
    except (OSError, ValueError) as error:
        print(f"Could not load orders.json: {error}", file=sys.stderr)
        return 1

    try:
        with Anthropic(api_key=api_key, max_retries=0) as client:
            answer = run(client, task=args.task, model=args.model, orders=orders)
    except NotImplementedError as error:
        print(error, file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
