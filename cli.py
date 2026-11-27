"""Select workflows and drive live and recorded agent runs."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from anthropic import Anthropic
from anthropic.types import MessageParam

from loop import CreateResponse, RunError, run
from replay import ReplayCursor, ReplayInputError, ReplayMismatch, load_recording
from run_trace import SessionTurn, seal_session
from workflow import WorkflowSession
from workflows import DEFAULT_WORKFLOW, WorkflowError, open_workflow

DEFAULT_MODEL = "claude-haiku-5-5"


def _reply(
    create_response: CreateResponse,
    *,
    task: str,
    model: str,
    workflow: WorkflowSession,
    history: list[MessageParam] | None = None,
    session: SessionTurn | None = None,
) -> int:
    try:
        answer = run(
            create_response, task=task, model=model, workflow=workflow,
            history=history, session=session,
        )
    except RunError as error:
        print(error, file=sys.stderr)
        return 1

    print(answer if history is None else f"Agent> {answer}")
    return 0


def _chat(create_response: CreateResponse, *, model: str, workflow: WorkflowSession) -> int:
    history: list[MessageParam] = []
    session_id = str(uuid4())
    prompt_count = 0
    print(f"{workflow.display_name}. Type exit or press Ctrl-D to quit.")
    try:
        while True:
            try:
                task = input("You> ")
            except EOFError:
                print()
                break

            command = task.strip().casefold()
            if command == "exit":
                break
            if command:
                prompt_count += 1
                _reply(
                    create_response, task=task, model=model, workflow=workflow, history=history,
                    session=SessionTurn(session_id, prompt_count),
                )
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130

    if prompt_count:
        seal_session(SessionTurn(session_id, prompt_count))
    return 0


def _replay(path: Path) -> int:
    try:
        recording = load_recording(path)
        first = recording.runs[0]
        workflow = open_workflow(first.workflow_id, first.workflow_version)
    except (ReplayInputError, WorkflowError) as error:
        print(f"Could not replay {path}: {error}", file=sys.stderr)
        return 2

    session_id = str(uuid4())
    history: list[MessageParam] | None = [] if recording.is_session else None
    last_turn = None
    try:
        for number, source in enumerate(recording.runs, 1):
            cursor = ReplayCursor(source)
            last_turn = SessionTurn(session_id, number) if recording.is_session else None
            try:
                answer = run(
                    cursor.create, task=source.task, model=source.model, workflow=workflow,
                    history=history, session=last_turn, source_run_id=source.run_id,
                    observe=cursor.observe,
                )
            except RunError as error:
                # A recorded failure reaches here only after its terminal event matches.
                print(f"Replayed prompt {number}: {error}", file=sys.stderr)
            else:
                print(answer)
        if last_turn is not None:
            seal_session(last_turn)
    except (ReplayMismatch, OSError) as error:
        print(f"Replay stopped: {error}", file=sys.stderr)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or replay an agent workflow.")
    parser.add_argument("task", nargs="?", help="A request to run once. Omit to chat.")
    parser.add_argument("--model", help=f"Claude model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--workflow", help=f"Installed workflow ID (default: {DEFAULT_WORKFLOW}).")
    parser.add_argument("--replay", type=Path, help="Replay a standalone trace or an interactive session directory.")
    args = parser.parse_args(argv)

    if args.replay is not None:
        if any(value is not None for value in (args.task, args.model, args.workflow)):
            parser.error("--replay cannot be combined with a task, --model, or --workflow")
        return _replay(args.replay)

    for name, value in (("task", args.task), ("--model", args.model), ("--workflow", args.workflow)):
        if value is not None and not value.strip():
            parser.error(f"{name} must not be empty")

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        parser.error("set ANTHROPIC_API_KEY before running a task")

    try:
        workflow = open_workflow(args.workflow or DEFAULT_WORKFLOW)
    except WorkflowError as error:
        parser.error(str(error))

    model = args.model or DEFAULT_MODEL
    try:
        with Anthropic(api_key=api_key, max_retries=0) as client:
            if args.task is None:
                return _chat(client.messages.create, model=model, workflow=workflow)
            return _reply(client.messages.create, task=args.task, model=model, workflow=workflow)
    except OSError as error:
        print(error, file=sys.stderr)
        return 1
