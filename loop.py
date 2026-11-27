"""Run a model and workflow through a bounded request and tool loop."""

import json
import sys
from collections.abc import Callable
from copy import deepcopy
from typing import Never, Protocol
from uuid import uuid4

from anthropic import APIResponseValidationError, AnthropicError
from anthropic.types import Message, MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock

from replay import RecordedResponseError, ReplayMismatch
from run_trace import RUNS, RunTrace, SessionTurn
from workflow import WorkflowSession

MAX_REQUESTS = 8
MAX_OUTPUT_TOKENS = 1_024


class CreateResponse(Protocol):
    def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[ToolParam],
        messages: list[MessageParam],
    ) -> Message: ...


class RunError(RuntimeError):
    """The run stopped before reaching a final answer."""


def run(
    create_response: CreateResponse,
    *,
    task: str,
    model: str,
    workflow: WorkflowSession,
    history: list[MessageParam] | None = None,
    session: SessionTurn | None = None,
    source_run_id: str | None = None,
    observe: Callable[[dict[str, object]], None] | None = None,
) -> str:
    """Run one prompt and commit its completed history to the calling session."""
    run_id = str(uuid4())
    directory = RUNS if session is None else session.directory
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"{run_id}.jsonl"
    messages: list[MessageParam] = [*(history or []), {"role": "user", "content": task}]
    started: dict[str, object] = {
        "run_id": run_id,
        "task": task,
        "model": model,
        "workflow": {"id": workflow.id, "version": workflow.version},
        "mode": "live" if source_run_id is None else "replay",
    }
    if session is not None:
        started.update(session_id=session.session_id, session_turn=session.number)
    if source_run_id is not None:
        started["source_run_id"] = source_run_id

    with trace_path.open("x", encoding="utf-8") as stream:
        print(f"Run trace at {trace_path}", file=sys.stderr)
        trace = RunTrace(stream, observe=observe, replay=source_run_id is not None)
        trace.record("run_started", **started)
        try:
            answer, final_message = _run_turns(create_response, model, workflow, messages, trace)
        except ReplayMismatch:
            trace.finish("failed", verify=False, reason="replay_mismatch")
            raise
        if history is not None:
            history[:] = [*messages, final_message]
        return answer


def _run_turns(
    create_response: CreateResponse,
    model: str,
    workflow: WorkflowSession,
    messages: list[MessageParam],
    trace: RunTrace,
) -> tuple[str, MessageParam]:
    def fail(status: str, reason: str, message: str) -> Never:
        trace.finish(status, reason=reason)
        raise RunError(message)

    for turn in range(1, MAX_REQUESTS + 1):
        try:
            response = create_response(
                model=model, max_tokens=MAX_OUTPUT_TOKENS, system=workflow.system_prompt,
                tools=workflow.tools, messages=messages,
            )
        except (APIResponseValidationError, json.JSONDecodeError):
            fail("failed", "invalid_response", "Claude returned an invalid response.")
        except AnthropicError as error:
            error_type = type(error).__name__
            fail("failed", f"provider_error/{error_type}", f"Claude API request failed ({error_type}).")
        except RecordedResponseError as error:
            fail("failed", error.reason, "Reproduced a recorded model request failure.")

        if not isinstance(response, Message):
            fail("failed", "invalid_response", "Claude returned an invalid response.")
        try:
            # The SDK can deserialize an invalid payload into a Message. Validate
            # before recording so a malformed response leaves a replayable failure.
            data = response.model_dump(exclude_unset=True, warnings=False)
            response = Message.model_validate(data, strict=True)
        except (ValueError, TypeError):
            fail("failed", "invalid_response", "Claude returned an invalid response.")
        if not response.model.strip():
            fail("failed", "invalid_response", "Claude returned an invalid response.")

        trace.response(turn, response)

        if response.stop_reason == "end_turn":
            if any(isinstance(block, ToolUseBlock) for block in response.content):
                fail("failed", "invalid_response", "Claude returned a tool call with its final answer.")
            text_blocks = [block.text for block in response.content if block.type == "text"]
            answer = "\n".join(text_blocks)
            if not answer.strip():
                fail("failed", "empty_answer", "Claude ended the turn without an answer.")
            trace.finish("ok", output=answer)
            return answer, response.to_param()

        if response.stop_reason == "refusal":
            fail("refused", "model_refusal", "Claude refused the request.")
        if response.stop_reason == "max_tokens":
            fail("truncated", "output_token_limit", "Claude reached the response token limit.")
        if response.stop_reason != "tool_use":
            fail("failed", "unexpected_stop_reason", "Claude returned an unexpected stop reason.")
        if turn == MAX_REQUESTS:
            break

        # Validate the whole batch before any tool can change workflow state.
        calls = [block for block in response.content if isinstance(block, ToolUseBlock)]
        invalid_call = any(
            not isinstance(call.id, str)
            or not call.id.strip()
            or not isinstance(call.name, str)
            or not call.name.strip()
            or not isinstance(call.input, dict)
            for call in calls
        )
        if not calls or invalid_call or len({call.id for call in calls}) != len(calls):
            fail("failed", "invalid_tool_calls", "Claude returned an invalid tool call batch.")

        messages.append(response.to_param())
        results: list[ToolResultBlockParam] = []
        for call in calls:
            trace.record("tool_intent", turn=turn, call_id=call.id, name=call.name, args=call.input)
            outcome = workflow.execute_tool(call.name, deepcopy(call.input))
            content = json.dumps(outcome.content)
            trace.record("tool_result", turn=turn, call_id=call.id, is_error=outcome.is_error, content=content)
            results.append({
                "type": "tool_result", "tool_use_id": call.id,
                "content": content, "is_error": outcome.is_error,
            })
        messages.append({"role": "user", "content": results})

    fail("budget_exceeded", "request_limit", "The run reached its eight-request limit.")


if __name__ == "__main__":
    from cli import main

    raise SystemExit(main())
