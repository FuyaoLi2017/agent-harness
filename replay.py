"""Read recorded runs and verify the events produced during offline replay."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Never

from anthropic.types import Message


class ReplayInputError(ValueError):
    """The recording is missing, incomplete, or malformed."""


class ReplayMismatch(RuntimeError):
    """The runtime diverged from the recorded events."""


class RecordedResponseError(RuntimeError):
    """Reproduce a sanitized failure that prevented a model response."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReplayRun:
    run_id: str
    task: str
    model: str
    workflow_id: str
    workflow_version: int
    _events: tuple[str, ...] = field(repr=False)
    session_id: str | None = None
    session_turn: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayRecording:
    runs: tuple[ReplayRun, ...]
    is_session: bool


_EVENT_FIELDS = {
    "llm_response": ("turn", "model", "stop_reason", "content", "usage"),
    "tool_intent": ("turn", "call_id", "name", "args"),
    "tool_result": ("turn", "call_id", "is_error", "content"),
    "run_finished": ("status", "reason", "output"),
}
_NEW_RUN_FIELDS = {"workflow", "mode", "source_run_id", "session_id", "session_turn"}
_TERMINAL_STATUSES = {"ok", "failed", "refused", "truncated", "budget_exceeded"}
_PROVIDER_ERROR = re.compile(r"provider_error/[A-Za-z_][A-Za-z0-9_]*")
_REASON = re.compile(r"[a-z][a-z0-9_]*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayInputError(message)


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayInputError(f"{field_name} must be a nonempty string")
    return value


def _number(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ReplayInputError(f"{field_name} must be a positive integer")
    return value


def _reject_constant(_: str) -> Never:
    raise ReplayInputError("non-finite JSON value")


def _object(data: str) -> dict[str, Any]:
    try:
        value = json.loads(data, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ReplayInputError("invalid JSON") from None
    if not isinstance(value, dict):
        raise ReplayInputError("expected a JSON object")
    return value


def _schema_version(event: dict[str, Any]) -> None:
    _require(_number(event.get("schema_version"), "schema_version") == 1, "unsupported schema version")


def _envelope(event: dict[str, Any], seq: int) -> str:
    _schema_version(event)
    _require(_number(event.get("seq"), "seq") == seq, "incorrect event sequence")
    try:
        timestamp = datetime.fromisoformat(_text(event.get("ts"), "ts"))
    except ValueError:
        raise ReplayInputError("invalid event timestamp") from None
    _require(timestamp.utcoffset() == timedelta(0), "event timestamp must be UTC")
    return _text(event.get("type"), "type")


def _message(event: dict[str, Any], model: str) -> Message:
    _require(all(field in event for field in ("content", "usage", "stop_reason")), "incomplete model response")
    response_model = _text(event.get("model", model), "response model")
    try:
        return Message.model_validate({
            "id": f"replay-message-{event['turn']}",
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "stop_reason": event["stop_reason"],
            "content": event["content"],
            "usage": event["usage"],
        }, strict=True)
    except ValueError:
        raise ReplayInputError("invalid recorded model response") from None


def _validate_terminal(event: dict[str, Any]) -> None:
    status = _text(event.get("status"), "status")
    _require(status in _TERMINAL_STATUSES, "unknown terminal status")
    if status == "ok":
        _text(event.get("output"), "output")
        _require("reason" not in event, "successful terminal event has a reason")
        return
    reason = _text(event.get("reason"), "reason")
    _require(
        reason != "replay_mismatch" and bool(_REASON.fullmatch(reason) or _PROVIDER_ERROR.fullmatch(reason)),
        "invalid terminal reason",
    )
    _require("output" not in event, "failed terminal event has output")


def _parse_run(lines: list[str], session_id: str | None) -> ReplayRun:
    _require(bool(lines), "empty run trace")
    start = _object(lines[0])
    _require(_envelope(start, 1) == "run_started", "first event must be run_started")
    run_id = _text(start.get("run_id"), "run_id")
    task = _text(start.get("task"), "task")
    model = _text(start.get("model"), "model")
    legacy = session_id is None and _NEW_RUN_FIELDS.isdisjoint(start)
    workflow = {"id": "orders", "version": 1} if legacy else start.get("workflow")
    if not isinstance(workflow, dict):
        raise ReplayInputError("workflow must be an object")
    workflow_id = _text(workflow.get("id"), "workflow id")
    workflow_version = _number(workflow.get("version"), "workflow version")
    if "mode" in start:
        _require(start["mode"] in ("live", "replay"), "invalid run mode")
    if "source_run_id" in start:
        _text(start["source_run_id"], "source_run_id")

    session_turn = None
    if session_id is None:
        _require(
            "session_id" not in start and "session_turn" not in start,
            "replay a session run using its session directory",
        )
    else:
        _require(start.get("session_id") == session_id, "session_id does not match the manifest")
        session_turn = _number(start.get("session_turn"), "session_turn")

    turn = 0
    pending_call = None
    terminal = False
    for seq, line in enumerate(lines[1:], 2):
        event = _object(line)
        kind = _envelope(event, seq)
        _require(kind in _EVENT_FIELDS, "unknown run event")
        _require(not terminal, "event after run_finished")
        if pending_call is not None:
            _require(kind == "tool_result", "tool intent has no matching result")
        if kind == "run_finished":
            _validate_terminal(event)
            terminal = True
            continue

        event_turn = _number(event.get("turn"), "turn")
        if kind == "llm_response":
            _require(event_turn == turn + 1, "incorrect response turn")
            _require(legacy or "model" in event, "missing response model")
            _message(event, model)
            turn = event_turn
            continue

        _require(event_turn == turn, "tool event outside its response turn")
        call_id = _text(event.get("call_id"), "call_id")
        if kind == "tool_intent":
            _text(event.get("name"), "tool name")
            _require(isinstance(event.get("args"), dict), "tool args must be an object")
            pending_call = call_id
        else:
            _require(call_id == pending_call, "tool result has no matching intent")
            _require(type(event.get("is_error")) is bool, "tool is_error must be a boolean")
            _object(_text(event.get("content"), "tool result content"))
            pending_call = None

    _require(terminal, "trace is missing run_finished")
    return ReplayRun(
        run_id, task, model, workflow_id, workflow_version, tuple(lines[1:]), session_id, session_turn,
    )


def _load_run(path: Path, session_id: str | None = None) -> ReplayRun:
    try:
        with path.open(encoding="utf-8") as stream:
            return _parse_run(list(stream), session_id)
    except (OSError, UnicodeError):
        raise ReplayInputError(f"{path}: could not read run trace") from None
    except ReplayInputError as error:
        raise ReplayInputError(f"{path}: {error}") from None


def load_recording(path: Path) -> ReplayRecording:
    """Validate a run or a sealed session before returning any recorded runs."""
    if not path.is_dir():
        return ReplayRecording((_load_run(path),), is_session=False)

    try:
        manifest = _object((path / "session.json").read_text(encoding="utf-8"))
        _schema_version(manifest)
        session_id = _text(manifest.get("session_id"), "session_id")
        count = _number(manifest.get("prompt_count"), "prompt_count")
        trace_paths = sorted(path.glob("*.jsonl"))
    except (OSError, UnicodeError):
        raise ReplayInputError(f"{path}: could not read session recording") from None
    except ReplayInputError as error:
        raise ReplayInputError(f"{path / 'session.json'}: {error}") from None

    runs = tuple(sorted(
        (_load_run(trace_path, session_id) for trace_path in trace_paths),
        key=lambda run: run.session_turn or 0,
    ))
    _require(
        len(runs) == count and all(run.session_turn == turn for turn, run in enumerate(runs, 1)),
        "session turns must be consecutive and match the manifest count",
    )
    _require(len({run.run_id for run in runs}) == count, "session contains duplicate run IDs")
    identities = {(run.model, run.workflow_id, run.workflow_version) for run in runs}
    _require(len(identities) == 1, "session runs must share a model and workflow version")
    return ReplayRecording(runs, is_session=True)


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    kind = event["type"]
    payload = {key: event[key] for key in _EVENT_FIELDS[kind] if key in event}
    if kind == "tool_result":
        payload["content"] = _object(_text(payload.get("content"), "tool result content"))
    return payload


class ReplayCursor:
    """Supply recorded responses and check the events emitted by a single run."""

    def __init__(self, source: ReplayRun):
        self._source = source
        self._index = 0

    def _next_event(self) -> dict[str, Any]:
        if self._index == len(self._source._events):
            raise ReplayMismatch("Run produced an event or model request after the recorded terminal event.")
        return _object(self._source._events[self._index])

    def create(self, **kwargs: object) -> Message:
        event = self._next_event()
        if event["type"] == "llm_response":
            return _message(event, self._source.model)
        if event["type"] == "run_finished" and event["status"] == "failed":
            reason = event["reason"]
            if reason == "invalid_response" or _PROVIDER_ERROR.fullmatch(reason):
                raise RecordedResponseError(reason)
        raise ReplayMismatch(f"Run requested a model response at recorded event {event['seq']} ({event['type']}).")

    def observe(self, event: dict[str, object]) -> None:
        if event.get("type") == "run_started":
            return
        expected = self._next_event()
        kind = expected["type"]
        if event.get("type") != kind:
            raise ReplayMismatch(f"Run diverged at recorded event {expected['seq']} (expected {kind}).")
        try:
            actual_payload = _payload(event)
            expected_payload = _payload(expected)
            if kind == "llm_response":
                expected_payload.setdefault("model", self._source.model)
            matches = json.dumps(actual_payload, sort_keys=True) == json.dumps(expected_payload, sort_keys=True)
        except (ValueError, TypeError, RecursionError):
            matches = False
        if not matches:
            raise ReplayMismatch(f"Run diverged at recorded event {expected['seq']} ({kind} payload differs).")
        self._index += 1
