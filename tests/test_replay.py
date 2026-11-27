"""Tests for offline replay input and event matching."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from anthropic.types import Message

from replay import (
    RecordedResponseError,
    ReplayCursor,
    ReplayInputError,
    ReplayMismatch,
    load_recording,
)

MODEL = "claude-haiku-5-5"
ORDERS_WORKFLOW = {"id": "orders", "version": 1}


def started(run_id="run-1", **fields):
    return {
        "type": "run_started",
        "run_id": run_id,
        "task": "Find my order.",
        "model": MODEL,
        **fields,
    }


def response(*, turn=1, stop_reason="end_turn", content=None, **fields):
    return {
        "type": "llm_response",
        "turn": turn,
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": "Your order is pending."}]
        if content is None else content,
        "usage": {"input_tokens": 5, "output_tokens": 4},
        **fields,
    }


def finished():
    return {"type": "run_finished", "status": "ok", "output": "Your order is pending."}


def tool_events(call_id, name):
    args = {"order_id": "ORD-1001"}
    return (
        {"type": "tool_use", "id": call_id, "name": name, "input": args},
        {"type": "tool_intent", "turn": 1, "call_id": call_id, "name": name, "args": args},
        {
            "type": "tool_result",
            "turn": 1,
            "call_id": call_id,
            "is_error": False,
            "content": '{"order_id": "ORD-1001", "status": "pending"}',
        },
    )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.trace = self.root / "recording.jsonl"

    def write_trace(self, events, path=None):
        (path or self.trace).write_text(
            "".join(
                json.dumps({
                    "schema_version": 1,
                    "seq": seq,
                    "ts": "2026-11-27T12:00:00Z",
                    **event,
                }) + "\n"
                for seq, event in enumerate(events, 1)
            ),
            encoding="utf-8",
        )

    def write_manifest(self, count):
        (self.root / "session.json").write_text(
            json.dumps({"schema_version": 1, "session_id": "session-1", "prompt_count": count}),
            encoding="utf-8",
        )

    def test_legacy_run_replays_with_original_model_and_workflow(self):
        start = started()
        recorded_response = response()
        terminal = finished()
        self.write_trace([start, recorded_response, terminal])

        recording = load_recording(self.trace)
        (run,) = recording.runs
        self.assertFalse(recording.is_session)
        self.assertEqual((run.run_id, run.task, run.model), ("run-1", "Find my order.", "claude-haiku-5-5"))
        self.assertEqual((run.workflow_id, run.workflow_version), ("orders", 1))

        cursor = ReplayCursor(run)
        cursor.observe(start)
        message = cursor.create(model=run.model)
        self.assertIsInstance(message, Message)
        self.assertEqual(message.model, "claude-haiku-5-5")
        self.assertEqual(message.content[0].text, "Your order is pending.")
        cursor.observe({**recorded_response, "model": message.model})
        cursor.observe({**terminal, "usage": {}, "cost": None})
        with self.assertRaises(ReplayMismatch):
            cursor.create()

    def test_session_order_comes_from_turn_metadata(self):
        self.write_manifest(2)
        for turn, filename in [(2, "a.jsonl"), (1, "z.jsonl")]:
            self.write_trace([
                started(f"run-{turn}", session_id="session-1", session_turn=turn,
                        workflow={"id": "triage", "version": 2}),
                response(model=MODEL),
                finished(),
            ], self.root / filename)

        recording = load_recording(self.root)
        self.assertTrue(recording.is_session)
        self.assertEqual([run.run_id for run in recording.runs], ["run-1", "run-2"])
        self.assertEqual(recording.runs[0].workflow_id, "triage")

    def test_session_members_require_a_matching_complete_manifest(self):
        self.write_manifest(2)
        self.write_trace([
            started(session_id="session-1", session_turn=1, workflow=ORDERS_WORKFLOW),
            response(model=MODEL),
            finished(),
        ])
        for path in [self.trace, self.root]:
            with self.subTest(path=path), self.assertRaises(ReplayInputError):
                load_recording(path)

    def test_session_runs_share_model_workflow_and_session_id(self):
        self.write_manifest(2)
        self.write_trace([
            started(session_id="session-1", session_turn=1, workflow=ORDERS_WORKFLOW),
            response(model=MODEL),
            finished(),
        ])
        for metadata in [
            {"session_id": "session-2"},
            {"session_turn": 1},
            {"run_id": "run-1"},
            {"model": "claude-sonnet-5-5"},
            {"workflow": {"id": "orders", "version": 2}},
        ]:
            with self.subTest(metadata=metadata), self.assertRaises(ReplayInputError):
                second = {
                    **started(
                        "run-2", session_id="session-1", session_turn=2, workflow=ORDERS_WORKFLOW,
                    ),
                    **metadata,
                }
                self.write_trace([second, response(model=second["model"]), finished()], self.root / "second.jsonl")
                load_recording(self.root)

    def test_new_standalone_trace_requires_explicit_workflow_and_response_model(self):
        for start, reply, missing in [
            (started(mode="live"), response(model=MODEL), "workflow"),
            (started(source_run_id="original"), response(model=MODEL), "workflow"),
            (started(workflow=ORDERS_WORKFLOW), response(), "response model"),
        ]:
            with self.subTest(missing=missing, start=start), self.assertRaisesRegex(ReplayInputError, missing):
                self.write_trace([start, reply, finished()])
                load_recording(self.trace)

    def test_session_trace_requires_explicit_workflow_and_response_model(self):
        self.write_manifest(1)
        member = started(session_id="session-1", session_turn=1)
        for start, reply, missing in [
            (member, response(model=MODEL), "workflow"),
            ({**member, "workflow": ORDERS_WORKFLOW}, response(), "response model"),
        ]:
            with self.subTest(missing=missing), self.assertRaisesRegex(ReplayInputError, missing):
                self.write_trace([start, reply, finished()])
                load_recording(self.root)

    def test_rejects_invalid_envelopes_and_sdk_responses(self):
        for bad_response in [
            response(schema_version=True),
            response(seq=3),
            response(ts="yesterday"),
            response(turn=True),
            response(turn=2),
            response(stop_reason="bogus"),
            response(content=[{"type": "text", "text": 2}]),
            response(usage={"input_tokens": "5", "output_tokens": 4}),
        ]:
            with self.subTest(event=bad_response), self.assertRaises(ReplayInputError):
                self.write_trace([started(), bad_response, finished()])
                load_recording(self.trace)

    def test_rejects_incomplete_traces_and_malformed_tool_pairs(self):
        block, intent, result = tool_events("call-1", "get_order")
        model = response(stop_reason="tool_use", content=[block])
        for events in [
            [started(), response()],
            [started(), response(), finished(), finished()],
            [started(), model, intent, finished()],
            [started(), model, result, finished()],
            [started(), model, intent, {**result, "content": "not JSON"}, finished()],
            [started(), model, intent, {**result, "call_id": "call-2"}, finished()],
            [started(), model, {**intent, "turn": 2}, result, finished()],
        ]:
            with self.subTest(events=events), self.assertRaises(ReplayInputError):
                self.write_trace(events)
                load_recording(self.trace)

    def test_cursor_preserves_tool_order_and_compares_result_json(self):
        block1, intent1, result1 = tool_events("call-1", "get_order")
        block2, intent2, result2 = tool_events("call-2", "cancel_order")
        first = response(stop_reason="tool_use", content=[block1, block2], model="recorded-model")
        second = response(turn=2)
        self.write_trace([started(), first, intent1, result1, intent2, result2, second, finished()])
        cursor = ReplayCursor(load_recording(self.trace).runs[0])
        message = cursor.create()
        self.assertEqual(message.model, "recorded-model")
        self.assertEqual([block.id for block in message.content], ["call-1", "call-2"])
        cursor.observe(first)
        with self.assertRaises(ReplayMismatch):
            cursor.observe(intent2)
        cursor.observe(intent1)
        with self.assertRaises(ReplayMismatch):
            cursor.observe({**result1, "content": '{"order_id": "ORD-1001", "status": "cancelled"}'})
        cursor.observe({**result1, "content": '{"status":"pending", "order_id":"ORD-1001"}'})
        cursor.observe(intent2)
        cursor.observe(result2)
        cursor.observe({**second, "model": "claude-haiku-5-5"})
        with self.assertRaises(ReplayMismatch):
            cursor.observe({**finished(), "output": "Different."})
        cursor.observe(finished())
        with self.assertRaises(ReplayMismatch):
            cursor.observe(finished())

    def test_cursor_reports_missing_response_or_tool_events_as_mismatch(self):
        block, intent, result = tool_events("call-1", "get_order")
        first = response(stop_reason="tool_use", content=[block])
        second = response(turn=2)
        self.write_trace([started(), first, intent, result, second, finished()])
        cursor = ReplayCursor(load_recording(self.trace).runs[0])
        cursor.observe({**first, "model": "claude-haiku-5-5"})
        with self.assertRaises(ReplayMismatch):
            cursor.create()
        with self.assertRaises(ReplayMismatch):
            cursor.observe(finished())

    def test_runtime_model_field_must_be_present_even_for_legacy_recordings(self):
        recorded = response()
        self.write_trace([started(), recorded, finished()])
        cursor = ReplayCursor(load_recording(self.trace).runs[0])
        with self.assertRaises(ReplayMismatch):
            cursor.observe(recorded)

    def test_recreates_terminal_provider_failure_after_completed_tool_turn(self):
        block, intent, result = tool_events("call-1", "get_order")
        first = response(stop_reason="tool_use", content=[block])
        for reason in ["invalid_response", "provider_error/APIConnectionError"]:
            with self.subTest(reason=reason):
                terminal = {"type": "run_finished", "status": "failed", "reason": reason}
                self.write_trace([started(), first, intent, result, terminal])
                cursor = ReplayCursor(load_recording(self.trace).runs[0])
                cursor.create()
                for event in [{**first, "model": "claude-haiku-5-5"}, intent, result]:
                    cursor.observe(event)
                with self.assertRaises(RecordedResponseError) as failure:
                    cursor.create()
                self.assertEqual(failure.exception.reason, reason)
                cursor.observe(terminal)
