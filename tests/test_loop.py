import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from anthropic.types import Message


def response(stop_reason, *blocks):
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "stop_reason": stop_reason, "content": list(blocks),
        "usage": {"input_tokens": 100, "output_tokens": 20},
    })


def tool_call(call_id, value):
    return {"type": "tool_use", "id": call_id, "name": "remember", "input": {"value": value}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        from workflow import ToolOutcome, WorkflowSession

        self.calls = []

        def remember(name, args):
            self.calls.append((name, args.pop("value")))
            return ToolOutcome({"error": "ordinary tool data"}, False)

        self.workflow = WorkflowSession(
            "memory", 1, "Memory", "Remember the supplied values.",
            [{"name": "remember", "input_schema": {"type": "object"}}], remember,
        )
        directory = self.enterContext(tempfile.TemporaryDirectory())
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(directory)
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def test_injected_workflow_drives_requests_and_tool_order(self):
        from loop import run

        requests = []
        replies = iter([
            response("tool_use", tool_call("call_a", "first"), tool_call("call_b", "second")),
            response("end_turn", {"type": "text", "text": "Values recorded."}),
        ])

        def create(**request):
            requests.append({**request, "messages": list(request["messages"])})
            return next(replies)

        answer = run(create, task="Remember these values.", model="claude-haiku-5-5", workflow=self.workflow)

        self.assertEqual(answer, "Values recorded.")
        self.assertEqual(requests[0]["system"], self.workflow.system_prompt)
        self.assertEqual(requests[0]["tools"], self.workflow.tools)
        self.assertEqual(self.calls, [("remember", "first"), ("remember", "second")])
        original_calls = requests[1]["messages"][-2]["content"]
        self.assertEqual([block.input for block in original_calls], [{"value": "first"}, {"value": "second"}])
        results = requests[1]["messages"][-1]["content"]
        self.assertEqual([item["tool_use_id"] for item in results], ["call_a", "call_b"])
        self.assertTrue(all(item["is_error"] is False for item in results))
        events = [json.loads(line) for line in next(Path("runs").glob("*.jsonl")).read_text().splitlines()]
        self.assertEqual(events[0]["workflow"], {"id": "memory", "version": 1})
        self.assertEqual(events[-1]["usage"]["total_tokens"], 240)
        self.assertEqual(events[-1]["cost"]["estimated_usd"], "0.00004")

    def test_request_budget_records_eighth_response_without_running_its_tool(self):
        from loop import RunError, run

        turns = iter(range(1, 9))

        def create(**request):
            turn = next(turns)
            return response("tool_use", tool_call(f"call_{turn}", str(turn)))

        with self.assertRaisesRegex(RunError, "eight-request limit"):
            run(create, task="Keep going.", model="claude-haiku-5-5", workflow=self.workflow)

        self.assertEqual([value for _, value in self.calls], list("1234567"))
        events = [json.loads(line) for line in next(Path("runs").glob("*.jsonl")).read_text().splitlines()]
        self.assertEqual(events[-1]["status"], "budget_exceeded")
        self.assertEqual(events[-1]["usage"]["response_count"], 8)

    def test_invalid_sdk_response_is_recorded_as_a_replayable_request_failure(self):
        from loop import RunError, run
        from replay import RecordedResponseError, ReplayCursor, load_recording

        malformed = Message.construct(
            id="msg_invalid", type="message", role="assistant", model="claude-haiku-5-5",
            stop_reason="end_turn", content=[{"type": "text", "text": 42}],
            usage={"input_tokens": 100, "output_tokens": 20},
        )
        with self.assertRaisesRegex(RunError, "invalid response"):
            run(lambda **_: malformed, task="Enter a value.", model="claude-haiku-5-5", workflow=self.workflow)

        trace, = Path("runs").glob("*.jsonl")
        cursor = ReplayCursor(load_recording(trace).runs[0])
        with self.assertRaises(RecordedResponseError) as replayed:
            cursor.create()
        self.assertEqual(replayed.exception.reason, "invalid_response")

    def test_blank_resolved_model_is_rejected_before_recording_a_response(self):
        from loop import RunError, run
        from replay import load_recording

        malformed = response("end_turn", {"type": "text", "text": "Done."}).model_copy(update={"model": " "})
        with self.assertRaisesRegex(RunError, "invalid response"):
            run(lambda **_: malformed, task="Enter a value.", model="claude-haiku-5-5", workflow=self.workflow)

        trace, = Path("runs").glob("*.jsonl")
        self.assertEqual(len(load_recording(trace).runs), 1)
