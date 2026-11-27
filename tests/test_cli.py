import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anthropic.types import Message

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
STANDALONE = FIXTURES / "orders-single.jsonl"
SESSION = FIXTURES / "orders-session"
REFUND = "I issued a simulated $25.00 refund for the mug."
DUPLICATE = "The mug has already been refunded, so no additional refund was issued."


def read_events(trace):
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def write_events(trace, events):
    trace.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = SimpleNamespace(create=self.create)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def create(self, **request):
        self.requests.append(list(request["messages"]))
        event = next(self.responses)
        return Message.model_validate({
            "type": "message", "role": "assistant", "id": "msg_fixture",
            **{key: event[key] for key in ("model", "content", "stop_reason", "usage")},
        })


class CLITests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def cli(self, *args, input_text=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith("ANTHROPIC_")}
        return subprocess.run(
            [sys.executable, str(ROOT / "loop.py"), *map(str, args)],
            cwd=self.cwd, env=env, input=input_text, capture_output=True, text=True, timeout=10,
        )

    def cli_in_process(self, *args):
        from cli import main

        old_cwd = Path.cwd()
        out, err = io.StringIO(), io.StringIO()
        try:
            os.chdir(self.cwd)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main(args)
        finally:
            os.chdir(old_cwd)
        return status, out.getvalue(), err.getvalue()

    def test_replay_legacy_standalone_without_credentials(self):
        original = STANDALONE.read_bytes()
        result = self.cli("--replay", STANDALONE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, REFUND + "\n")
        new_trace, = (self.cwd / "runs").glob("*.jsonl")
        events = read_events(new_trace)
        self.assertEqual(events[0]["source_run_id"], read_events(STANDALONE)[0]["run_id"])
        self.assertEqual(events[0]["workflow"], {"id": "orders", "version": 1})
        self.assertEqual(events[-1]["usage"]["response_count"], 3)
        self.assertIn("Estimated cost for original run", result.stderr)
        self.assertEqual(STANDALONE.read_bytes(), original)

    def test_replay_matches_both_prompts_without_constructing_a_provider_client(self):
        with patch("cli.Anthropic", side_effect=AssertionError("replay called Anthropic")):
            status, out, err = self.cli_in_process("--replay", str(SESSION))
        self.assertEqual(status, 0, err)
        self.assertEqual(out.splitlines(), [REFUND, DUPLICATE])
        manifest, = (self.cwd / "runs").glob("*/session.json")
        self.assertEqual(json.loads(manifest.read_text())["prompt_count"], 2)
        replayed = self.cli("--replay", manifest.parent)
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertEqual(replayed.stdout, out)

    def test_live_session_seals_on_eof_and_can_be_replayed(self):
        recorded = [
            event for trace in sorted(SESSION.glob("*.jsonl")) for event in read_events(trace)
            if event["type"] == "llm_response"
        ]
        provider = ScriptedProvider(recorded)

        with (
            patch("cli.Anthropic", return_value=provider),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture"}),
            patch("sys.stdin", io.StringIO("Please refund the mug from order ORD-1001.\nPlease refund it again.\n")),
        ):
            status, out, err = self.cli_in_process("--workflow", "orders")
        self.assertEqual(status, 0, err)
        self.assertIn(DUPLICATE, out)
        self.assertTrue(out.startswith("Synthetic order support."))
        follow_up = provider.requests[3]
        self.assertEqual(follow_up[-2]["content"][0].text, REFUND)
        self.assertEqual(follow_up[-1], {"role": "user", "content": "Please refund it again."})
        manifest, = (self.cwd / "runs").glob("*/session.json")
        replay = self.cli("--replay", manifest.parent)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(replay.stdout.splitlines(), [REFUND, DUPLICATE])

    def test_trace_write_failure_leaves_the_session_unsealed(self):
        live_response = next(event for event in read_events(STANDALONE)
                             if event.get("stop_reason") == "end_turn")
        trace_opens = 0
        open_path = Path.open

        def fail_second_trace(path, mode="r", *args, **kwargs):
            nonlocal trace_opens
            if mode == "x" and path.suffix == ".jsonl":
                trace_opens += 1
                if trace_opens == 2:
                    raise OSError("could not write second trace")
            return open_path(path, mode, *args, **kwargs)

        with (
            patch("cli.Anthropic", return_value=ScriptedProvider([{**live_response, "model": "claude-haiku-5-5"}])),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture"}),
            patch("sys.stdin", io.StringIO("first prompt\nsecond prompt\nexit\n")),
            patch.object(Path, "open", fail_second_trace),
        ):
            status, _, err = self.cli_in_process()

        self.assertEqual(status, 1, err)
        self.assertIn("could not write second trace", err)
        self.assertEqual(len(list((self.cwd / "runs").glob("*/*.jsonl"))), 1)
        self.assertEqual(list((self.cwd / "runs").glob("*/session.json")), [])

    def test_session_preserves_tool_effects_before_provider_failure(self):
        source = shutil.copytree(SESSION, self.cwd / "source")
        first, _ = sorted(source.glob("*.jsonl"))
        events = read_events(first)
        terminal = events.pop()
        events.pop()  # Replace the final model response with a provider failure.
        terminal.pop("output")
        terminal.update(seq=len(events) + 1, status="failed", reason="provider_error/APIConnectionError")
        write_events(first, [*events, terminal])

        result = self.cli("--replay", source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, DUPLICATE + "\n")
        traces = [read_events(trace) for trace in (self.cwd / "runs").glob("*/*.jsonl")]
        failed = next(events for events in traces if events[0]["session_turn"] == 1)
        self.assertEqual(failed[-1]["reason"], "provider_error/APIConnectionError")
        self.assertEqual(failed[-1]["usage"]["response_count"], 2)

    def test_missing_final_prompt_fails_before_replay(self):
        source = shutil.copytree(SESSION, self.cwd / "source")
        sorted(source.glob("*.jsonl"))[-1].unlink()
        result = self.cli("--replay", source)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.cwd / "runs").exists())

    def test_unavailable_workflow_version_fails_before_replay(self):
        trace = self.cwd / "source.jsonl"
        events = read_events(STANDALONE)
        events[0]["workflow"] = {"id": "orders", "version": 999}
        for event in events:
            if event["type"] == "llm_response":
                event["model"] = events[0]["model"]
        write_events(trace, events)
        result = self.cli("--replay", trace)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("999", result.stderr)
        self.assertFalse((self.cwd / "runs").exists())

    def test_tool_result_drift_fails_before_printing_an_answer(self):
        trace = self.cwd / "source.jsonl"
        events = read_events(STANDALONE)
        refund = [event for event in events if event["type"] == "tool_result"][-1]
        payload = json.loads(refund["content"])
        refund["content"] = json.dumps({**payload, "amount_cents": 2600})
        write_events(trace, events)
        result = self.cli("--replay", trace)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        output_trace, = (self.cwd / "runs").glob("*.jsonl")
        self.assertEqual(read_events(output_trace)[-1]["reason"], "replay_mismatch")

    def test_replay_rejects_live_cli_arguments(self):
        for conflicts in [("task",), ("--model", "claude-haiku-5-5"), ("--workflow", "orders")]:
            with self.subTest(conflicts=conflicts):
                self.assertEqual(self.cli("--replay", STANDALONE, *conflicts).returncode, 2)
