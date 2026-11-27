"""The installed workflow interface and order-tool compatibility."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import workflows
from workflow import ToolOutcome
from workflows import WorkflowError, open_workflow


class WorkflowTests(unittest.TestCase):
    def test_orders_is_the_default_workflow(self) -> None:
        session = open_workflow()

        self.assertEqual((session.id, session.version), ("orders", 1))
        self.assertEqual(session.display_name, "Synthetic order support")
        self.assertEqual(open_workflow("orders", 1).system_prompt, session.system_prompt)
        self.assertEqual(
            [tool["name"] for tool in session.tools],
            ["get_order", "cancel_order", "issue_refund"],
        )

    def test_selection_requires_an_exact_installed_id_and_version(self) -> None:
        for workflow_id, version in [("Orders", None), ("../orders", None), ("orders", 2)]:
            with self.subTest(workflow_id=workflow_id, version=version):
                with self.assertRaises(WorkflowError):
                    open_workflow(workflow_id, version)

    def test_sessions_isolate_order_changes(self) -> None:
        session = open_workflow()
        fresh_session = open_workflow()
        arguments = {"order_id": "ORD-1002"}

        canceled = ToolOutcome({"order_id": "ORD-1002", "status": "cancelled"}, False)
        self.assertEqual(session.execute_tool("cancel_order", arguments), canceled)
        self.assertEqual(session.execute_tool("cancel_order", arguments), canceled)
        self.assertEqual(session.execute_tool("get_order", arguments).content["status"], "cancelled")
        self.assertEqual(fresh_session.execute_tool("get_order", arguments).content["status"], "pending")

    def test_refund_decisions_keep_the_existing_payloads(self) -> None:
        session = open_workflow()
        arguments: dict[str, object] = {"order_id": "ORD-1001", "item_ids": ["ITEM-001"]}

        self.assertEqual(
            session.execute_tool("issue_refund", arguments),
            ToolOutcome({
                "order_id": "ORD-1001",
                "refunded_item_ids": ["ITEM-001"],
                "amount_cents": 2500,
                "refund_destination": "original_payment_method",
                "simulated": True,
            }, False),
        )
        self.assertEqual(
            session.execute_tool("issue_refund", arguments),
            ToolOutcome({
                "error": {"code": "item_already_refunded", "message": "A selected item was already refunded."},
            }, True),
        )

    def test_dispatch_returns_compatible_errors(self) -> None:
        session = open_workflow()

        self.assertEqual(
            session.execute_tool("load_orders", {}),
            ToolOutcome({"error": {"code": "unknown_tool", "message": "Tool is unavailable."}}, True),
        )
        self.assertEqual(
            session.execute_tool("issue_refund", {"order_id": "ORD-1001"}),
            ToolOutcome({
                "error": {"code": "invalid_arguments", "message": "issue_refund accepts exactly item_ids and order_id."},
            }, True),
        )

    def test_workflow_rejects_mismatched_and_duplicate_tool_names(self) -> None:
        path, open_handlers = workflows._REGISTRY["orders"][1]
        with path.open(encoding="utf-8") as source:
            definition = json.load(source)

        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "workflow.json"
            with patch.dict(workflows._REGISTRY, {"orders": {1: (candidate, open_handlers)}}):
                for tools in [
                    [{**definition["tools"][0], "name": "missing_handler"}],
                    [*definition["tools"], definition["tools"][0]],
                ]:
                    with self.subTest(tools=[tool["name"] for tool in tools]):
                        candidate.write_text(json.dumps({**definition, "tools": tools}), encoding="utf-8")
                        with self.assertRaises(WorkflowError):
                            open_workflow()

    def test_workflow_dispatches_only_tools_advertised_in_metadata(self) -> None:
        path, open_handlers = workflows._REGISTRY["orders"][1]
        definition = json.loads(path.read_text(encoding="utf-8"))
        definition["tools"] = definition["tools"][:1]

        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "workflow.json"
            candidate.write_text(json.dumps(definition), encoding="utf-8")
            with patch.dict(workflows._REGISTRY, {"orders": {1: (candidate, open_handlers)}}):
                session = open_workflow()

        args = {"order_id": "ORD-1002"}
        self.assertEqual(session.execute_tool("cancel_order", args).content["error"]["code"], "unknown_tool")
        self.assertEqual(session.execute_tool("get_order", args).content["status"], "pending")
