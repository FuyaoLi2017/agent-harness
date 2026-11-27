"""Open sessions from the installed workflow registry."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from anthropic.types import ToolParam

from workflow import ToolOutcome, WorkflowSession

from .orders.tools import open_handlers as open_order_handlers

DEFAULT_WORKFLOW = "orders"

type _Handler = Callable[[dict[str, object]], ToolOutcome]
type _HandlerFactory = Callable[[], dict[str, _Handler]]

_REGISTRY: dict[str, dict[int, tuple[Path, _HandlerFactory]]] = {
    "orders": {
        1: (Path(__file__).parent / "orders" / "workflow.json", open_order_handlers),
    },
}


class WorkflowError(ValueError):
    """The requested workflow version could not be opened."""


def open_workflow(
    workflow_id: str = DEFAULT_WORKFLOW, version: int | None = None
) -> WorkflowSession:
    """Open an exact installed workflow version or the latest when omitted."""
    versions = _REGISTRY.get(workflow_id)
    if versions is None:
        raise WorkflowError(f"Unknown workflow {workflow_id!r}.")

    version = max(versions) if version is None else version
    registration = versions.get(version) if type(version) is int else None
    label = f"workflow {workflow_id!r} version {version}"
    if registration is None:
        raise WorkflowError(f"Unknown {label}.")

    path, open_handlers = registration
    try:
        definition = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WorkflowError(f"Could not read metadata for {label}: {error}") from error

    if (
        not isinstance(definition, dict)
        or definition.get("id") != workflow_id
        or type(definition.get("version")) is not int
        or definition["version"] != version
        or not _nonempty_string(definition.get("display_name"))
        or not _nonempty_string(definition.get("system_prompt"))
    ):
        raise WorkflowError(f"Invalid metadata for {label}.")

    tools = definition.get("tools")
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict)
        or not _nonempty_string(tool.get("name"))
        or not isinstance(tool.get("input_schema"), dict)
        for tool in tools
    ):
        raise WorkflowError(f"Invalid tool definitions for {label}.")

    tool_names = {tool["name"] for tool in tools}
    if len(tool_names) != len(tools):
        raise WorkflowError(f"Duplicate tool names in {label}.")

    try:
        handlers = open_handlers()
    except (OSError, ValueError) as error:
        raise WorkflowError(f"Could not open {label}: {error}") from error
    if not tool_names <= handlers.keys():
        raise WorkflowError(f"Missing handlers for advertised tools in {label}.")

    def execute_tool(name: str, arguments: dict[str, object]) -> ToolOutcome:
        if name not in tool_names:
            return ToolOutcome({"error": {"code": "unknown_tool", "message": "Tool is unavailable."}}, True)
        return handlers[name](arguments)

    return WorkflowSession(
        id=workflow_id,
        version=version,
        display_name=definition["display_name"],
        system_prompt=definition["system_prompt"],
        tools=cast(list[ToolParam], tools),
        execute_tool=execute_tool,
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
