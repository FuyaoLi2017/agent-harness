"""The workflow interface consumed by the agent loop."""

from collections.abc import Callable
from dataclasses import dataclass

from anthropic.types import ToolParam


@dataclass(frozen=True)
class ToolOutcome:
    content: dict[str, object]
    is_error: bool


@dataclass(frozen=True)
class WorkflowSession:
    id: str
    version: int
    display_name: str
    system_prompt: str
    tools: list[ToolParam]
    execute_tool: Callable[[str, dict[str, object]], ToolOutcome]
