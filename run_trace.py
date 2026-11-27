"""Write per-prompt traces and seal completed interactive sessions."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from anthropic.types import Message

from usage import UsageTotals, format_usage

SCHEMA_VERSION = 1
RUNS = Path("runs")


@dataclass(frozen=True)
class SessionTurn:
    session_id: str
    number: int

    @property
    def directory(self) -> Path:
        return RUNS / self.session_id


def seal_session(turn: SessionTurn) -> None:
    with (turn.directory / "session.json").open("x", encoding="utf-8") as manifest:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "session_id": turn.session_id,
            "prompt_count": turn.number,
        }, manifest)
        manifest.write("\n")


class RunTrace:
    def __init__(
        self,
        stream: TextIO,
        *,
        observe: Callable[[dict[str, object]], None] | None,
        replay: bool,
    ):
        self._stream = stream
        self._observe = observe
        self._replay = replay
        self._seq = 0
        self._usage = UsageTotals()

    def record(self, event_type: str, *, verify: bool = True, **fields: object) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "seq": self._seq + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **fields,
        }
        if verify and self._observe is not None:
            self._observe(event)
        self._stream.write(json.dumps(event) + "\n")
        self._stream.flush()
        self._seq += 1

    def response(self, turn: int, response: Message) -> None:
        usage = response.usage.model_dump(mode="json", exclude_unset=True)
        self.record(
            "llm_response",
            turn=turn,
            model=response.model,
            stop_reason=response.stop_reason,
            content=[block.model_dump(mode="json", exclude_unset=True) for block in response.content],
            usage=usage,
        )
        self._usage.add(response.model, usage)

    def finish(self, status: str, *, verify: bool = True, **fields: object) -> None:
        summary = self._usage.summary()
        self.record("run_finished", verify=verify, status=status, **fields, **summary)
        print(format_usage(summary, replay=self._replay), file=sys.stderr)
