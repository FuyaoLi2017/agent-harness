"""Accumulate recorded token usage and estimate standard Claude API charges."""

from decimal import Decimal
from typing import NamedTuple, cast


class _TokenRates(NamedTuple):
    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal


# Standard Claude API rates in USD per million tokens.
# https://platform.claude.com/docs/en/about-claude/pricing
_PRICING_AS_OF = "2026-11-27"
_RATES = {
    "claude-haiku-5-5": _TokenRates(*map(Decimal, ("0.10", "0.50", "0.125", "0.20", "0.01"))),
    "claude-opus-5-5": _TokenRates(*map(Decimal, ("4", "20", "5", "8", "0.20"))),
}
_HAIKU_LONG_RATES = _TokenRates(*map(Decimal, ("0.50", "2.50", "0.625", "1", "0.05")))
_STANDARD_BILLING = {"inference_geo": "global", "service_tier": "standard", "speed": "standard"}


def _token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _cache_writes(usage: dict[str, object]) -> tuple[int, tuple[int, int] | None]:
    raw_total = usage.get("cache_creation_input_tokens")
    reported_total = _token_count(raw_total)
    details = usage.get("cache_creation")
    if details is None and (raw_total is None or reported_total == 0):
        return 0, (0, 0)
    if not isinstance(details, dict):
        return reported_total or 0, None

    write_5m = _token_count(details.get("ephemeral_5m_input_tokens"))
    write_1h = _token_count(details.get("ephemeral_1h_input_tokens"))
    known_writes = (write_5m or 0) + (write_1h or 0)
    total = known_writes if reported_total is None else reported_total
    if (
        write_5m is None
        or write_1h is None
        or total != known_writes
        or (raw_total is not None and reported_total is None)
    ):
        return total, None
    return total, (write_5m, write_1h)


def _standard_billing(usage: dict[str, object]) -> bool:
    if any(
        usage.get(field) not in (None, expected)
        for field, expected in _STANDARD_BILLING.items()
    ):
        return False
    server_tools = usage.get("server_tool_use")
    if server_tools is None:
        return True
    if not isinstance(server_tools, dict):
        return False
    for tool, raw_count in server_tools.items():
        count = _token_count(raw_count)
        if count is None or (count > 0 and tool != "web_fetch_requests"):
            return False
    return True


class UsageTotals:
    """Track one run, retaining token counts even when pricing is unavailable."""

    def __init__(self) -> None:
        self._response_count = 0
        self._input_tokens = 0
        self._cache_creation_input_tokens = 0
        self._cache_read_input_tokens = 0
        self._output_tokens = 0
        self._estimated_usd: Decimal | None = Decimal(0)

    def add(self, model: str, usage: dict[str, object]) -> None:
        input_tokens = _token_count(usage.get("input_tokens"))
        output_tokens = _token_count(usage.get("output_tokens"))
        raw_cache_read = usage.get("cache_read_input_tokens")
        cache_read = 0 if raw_cache_read is None else _token_count(raw_cache_read)
        cache_writes, cache_write_split = _cache_writes(usage)

        self._response_count += 1
        self._input_tokens += input_tokens or 0
        self._cache_creation_input_tokens += cache_writes
        self._cache_read_input_tokens += cache_read or 0
        self._output_tokens += output_tokens or 0

        if self._estimated_usd is None:
            return
        rates = _RATES.get(model)
        if (
            rates is None
            or input_tokens is None
            or output_tokens is None
            or cache_read is None
            or cache_write_split is None
            or not _standard_billing(usage)
        ):
            self._estimated_usd = None
            return

        if model == "claude-haiku-5-5" and input_tokens + cache_writes + cache_read > 100_000:
            rates = _HAIKU_LONG_RATES
        cache_write_5m, cache_write_1h = cache_write_split
        self._estimated_usd += (
            input_tokens * rates.input
            + output_tokens * rates.output
            + cache_write_5m * rates.cache_write_5m
            + cache_write_1h * rates.cache_write_1h
            + cache_read * rates.cache_read
        ) / 1_000_000

    def summary(self) -> dict[str, object]:
        total_tokens = (
            self._input_tokens
            + self._cache_creation_input_tokens
            + self._cache_read_input_tokens
            + self._output_tokens
        )
        estimate = None
        if self._response_count and self._estimated_usd is not None:
            estimate = format(self._estimated_usd.normalize(), "f")
        return {
            "usage": {
                "response_count": self._response_count,
                "input_tokens": self._input_tokens,
                "cache_creation_input_tokens": self._cache_creation_input_tokens,
                "cache_read_input_tokens": self._cache_read_input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": total_tokens,
            },
            "cost": {"estimated_usd": estimate, "pricing_as_of": _PRICING_AS_OF},
        }


def format_usage(summary: dict[str, object], *, replay: bool) -> str:
    usage = cast(dict[str, int], summary["usage"])
    cost = cast(dict[str, str | None], summary["cost"])
    count = usage["response_count"]
    responses = "response" if count == 1 else "responses"
    cost_label = "Estimated cost for original run" if replay else "Estimated cost"
    estimate = cost["estimated_usd"]
    price = f"${estimate} USD" if estimate is not None else "unavailable"
    return (
        f"Usage — {count:,} {responses}, {usage['input_tokens']:,} input, "
        f"{usage['cache_creation_input_tokens']:,} cache write, "
        f"{usage['cache_read_input_tokens']:,} cache read, "
        f"{usage['output_tokens']:,} output, {usage['total_tokens']:,} total tokens. "
        f"{cost_label} {price} (prices as of {cost['pricing_as_of']})."
    )
