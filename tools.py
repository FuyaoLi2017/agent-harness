"""Tools for the synthetic order-support example."""

import json
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict, cast

from anthropic.types import ToolParam


class OrderItem(TypedDict):
    item_id: str
    name: str
    price_cents: int


class Order(TypedDict):
    order_id: str
    customer_id: str
    status: Literal["pending", "shipped", "delivered", "cancelled"]
    delivered_on: str | None
    payment_ref: str
    items: list[OrderItem]
    refunded_item_ids: list[str]


class OrderFixture(TypedDict):
    customer_id: str
    as_of: str
    orders: list[Order]


GET_ORDER_TOOL: ToolParam = {
    "name": "get_order",
    "description": "Look up the current customer's order by ID, including its status and items.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID, such as ORD-1001.",
                "minLength": 1,
            },
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}

CANCEL_ORDER_TOOL: ToolParam = {
    "name": "cancel_order",
    "description": "Cancel the current customer's pending order by ID.",
    "input_schema": GET_ORDER_TOOL["input_schema"],
}

ISSUE_REFUND_TOOL: ToolParam = {
    "name": "issue_refund",
    "description": "Simulate refunding eligible items to the order's original payment method.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1},
            "item_ids": {
                "type": "array",
                "description": "Distinct item IDs from the delivered order.",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        "required": ["order_id", "item_ids"],
        "additionalProperties": False,
    },
}

TOOLS: list[ToolParam] = [GET_ORDER_TOOL, CANCEL_ORDER_TOOL, ISSUE_REFUND_TOOL]


def load_orders() -> OrderFixture:
    """Load fresh synthetic order data for one run."""
    fixture_path = Path(__file__).with_name("orders.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return cast(OrderFixture, json.load(fixture_file))


def _error(code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message}}


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_order(fixture: OrderFixture, order_id: str) -> Order | None:
    for order in fixture["orders"]:
        if order["order_id"] == order_id and order["customer_id"] == fixture["customer_id"]:
            return order
    return None


def get_order(fixture: OrderFixture, order_id: str) -> dict[str, object]:
    """Return the order details visible to the fixture's customer."""
    if not _valid_id(order_id):
        return _error("invalid_order_id", "Order ID must be a nonempty string.")

    order = _find_order(fixture, order_id)
    if order is None:
        return _error("order_not_found", "Order was not found.")
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "delivered_on": order["delivered_on"],
        "items": [item.copy() for item in order["items"]],
        "refunded_item_ids": order["refunded_item_ids"].copy(),
    }


def cancel_order(fixture: OrderFixture, order_id: str) -> dict[str, object]:
    """Cancel a pending order in the current run's fixture."""
    if not _valid_id(order_id):
        return _error("invalid_order_id", "Order ID must be a nonempty string.")

    order = _find_order(fixture, order_id)
    if order is None:
        return _error("order_not_found", "Order was not found.")
    if order["status"] not in ("pending", "cancelled"):
        return _error("order_not_cancellable", "Only pending orders can be cancelled.")

    order["status"] = "cancelled"
    return {"order_id": order_id, "status": order["status"]}


def issue_refund(
    fixture: OrderFixture, order_id: str, item_ids: list[str]
) -> dict[str, object]:
    """Simulate an eligible item refund in the current run's fixture."""
    if not _valid_id(order_id):
        return _error("invalid_order_id", "Order ID must be a nonempty string.")
    if (
        not isinstance(item_ids, list)
        or not item_ids
        or any(not _valid_id(item_id) for item_id in item_ids)
        or len(set(item_ids)) != len(item_ids)
    ):
        return _error("invalid_item_ids", "Item IDs must be a nonempty list of distinct IDs.")

    order = _find_order(fixture, order_id)
    if order is None:
        return _error("order_not_found", "Order was not found.")

    delivered_on = order["delivered_on"]
    if order["status"] != "delivered" or delivered_on is None:
        return _error("order_not_delivered", "Only delivered orders are eligible for refunds.")
    days_since_delivery = (
        date.fromisoformat(fixture["as_of"]) - date.fromisoformat(delivered_on)
    ).days
    if not 0 <= days_since_delivery <= 30:
        return _error("refund_outside_window", "Refunds are available through 30 days after delivery.")

    prices = {item["item_id"]: item["price_cents"] for item in order["items"]}
    if any(item_id not in prices for item_id in item_ids):
        return _error("item_not_found", "Each selected item must belong to the order.")
    if any(item_id in order["refunded_item_ids"] for item_id in item_ids):
        return _error("item_already_refunded", "A selected item was already refunded.")
    amount_cents = sum(prices[item_id] for item_id in item_ids)
    if amount_cents > 50_000:
        return _error("approval_required", "Refunds above 500 USD require approval.")

    order["refunded_item_ids"].extend(item_ids)
    return {
        "order_id": order_id,
        "refunded_item_ids": item_ids.copy(),
        "amount_cents": amount_cents,
        "refund_destination": "original_payment_method",
        "simulated": True,
    }


def execute_tool(
    orders: OrderFixture, name: str, arguments: object
) -> dict[str, object]:
    """Validate the tool arguments and dispatch an allowlisted handler."""
    if name not in ("get_order", "cancel_order", "issue_refund"):
        return _error("unknown_tool", "Tool is unavailable.")
    if not isinstance(arguments, dict):
        return _error("invalid_arguments", "Tool arguments must be a JSON object.")

    required_fields = {"order_id", "item_ids"} if name == "issue_refund" else {"order_id"}
    if arguments.keys() != required_fields:
        field_names = " and ".join(sorted(required_fields))
        return _error("invalid_arguments", f"{name} accepts exactly {field_names}.")

    order_id = arguments["order_id"]
    if name == "issue_refund":
        return issue_refund(orders, order_id, arguments["item_ids"])
    if name == "cancel_order":
        return cancel_order(orders, order_id)
    return get_order(orders, order_id)
