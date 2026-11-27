"""Tools for the synthetic order-support example."""

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict, cast

from workflow import ToolOutcome


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


def open_handlers() -> dict[str, Callable[[dict[str, object]], ToolOutcome]]:
    """Bind each order tool to the same fresh session fixture."""
    fixture_path = Path(__file__).with_name("orders.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        orders = cast(OrderFixture, json.load(fixture_file))

    def bind(
        handler: Callable[..., dict[str, object]], name: str, required_fields: set[str]
    ) -> Callable[[dict[str, object]], ToolOutcome]:
        def execute(arguments: dict[str, object]) -> ToolOutcome:
            if not isinstance(arguments, dict):
                content = _error("invalid_arguments", "Tool arguments must be a JSON object.")
            elif arguments.keys() != required_fields:
                field_names = " and ".join(sorted(required_fields))
                content = _error("invalid_arguments", f"{name} accepts exactly {field_names}.")
            else:
                content = handler(orders, **arguments)
            return ToolOutcome(content=content, is_error="error" in content)

        return execute

    return {
        "get_order": bind(get_order, "get_order", {"order_id"}),
        "cancel_order": bind(cancel_order, "cancel_order", {"order_id"}),
        "issue_refund": bind(issue_refund, "issue_refund", {"order_id", "item_ids"}),
    }
