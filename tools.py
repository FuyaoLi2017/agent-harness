"""Read-only tools for the synthetic order-support example."""

import json
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


def load_orders() -> OrderFixture:
    """Load fresh synthetic order data for one run."""
    fixture_path = Path(__file__).with_name("orders.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return cast(OrderFixture, json.load(fixture_file))


def get_order(fixture: OrderFixture, order_id: str) -> dict[str, object]:
    """Return the order details visible to the fixture's customer."""
    if not isinstance(order_id, str) or not order_id.strip():
        return {
            "error": {
                "code": "invalid_order_id",
                "message": "Order ID must be a nonempty string.",
            },
        }

    for order in fixture["orders"]:
        if order["order_id"] == order_id and order["customer_id"] == fixture["customer_id"]:
            return {
                "order_id": order["order_id"],
                "status": order["status"],
                "delivered_on": order["delivered_on"],
                "items": [item.copy() for item in order["items"]],
                "refunded_item_ids": order["refunded_item_ids"].copy(),
            }

    return {
        "error": {
            "code": "order_not_found",
            "message": "Order was not found.",
        },
    }
