"""Order/payment reporting helpers used by the finance export."""

from __future__ import annotations


def total_paid(payments: list[dict]) -> float:
    return round(sum(p["amount"] for p in payments), 2)


def join_orders(orders: list[dict], payments: list[dict]) -> list[tuple[dict, dict]]:
    """Pair every order with each payment that references it.

    Result order: orders in their input order; for one order, its payments in
    their input order. Orders without payments are left out. A payment whose
    order_id matches no order is ignored.
    """
    pairs = []
    for order in orders:
        for payment in payments:
            if payment["order_id"] == order["id"]:
                pairs.append((order, payment))
    return pairs


def unpaid_orders(orders: list[dict], payments: list[dict]) -> list[dict]:
    paid_ids = {p["order_id"] for p in payments}
    return [o for o in orders if o["id"] not in paid_ids]


def format_pair(pair: tuple[dict, dict]) -> str:
    order, payment = pair
    return f"{order['id']}: {payment['amount']:.2f} via {payment.get('method', '?')}"
