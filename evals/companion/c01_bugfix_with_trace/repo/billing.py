"""Invoice totals, currency conversion and the nightly summary.

Money is Decimal throughout. Amounts are rounded half-up to cents only at the
points marked below, never in the middle of a calculation.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

log = logging.getLogger("billing")

CENT = Decimal("0.01")
ZERO = Decimal("0")

TAX_RATES = {
    "NONE": Decimal("0"),
    "DE": Decimal("0.19"),
    "UK": Decimal("0.20"),
    "US-CA": Decimal("0.0725"),
    "US-NY": Decimal("0.08"),
    "US": Decimal("0"),
}

DISCOUNT_TIERS = [
    (Decimal("5000"), Decimal("0.10")),
    (Decimal("1000"), Decimal("0.05")),
    (Decimal("0"), Decimal("0")),
]

LATE_FEE_RATE = Decimal("0.015")
GRACE_DAYS = 5
AGING_BUCKETS = (30, 60, 90)


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class LineItem:
    description: str
    qty: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return self.unit_price * self.qty


@dataclass
class Invoice:
    number: str
    customer: str
    currency: str
    region: str = "NONE"
    lines: list[LineItem] = field(default_factory=list)
    issued: date | None = None
    paid: bool = False

    def add_line(self, description: str, qty: int, unit_price) -> None:
        self.lines.append(LineItem(description, int(qty), Decimal(str(unit_price))))

    @property
    def subtotal(self) -> Decimal:
        return sum((line.amount for line in self.lines), ZERO)


# --------------------------------------------------------------------------
# rounding helpers
# --------------------------------------------------------------------------

def money(value) -> Decimal:
    """Round to whole cents, half up."""
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def percent(value, rate) -> Decimal:
    return Decimal(value) * Decimal(rate)


def clamp_non_negative(value: Decimal) -> Decimal:
    return value if value > ZERO else ZERO


def split_evenly(total: Decimal, parts: int) -> list[Decimal]:
    """Split total into `parts` cent-exact shares that sum back to total."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    total = money(total)
    base = (total / parts).quantize(CENT, rounding=ROUND_HALF_UP)
    shares = [base] * parts
    shares[-1] += total - sum(shares, ZERO)
    return shares


# --------------------------------------------------------------------------
# exchange rates
# --------------------------------------------------------------------------

def parse_rates(text: str) -> dict[str, Decimal]:
    """Parse 'CODE, rate' lines into {code: units per 1 USD}.

    Blank lines and lines starting with '#' are ignored. Lines that cannot be
    parsed are skipped so one bad row does not take the whole report down.
    """
    rates: dict[str, Decimal] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            code, value = (part.strip() for part in line.split(",", 1))
            rates[code.upper()] = Decimal(value)
        except (ValueError, ArithmeticError):
            log.debug("skipping rate line %r", raw)
            continue
    return rates


def load_rates(path: str) -> dict[str, Decimal]:
    with open(path, encoding="utf-8") as handle:
        return parse_rates(handle.read())


def convert(amount: Decimal, src: str, dst: str, rates: dict[str, Decimal]) -> Decimal:
    """Convert amount from currency src to dst via the USD-based rate table."""
    if src == dst:
        return amount
    return amount / rates[src] * rates[dst]


# --------------------------------------------------------------------------
# tax and discount
# --------------------------------------------------------------------------

def tax_rate(region: str) -> Decimal:
    try:
        return TAX_RATES[region]
    except KeyError:
        raise ValueError(f"unknown tax region {region!r}") from None


def discount_rate(subtotal: Decimal) -> Decimal:
    for threshold, rate in DISCOUNT_TIERS:
        if subtotal >= threshold:
            return rate
    return ZERO


def invoice_total(invoice: Invoice) -> Decimal:
    """Total in the invoice's own currency: subtotal - discount + tax."""
    subtotal = invoice.subtotal
    discount = money(percent(subtotal, discount_rate(subtotal)))
    taxable = subtotal - discount
    tax = money(percent(taxable, tax_rate(invoice.region)))
    return money(taxable + tax)


# --------------------------------------------------------------------------
# late fees and ageing
# --------------------------------------------------------------------------

def days_overdue(invoice: Invoice, today: date, terms_days: int = 30) -> int:
    if invoice.issued is None or invoice.paid:
        return 0
    due = invoice.issued + timedelta(days=terms_days)
    return max((today - due).days, 0)


def late_fee(invoice: Invoice, today: date, terms_days: int = 30) -> Decimal:
    overdue = days_overdue(invoice, today, terms_days)
    if overdue <= GRACE_DAYS:
        return ZERO
    months = (overdue + 29) // 30
    return money(percent(invoice_total(invoice), LATE_FEE_RATE) * months)


def aging_bucket(invoice: Invoice, today: date) -> str:
    overdue = days_overdue(invoice, today)
    if overdue == 0:
        return "current"
    previous = 0
    for limit in AGING_BUCKETS:
        if overdue <= limit:
            return f"{previous + 1}-{limit}"
        previous = limit
    return f"{AGING_BUCKETS[-1]}+"


def aging_report(invoices: list[Invoice], today: date) -> dict[str, int]:
    report: dict[str, int] = {}
    for invoice in invoices:
        bucket = aging_bucket(invoice, today)
        report[bucket] = report.get(bucket, 0) + 1
    return report


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = ("number", "customer", "currency", "region", "item", "qty", "unit_price")


def load_invoices(path: str) -> list[Invoice]:
    """Read one CSV row per line item; rows sharing a number form one invoice."""
    invoices: dict[str, Invoice] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            number = row["number"].strip()
            invoice = invoices.get(number)
            if invoice is None:
                invoice = Invoice(
                    number=number,
                    customer=row["customer"].strip(),
                    currency=row["currency"].strip().upper(),
                    region=row["region"].strip() or "NONE",
                )
                invoices[number] = invoice
            invoice.add_line(row["item"].strip(), int(row["qty"]), row["unit_price"].strip())
    return list(invoices.values())


def validate_invoice(invoice: Invoice) -> list[str]:
    problems = []
    if not invoice.lines:
        problems.append("no line items")
    if invoice.region not in TAX_RATES:
        problems.append(f"unknown region {invoice.region}")
    for line in invoice.lines:
        if line.qty <= 0:
            problems.append(f"{line.description}: qty must be positive")
        if line.unit_price < ZERO:
            problems.append(f"{line.description}: negative price")
    return problems


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summarize(invoices: list[Invoice], rates: dict[str, Decimal], base: str = "USD") -> dict:
    """Per-customer and grand totals, all converted into `base`."""
    by_customer: dict[str, Decimal] = {}
    total = ZERO
    for invoice in invoices:
        problems = validate_invoice(invoice)
        if problems:
            log.warning("invoice %s skipped: %s", invoice.number, "; ".join(problems))
            continue
        amount = convert(invoice_total(invoice), invoice.currency, base, rates)
        by_customer[invoice.customer] = by_customer.get(invoice.customer, ZERO) + amount
        total += amount
    return {
        "base": base,
        "by_customer": {name: money(value) for name, value in sorted(by_customer.items())},
        "total": money(total),
    }


def format_summary(summary: dict) -> str:
    width = max((len(name) for name in summary["by_customer"]), default=8)
    lines = [f"Billing summary ({summary['base']})", "-" * (width + 16)]
    for name, value in summary["by_customer"].items():
        lines.append(f"{name:<{width}}  {value:>12,.2f}")
    lines.append("-" * (width + 16))
    lines.append(f"{'TOTAL':<{width}}  {summary['total']:>12,.2f}")
    return "\n".join(lines)
