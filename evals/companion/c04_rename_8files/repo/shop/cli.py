"""python -m shop.cli lookup SKU | order SKU:QTY ..."""

from __future__ import annotations

import argparse

from shop import exporter, importer
from shop.catalog import find
from shop.skus import normalize_sku


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="shop")
    sub = parser.add_subparsers(dest="command", required=True)
    look = sub.add_parser("lookup")
    look.add_argument("sku")
    order = sub.add_parser("order")
    order.add_argument("lines", nargs="+", help="SKU:QTY")
    args = parser.parse_args(argv)

    if args.command == "lookup":
        entry = find(args.sku)
        print(f"{normalize_sku(args.sku)} {entry['name']} {entry['price']}")
        return 0
    rows = [line.replace(":", ",") for line in args.lines]
    built, rejected = importer.import_rows(rows)
    print(exporter.export_order(built))
    for row in rejected:
        print(f"rejected: {row}")
    print(f"total {built.total()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
