"""Customer order viewer: python app.py CUSTOMER_ID"""

import csv
import json
import sys

from lib.format import money


def load_customers(path="data/customers.csv"):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def load_orders(path="data/orders.json"):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv):
    customers, orders = load_customers(), load_orders()
    cid = argv[0]
    print(customers[cid]["name"])
    for order in (o for o in orders if o["customer"] == cid):
        print(f"  #{order['id']}  {money(order['cents'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
