from pricing import compute_total


def checkout(items):
    return compute_total([i["price"] for i in items], tax=0.1)
