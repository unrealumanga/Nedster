import pricing


def summary(prices):
    return f"Total: {pricing.compute_total(prices)}"
