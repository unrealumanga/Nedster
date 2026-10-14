def compute_total(prices, tax=0.0):
    return round(sum(prices) * (1 + tax), 2)
