from __future__ import annotations

from pathlib import Path

from evals.companion import helpers

HIDDEN = r'''
import copy, random, time, statistics
import reports

def brute(orders, payments):
    return [(o, p) for o in orders for p in payments if p["order_id"] == o["id"]]

random.seed(11)
for trial in range(60):
    n_o, n_p = random.randint(0, 25), random.randint(0, 40)
    ids = [random.randint(1, 15) for _ in range(n_o)]          # duplicate order ids are allowed
    orders = [{"id": i, "n": k} for k, i in enumerate(ids)]
    payments = [{"order_id": random.randint(1, 20), "amount": float(k), "n": k} for k in range(n_p)]
    o_copy, p_copy = copy.deepcopy(orders), copy.deepcopy(payments)
    got = reports.join_orders(orders, payments)
    want = brute(orders, payments)
    assert len(got) == len(want) and all(g[0] is w[0] and g[1] is w[1] for g, w in zip(got, want)), \
        f"trial {trial}: wrong pairs/order\n got={got}\n want={want}"
    assert orders == o_copy and payments == p_copy, "inputs were mutated"
assert reports.join_orders([], []) == []
assert reports.join_orders([{"id": 1}], []) == []
assert isinstance(reports.join_orders([{"id": 1}], [{"order_id": 1, "amount": 1}])[0], tuple)
orders = [{'id': key} for key in ('007', 7, 'alpha', '007', -2)]
payments = [{'order_id': key, 'amount': i} for i, key in enumerate((7, '007', 'unknown', -2, '007'))]
got, want = reports.join_orders(orders, payments), brute(orders, payments)
assert isinstance(got, list) and len(got) == len(want)
assert all(g[0] is w[0] and g[1] is w[1] for g, w in zip(got, want))

N = 200_000
orders = [{"id": i} for i in range(N)]
payments = [{"order_id": (i * 7919) % N, "amount": 1.0} for i in range(N)] + [{"order_id": 5, "amount": 2.0}]
def indexed():
    by_id = {}
    for p in payments:
        by_id.setdefault(p['order_id'], []).append(p)
    return [(o, p) for o in orders for p in by_id.get(o['id'], ())]

# Calibrate against the same workload on this machine, outside import/startup.
samples = []
for _ in range(3):
    start = time.perf_counter()
    expected = indexed()
    samples.append(time.perf_counter() - start)
limit = max(10.0, statistics.median(samples) * 20)
start = time.perf_counter()
out = reports.join_orders(orders, payments)
elapsed = time.perf_counter() - start
assert len(out) == N + 1, len(out)
assert all(isinstance(g, tuple) and g[0] is w[0] and g[1] is w[1] for g, w in zip(out, expected)), 'large-input pairs/order differ'
assert all(o == {'id': i} for i, o in enumerate(orders)), 'large orders were mutated'
assert all(p == {'order_id': (i * 7919) % N, 'amount': 1.0} for i, p in enumerate(payments[:-1])), 'large payments were mutated'
assert payments[-1] == {'order_id': 5, 'amount': 2.0}
assert elapsed < limit, f"join_orders took {elapsed:.1f}s (calibrated limit {limit:.1f}s)"
print("ok", round(elapsed, 2))
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, detail = helpers.run_py(root, HIDDEN, timeout=60)
    return (True, detail) if ok else (False, detail)
