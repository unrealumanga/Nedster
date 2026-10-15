from __future__ import annotations

from pathlib import Path

from evals.companion import helpers

HIDDEN = r'''
import random
from decimal import Decimal as D
import splitter

random.seed(7)
cases = [("100.00", 3), ("0.01", 2), ("0.02", 3), ("10", 7), ("99.99", 4), ("1", 3), ("1000000.01", 9), (100, 3), (33.33, 3)]
cases += [(f"{random.randint(1, 10**7) / 100:.2f}", random.randint(1, 13)) for _ in range(400)]
cases += [('0', 3), ('1.005', 3), (D('2.675'), 7), ('-100.00', 3), ('-0.01', 2)]
for total, n in cases:
    shares = [D(str(s)) for s in splitter.split_evenly(total, n)]
    want = D(str(total)).quantize(D("0.01"))
    assert len(shares) == n, f"{total}/{n}: got {len(shares)} shares"
    assert sum(shares) == want, f"{total}/{n}: shares {shares} sum to {sum(shares)}, not {want}"
    assert all(s == s.quantize(D("0.01")) for s in shares), f"{total}/{n}: not whole cents {shares}"
    assert max(shares) - min(shares) <= D("0.01"), f"{total}/{n}: uneven shares {shares}"
    assert (min(shares) >= 0 if want >= 0 else max(shares) <= 0), shares

assert [D(str(s)) for s in splitter.split_evenly("100.00", 4)] == [D("25.00")] * 4
assert [D(str(s)) for s in splitter.split_evenly("5.00", 1)] == [D("5.00")]
for bad in (0, -2):
    try:
        splitter.split_evenly("10", bad)
    except ValueError:
        pass
    else:
        raise AssertionError("people < 1 must raise ValueError")
# neighbouring behaviour is untouched
assert sum(D(str(s)) for s in splitter.split_weighted("100.00", [1, 1, 1])) == D("100.00")
assert splitter.format_shares(splitter.split_evenly("100.00", 3)).count(",") == 2
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, detail = helpers.run_py(root, HIDDEN)
    return (True, "shares always sum to the total") if ok else (False, detail)
