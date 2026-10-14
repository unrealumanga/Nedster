from __future__ import annotations

from pathlib import Path

from evals.companion import helpers

HIDDEN = r'''
import billing
from decimal import Decimal as D
import contextlib, io, runpy

expected = billing.format_summary(billing.summarize(
    billing.load_invoices('data/invoices.csv'), billing.load_rates('data/rates.txt')))
output = io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_path('run_report.py', run_name='__main__')
assert output.getvalue().strip() == expected, 'report entry point no longer prints the actual summary'

text = """# header
USD, 1.0
EUR, 0.92   # euro
GBP,0.79#pound
 jpy , 151.2   # yen, lower-case code
"""
rates = billing.parse_rates(text)
assert rates.get("EUR") == D("0.92"), f"EUR missing/wrong: {rates}"
assert rates.get("GBP") == D("0.79"), f"GBP missing/wrong: {rates}"
assert rates.get("JPY") == D("151.2"), f"JPY missing/wrong: {rates}"
assert rates.get("USD") == D("1.0"), rates

inv = billing.Invoice("1", "Acme", "EUR")
inv.add_line("thing", 1, "92.00")
out = billing.summarize([inv], rates)
assert out["total"] == D("100.00"), f"EUR 92 should be 100.00 USD, got {out['total']}"
assert out["by_customer"]["Acme"] == D("100.00"), out

# an unknown currency must still fail loudly, never convert at a made-up rate
bad = billing.Invoice("2", "Acme", "XYZ")
bad.add_line("thing", 1, "10")
try:
    r = billing.summarize([bad], rates)
except Exception:
    pass
else:
    raise AssertionError(f"unknown currency silently converted: {r}")

# well-formed lines still parse the same
assert billing.parse_rates("USD, 1\nCAD, 1.35\n")["CAD"] == D("1.35")
assert billing.parse_rates(' # comment\ninvalid\nEUR, nonsense\nUSD,1\n') == {'USD': D('1')}
# Vary values and codes so patching just the shipped currencies does not pass.
for i in range(1, 20):
    value = D(i) / 7
    parsed = billing.parse_rates(f'USD,1\r\n zz{i}, {value}# note, ignored\r\n')
    assert parsed == {'USD': D(1), f'ZZ{i}': value}, parsed
    assert billing.convert(D(7), 'USD', f'ZZ{i}', parsed) == D(7) * value
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = helpers.run_argv(root, [__import__("sys").executable, "-B", "run_report.py"])
    if not ok:
        return False, f"run_report.py still fails: {out[-300:]}"
    if "TOTAL" not in out:
        return False, f"report has no TOTAL line: {out[-200:]}"
    ok, detail = helpers.run_py(root, HIDDEN)
    return (True, "rates with inline comments parse; unknown currency still raises") if ok else (False, detail)
