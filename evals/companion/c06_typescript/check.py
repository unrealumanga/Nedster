from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from evals.companion import helpers

HIDDEN = r"""
const assert = require('node:assert');
const { Cart } = require('./src/cart');

let c = new Cart();
c.addItem('A1', 250, 2);
c.addItem('A1', 250, 3);
assert.strictEqual(c.lines.length, 1, 'same sku must stay one line');
assert.strictEqual(c.itemCount(), 5, 'quantities must add up');
assert.strictEqual(c.total(), 1250);

c.addItem('B2', 100, 1);
assert.strictEqual(c.removeItem('A1'), true);
assert.strictEqual(c.removeItem('A1'), false);
assert.strictEqual(c.removeItem('nope'), false);
assert.strictEqual(c.total(), 100);
assert.strictEqual(c.itemCount(), 1);

c.addItem('A1', 400, 1);
c.applyCoupon('TWENTY');
assert.strictEqual(c.total(), 400);
assert.strictEqual(c.removeItem('B2'), true);
assert.strictEqual(c.total(), 320, 'coupon still applies after a removal');
assert.strictEqual(c.removeItem('A1'), true);
assert.strictEqual(c.total(), 0);

c = new Cart();
assert.throws(() => c.addItem('A1', 100, 0), RangeError);
assert.throws(() => c.addItem('', 100, 1), TypeError);
c.addItem('A1', 100, 1);
assert.throws(() => c.addItem('A1', 100, -1), RangeError);
assert.strictEqual(c.itemCount(), 1, 'a rejected add must not change the cart');
for (const qty of [1.5, NaN, Infinity]) assert.throws(() => c.addItem('A1', 100, qty), RangeError);
for (const price of [-1, 1.5, NaN, Infinity]) assert.throws(() => c.addItem('X', price), RangeError);
c.addItem('A1', 900, 2);
assert.strictEqual(c.subtotal(), 300, 'existing line price is retained');
c.clear();
assert.strictEqual(c.itemCount(), 0);
assert.strictEqual(c.total(), 0);
for (const sku of ['__proto__', 'constructor', 'MiXeD', 'mixed']) c.addItem(sku, 10);
assert.strictEqual(c.itemCount(), 4);
assert.strictEqual(c.removeItem('MiXeD'), true);
assert.strictEqual(c.itemCount(), 3);
"""


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = helpers.run_argv(root, ["node", "-e", HIDDEN])
    if not ok:
        return False, f"hidden cart cases failed: {out}"
    ok, detail = helpers.run_argv(root, ["node", "--test", "--test-reporter=tap"])
    if not ok or not re.search(r'# pass [1-9]\d*', detail):
        return False, f"tests must run and pass: {detail}"
    # Runtime wrappers avoid depending on the candidate's implementation spelling.
    mutants = {
        'duplicate quantity': "const add = this.addItem; this.addItem = function(s,p,q=1) { const existed = this.lines.some(x=>x.sku===s); add.call(this,s,p,q); if(existed) this.lines.find(x=>x.sku===s).qty=q; };",
        'removeItem': "this.removeItem = function() { return false; };",
    }
    for name, mutation in mutants.items():
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'repo'
            shutil.copytree(root, work, ignore=shutil.ignore_patterns('node_modules', '.git'))
            with (work / 'src/cart.js').open('a', encoding='utf-8') as f:
                f.write('\n;module.exports.Cart = class extends module.exports.Cart { constructor(...args) { super(...args); ' + mutation + ' } };\n')
            ok, detail = helpers.run_argv(work, ['node', '--test', '--test-reporter=tap'])
        if ok or not re.search(r'# fail [1-9]\d*', detail):
            return False, f"tests did not catch broken {name}: {detail}"
    return True, 'cart behavior and tests for both changes passed'
