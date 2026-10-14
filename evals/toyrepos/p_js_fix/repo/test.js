const assert = require('node:assert');
const { rangeSum } = require('./index');

assert.strictEqual(rangeSum(5), 15);
assert.strictEqual(rangeSum(1), 1);
assert.strictEqual(rangeSum(0), 0);
console.log('ok');
