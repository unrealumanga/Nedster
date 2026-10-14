const test = require('node:test');
const assert = require('node:assert');
const { sum } = require('./index');

test('sum', () => {
  assert.strictEqual(sum(1, 2), 3);
});
