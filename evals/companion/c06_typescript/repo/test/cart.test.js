'use strict';

const test = require('node:test');
const assert = require('node:assert');
const { Cart } = require('../src/cart');

test('totals a single line', () => {
  const cart = new Cart();
  cart.addItem('A1', 250, 2);
  assert.strictEqual(cart.total(), 500);
});

test('applies a percentage coupon, rounding the discount down', () => {
  const cart = new Cart();
  cart.addItem('A1', 333, 1);
  cart.applyCoupon('ten');
  assert.strictEqual(cart.total(), 300);
});

test('rejects unknown coupons', () => {
  const cart = new Cart();
  assert.throws(() => cart.applyCoupon('FREE'), /unknown coupon/);
});

test('rejects bad quantities', () => {
  const cart = new Cart();
  assert.throws(() => cart.addItem('A1', 100, 0), RangeError);
});
