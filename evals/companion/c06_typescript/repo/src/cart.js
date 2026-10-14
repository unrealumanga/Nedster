'use strict';

const COUPONS = { TEN: 10, TWENTY: 20 };

class Cart {
  constructor() {
    this.lines = [];
    this.coupon = null;
  }

  addItem(sku, priceCents, qty = 1) {
    if (typeof sku !== 'string' || sku === '') {
      throw new TypeError('sku must be a non-empty string');
    }
    if (!Number.isInteger(priceCents) || priceCents < 0) {
      throw new RangeError('price must be a non-negative integer number of cents');
    }
    if (!Number.isInteger(qty) || qty < 1) {
      throw new RangeError('qty must be a positive integer');
    }
    const existing = this.lines.find((line) => line.sku === sku);
    if (existing) {
      existing.qty = qty;
    } else {
      this.lines.push({ sku, priceCents, qty });
    }
  }

  applyCoupon(code) {
    const percent = COUPONS[String(code).toUpperCase()];
    if (percent === undefined) {
      throw new Error(`unknown coupon ${code}`);
    }
    this.coupon = String(code).toUpperCase();
  }

  subtotal() {
    return this.lines.reduce((sum, line) => sum + line.priceCents * line.qty, 0);
  }

  total() {
    const subtotal = this.subtotal();
    if (this.coupon === null) {
      return subtotal;
    }
    const discount = Math.floor((subtotal * COUPONS[this.coupon]) / 100);
    return subtotal - discount;
  }

  itemCount() {
    return this.lines.reduce((sum, line) => sum + line.qty, 0);
  }

  clear() {
    this.lines = [];
    this.coupon = null;
  }
}

module.exports = { Cart, COUPONS };
