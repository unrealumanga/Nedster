# shop

Tiny order-entry library. Every SKU goes through `normalize_sku()` in `shop/skus.py`
before it is looked up, priced or reserved.

Run the tests with `python -m unittest discover -s tests`.
