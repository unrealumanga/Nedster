# half-finished experiment with sorting orders by value; not imported anywhere
from app import load_orders

print(sorted(load_orders(), key=lambda o: -o["cents"])[:3])
