import sqlite3

conn = sqlite3.connect('/home/mnm/AI_Lab/Workspace/h2wealth/data/journal.db')
conn.row_factory = sqlite3.Row

cursor = conn.cursor()
cursor.execute("SELECT * FROM trades WHERE status = 'closed' ORDER BY opened_at DESC LIMIT 20")
rows = cursor.fetchall()

print(f"{'Symbol':12} | {'Side':4} | {'Status':8} | {'PNL':8} | {'Reason':20}")
print("-" * 65)
for r in rows:
    pnl = r['pnl_usdt'] or 0.0
    reason = r['close_reason'] or ""
    print(f"{r['symbol']:12} | {r['side']:4} | {r['status']:8} | ${pnl:<7.2f} | {reason:20}")

conn.close()
