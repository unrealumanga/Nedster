import sqlite3
import datetime

conn = sqlite3.connect('/home/mnm/AI_Lab/Workspace/h2wealth/data/journal.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM trades ORDER BY opened_at DESC LIMIT 15")
rows = cursor.fetchall()

print(f"{'Symbol':10} | {'Side':4} | {'Status':10} | {'PNL':8} | {'Reason':20} | {'Opened At'}")
print("-" * 75)
for r in rows:
    dt = datetime.datetime.fromtimestamp(r['opened_at']).strftime('%Y-%m-%d %H:%M:%S')
    pnl = r['pnl_usdt'] or 0.0
    reason = r['close_reason'] or ""
    print(f"{r['symbol']:10} | {r['side']:4} | {r['status']:10} | ${pnl:<7.2f} | {reason:20} | {dt}")

conn.close()
