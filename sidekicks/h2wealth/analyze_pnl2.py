import sqlite3
import datetime

conn = sqlite3.connect('/home/mnm/AI_Lab/Workspace/h2wealth/data/journal.db')
conn.row_factory = sqlite3.Row

cursor = conn.cursor()
cursor.execute("SELECT * FROM trades WHERE status = 'closed' AND opened_at > (SELECT opened_at FROM trades ORDER BY opened_at DESC LIMIT 1 OFFSET 20)")
rows = cursor.fetchall()

if not rows:
    print("No recent closed trades found.")
else:
    total_pnl = sum(r['pnl_usdt'] or 0 for r in rows)
    total_trades = len(rows)
    winners = [r for r in rows if (r['pnl_usdt'] or 0) > 0]
    losers = [r for r in rows if (r['pnl_usdt'] or 0) <= 0]
    
    win_rate = (len(winners) / total_trades) * 100
    avg_win = sum(r['pnl_usdt'] or 0 for r in winners) / len(winners) if winners else 0
    avg_loss = sum(r['pnl_usdt'] or 0 for r in losers) / len(losers) if losers else 0
    
    print("=== RECENT Trades (Since Last Fix) ===")
    print(f"Total Closed Trades: {total_trades}")
    print(f"Total PNL: ${total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    
    reasons = {}
    for r in rows:
        reason = r['close_reason'] or "unknown"
        if reason not in reasons:
            reasons[reason] = {'count': 0, 'pnl': 0}
        reasons[reason]['count'] += 1
        reasons[reason]['pnl'] += (r['pnl_usdt'] or 0)
        
    print("\n--- PNL by Close Reason (Recent) ---")
    for reason, data in reasons.items():
        print(f"{reason:25} Count: {data['count']:<4} Sum PNL: ${data['pnl']:.2f}")

conn.close()
