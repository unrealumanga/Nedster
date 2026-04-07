# Nedster Project Log
from datetime import datetime

def log_entry(title, description):
    date = datetime.now().strftime("%Y-%m-%d")
    print(f"[{date}] {title}: {description}")
