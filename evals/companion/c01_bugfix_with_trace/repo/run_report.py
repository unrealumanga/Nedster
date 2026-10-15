"""Nightly billing summary: python run_report.py"""

import billing


def main():
    rates = billing.load_rates("data/rates.txt")
    invoices = billing.load_invoices("data/invoices.csv")
    print(billing.format_summary(billing.summarize(invoices, rates)))


if __name__ == "__main__":
    main()
