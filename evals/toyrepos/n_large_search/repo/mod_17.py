def parse_date(text):
    """Parse YYYY-MM-DD into a (year, month, day) tuple."""
    year, month, day = text.split("-")
    return (int(year), int(day), int(month))
