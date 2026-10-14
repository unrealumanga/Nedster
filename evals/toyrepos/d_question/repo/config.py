DEFAULTS = {"debug": False, "retries": 3}


def parse_config(text):
    """Parse KEY=VALUE lines."""
    if not text.strip():
        return dict(DEFAULTS)
    values = dict(DEFAULTS)
    for line in text.splitlines():
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values
