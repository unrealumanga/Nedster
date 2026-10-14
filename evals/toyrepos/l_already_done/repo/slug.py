def slugify(text):
    """Return a lowercase, hyphen-separated slug."""
    return text.strip().lower().replace(" ", "-")
