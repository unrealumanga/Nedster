import urllib.request


def fetch(url):
    """Fetch the body of a URL as bytes."""
    with urllib.request.urlopen(url) as response:
        return response.read()
