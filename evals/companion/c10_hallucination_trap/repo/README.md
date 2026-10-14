# sync client

`settings.py` holds the options (`base_url`, `timeout`, `retries`, `page_size`).
`client.fetch()` retries failed requests with a fixed pause between attempts.
