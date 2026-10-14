"""Static settings; a real deployment overrides these from the environment."""

SETTINGS = {
    "webhook_secret": "dev-secret",
    "slack_webhook_url": "https://hooks.slack.invalid/services/T000/B000/XXX",
    "dedupe_window_seconds": 300,
    "max_retries": 4,
    "channels_by_team": {"ops": ["slack"], "db": ["slack", "email"]},
}
