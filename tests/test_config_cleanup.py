"""Regression test: cloud oracle fields are removed from ModelConfig."""

import os
from pathlib import Path

import pytest

from ironclad.config import NedsterConfig, load_config


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_cloud_oracle_config_loads_but_field_gone(_isolated, monkeypatch):
    """A config.toml with cloud_model / cloud_oracle_enabled keys still loads without error,
    but the resulting NedsterConfig must NOT have a 'cloud_model' attribute on its models."""
    config_dir = _isolated
    (config_dir / "config.toml").write_text('[models]\ncloud_model = "x"\ncloud_oracle_enabled = true\n')

    # Should not raise — the fields in config.toml are just silently ignored now.
    cfg = load_config(str(config_dir / "config.toml"), project_root=config_dir)
    assert hasattr(cfg.models, "local_thinking")

    # The old cloud_model field must no longer exist on ModelConfig.
    assert not hasattr(cfg.models, "cloud_model")
    assert not hasattr(cfg.models, "cloud_oracle_enabled")


def test_env_aliases_no_cloud(_isolated, monkeypatch):
    """NEDSTER_CLOUD_MODEL and NEDSTER_CLOUD_ORACLE env vars are no longer honoured."""
    monkeypatch.setenv("NEDSTER_CLOUD_MODEL", "gpt-x")
    monkeypatch.setenv("NEDSTER_CLOUD_ORACLE", "1")
    cfg = load_config(project_root=_isolated)
    assert not hasattr(cfg.models, "cloud_model")


def test_models_has_no_cloud_fields():
    """ModelConfig class itself must not define any cloud_ field."""
    for attr in dir(NedsterConfig().models):
        assert not attr.startswith("cloud_"), f"unexpected cloud_ attribute: {attr}"
