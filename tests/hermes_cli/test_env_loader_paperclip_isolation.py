from __future__ import annotations

import os

from hermes_cli import env_loader


def _write_env(path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")


def test_paperclip_child_preserves_incoming_run_key_and_drops_signing_secret(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _write_env(
        hermes_home / ".env",
        {
            "PAPERCLIP_API_KEY": "persistent-codex-home-key",
            "PAPERCLIP_AGENT_JWT_SECRET": "persistent-signing-secret",
        },
    )
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-simulated")
    monkeypatch.setenv("PAPERCLIP_API_KEY", "run-scoped-tf-hermes-lead-key")
    monkeypatch.setenv("PAPERCLIP_AGENT_JWT_SECRET", "inherited-signing-secret")
    monkeypatch.setattr(env_loader, "_apply_external_secret_sources", lambda _home: None)
    monkeypatch.setattr(env_loader, "_apply_managed_env", lambda: None)

    env_loader.load_hermes_dotenv(hermes_home=hermes_home)

    assert os.environ["PAPERCLIP_API_KEY"] == "run-scoped-tf-hermes-lead-key"
    assert "PAPERCLIP_AGENT_JWT_SECRET" not in os.environ


def test_normal_hermes_dotenv_precedence_is_unchanged(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _write_env(
        hermes_home / ".env",
        {
            "PAPERCLIP_API_KEY": "persistent-codex-home-key",
            "PAPERCLIP_AGENT_JWT_SECRET": "persistent-signing-secret",
        },
    )
    monkeypatch.delenv("PAPERCLIP_RUN_ID", raising=False)
    monkeypatch.setenv("PAPERCLIP_API_KEY", "stale-shell-key")
    monkeypatch.delenv("PAPERCLIP_AGENT_JWT_SECRET", raising=False)
    monkeypatch.setattr(env_loader, "_apply_external_secret_sources", lambda _home: None)
    monkeypatch.setattr(env_loader, "_apply_managed_env", lambda: None)

    env_loader.load_hermes_dotenv(hermes_home=hermes_home)

    assert os.environ["PAPERCLIP_API_KEY"] == "persistent-codex-home-key"
    assert os.environ["PAPERCLIP_AGENT_JWT_SECRET"] == "persistent-signing-secret"
