"""Tests for config.py."""

import json

import pytest

from config import (
    HARD_MAX_DURATION_SECONDS,
    AIConfig,
    AppConfig,
    ConfigError,
    DurationConfig,
    ScoringWeights,
    load_config,
)


def test_defaults_are_valid():
    AppConfig().validate()


def test_duration_cannot_exceed_hard_maximum():
    cfg = AppConfig()
    cfg.durations.absolute_max_seconds = HARD_MAX_DURATION_SECONDS + 1
    with pytest.raises(ConfigError):
        cfg.validate()


def test_duration_ordering_enforced():
    with pytest.raises(ConfigError):
        DurationConfig(min_seconds=200, preferred_max_seconds=180).validate()
    with pytest.raises(ConfigError):
        DurationConfig(preferred_max_seconds=199, absolute_max_seconds=190).validate()


def test_overlap_threshold_range():
    cfg = AppConfig()
    cfg.selection.overlap_threshold = 1.5
    with pytest.raises(ConfigError):
        cfg.validate()


def test_vertical_must_be_9_16():
    cfg = AppConfig()
    cfg.vertical.width, cfg.vertical.height = 1920, 1080
    with pytest.raises(ConfigError):
        cfg.validate()


def test_weights_normalize_to_one():
    weights = ScoringWeights(
        interestingness=3, hook_strength=1, context_completeness=0,
        emotional_impact=0, standalone_quality=0, relevance=0,
    )
    normalized = weights.normalized()
    assert pytest.approx(sum(normalized.values())) == 1.0
    assert pytest.approx(normalized["interestingness"]) == 0.75


def test_all_zero_weights_rejected():
    weights = ScoringWeights(0, 0, 0, 0, 0, 0)
    with pytest.raises(ConfigError):
        weights.validate()


def test_provider_presets_resolve():
    ollama = AIConfig(provider="ollama")
    assert ollama.resolved_base_url() == "http://localhost:11434/v1"
    assert ollama.resolved_model()
    assert not ollama.requires_api_key()

    lmstudio = AIConfig(provider="lmstudio", model="my-model")
    assert lmstudio.resolved_base_url() == "http://localhost:1234/v1"
    assert lmstudio.resolved_model() == "my-model"


def test_unknown_provider_rejected():
    cfg = AppConfig()
    cfg.ai.provider = "does-not-exist"
    with pytest.raises(ConfigError):
        cfg.validate()


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ai = AIConfig(provider="openrouter")
    with pytest.raises(ConfigError):
        ai.get_api_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-1234567890")
    assert ai.get_api_key() == "sk-or-test-1234567890"


def test_local_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert AIConfig(provider="ollama").get_api_key() is None


def test_chunk_overlap_must_be_smaller_than_chunk():
    cfg = AppConfig()
    cfg.ai.chunk_overlap_chars = cfg.ai.chunk_max_chars
    with pytest.raises(ConfigError):
        cfg.validate()


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    for name in ("SHORTS_AI_PROVIDER", "SHORTS_AI_MODEL", "SHORTS_AI_BASE_URL", "SHORTS_OUTPUT_DIR"):
        monkeypatch.delenv(name, raising=False)
    cfg = AppConfig()
    cfg.selection.num_clips = 9
    cfg.ai.provider = "lmstudio"
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = load_config(path)
    assert loaded.selection.num_clips == 9
    assert loaded.ai.provider == "lmstudio"


def test_unknown_setting_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"selection": {"num_clipz": 3}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_env_override_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("SHORTS_AI_PROVIDER", "lmstudio")
    cfg = load_config(None)
    assert cfg.ai.provider == "lmstudio"


def test_config_dict_never_contains_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-super-secret-value")
    cfg = AppConfig()
    cfg.ai.provider = "openrouter"
    assert "sk-or-super-secret-value" not in json.dumps(cfg.to_dict())