"""Central configuration for the application.

All user-adjustable settings live here as dataclasses. Nothing important is
hardcoded elsewhere in the project.

API keys are never stored in the config file or in source code. The config only
stores the *name* of the environment variable that holds the key, and the key is
read at the moment it is needed via ``AIConfig.get_api_key()``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The project requires that no exported clip ever exceeds this many seconds.
HARD_MAX_DURATION_SECONDS = 200.0

# Providers that speak the OpenAI-compatible chat completions protocol.
# Free and local backends only: no paid SDKs are required.
PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
        "requires_api_key": False,
        "default_model": "llama3.1:8b",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "",
        "requires_api_key": False,
        # LM Studio serves whichever model is loaded; set the exact name in config.
        "default_model": "local-model",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "requires_api_key": True,
        # Free model names change over time. Models ending in ":free" cost nothing.
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "openai_compatible": {
        # Any other OpenAI-compatible server. Set base_url (and model) yourself.
        "base_url": "",
        "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
        "requires_api_key": False,
        "default_model": "",
    },
}

CROP_MODES = ("auto", "center_crop", "blurred_background")
FOREGROUND_POSITIONS = ("center", "top", "bottom")
SUBTITLE_POSITIONS = ("top", "middle", "lower_third")

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class AIConfig:
    """Settings for the AI model used for transcript analysis."""

    provider: str = "ollama"
    model: str = ""  # empty -> provider preset default
    base_url: str = ""  # empty -> provider preset default
    api_key_env: str = ""  # name of the environment variable; empty -> preset default
    temperature: float = 0.2
    request_timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    # Local models often have small context windows, so chunking is configurable.
    chunk_max_chars: int = 12000
    chunk_overlap_chars: int = 1500
    # Some local servers reject response_format; keep off unless your backend supports it.
    request_json_mode: bool = False
    max_output_tokens: int = 8000  # reasoning models spend part of this on thinking
    candidates_per_chunk: int = 5  # most clips kept from each transcript part
    min_seconds_between_requests: float = 3.0  # throttle for free-tier rate limits; 0 for local
    fallback_models: List[str] = field(default_factory=list)  # tried in order if the main model fails

    def _preset(self) -> Dict[str, Any]:
        return PROVIDER_PRESETS.get(self.provider, {})

    def resolved_model(self) -> str:
        return self.model or self._preset().get("default_model", "")

    def resolved_base_url(self) -> str:
        return (self.base_url or self._preset().get("base_url", "")).rstrip("/")

    def resolved_api_key_env(self) -> str:
        return self.api_key_env or self._preset().get("api_key_env", "")

    def requires_api_key(self) -> bool:
        return bool(self._preset().get("requires_api_key", False))

    def get_api_key(self) -> Optional[str]:
        """Return the API key from the environment, or None if not set.

        Raises ConfigError if the provider requires a key and none is available.
        The key is never logged or stored.
        """
        env_name = self.resolved_api_key_env()
        key = os.environ.get(env_name, "").strip() if env_name else ""
        if self.requires_api_key() and not key:
            raise ConfigError(
                f"Provider '{self.provider}' needs an API key. Set the "
                f"environment variable {env_name} (see .env.example)."
            )
        return key or None

    def validate(self) -> None:
        _require(
            self.provider in PROVIDER_PRESETS,
            f"Unknown AI provider '{self.provider}'. "
            f"Choose one of: {', '.join(PROVIDER_PRESETS)}.",
        )
        _require(bool(self.resolved_base_url()), "AI base_url is empty; set ai.base_url.")
        _require(
            self.base_url == "" or self.base_url.startswith(("http://", "https://")),
            "ai.base_url must start with http:// or https://.",
        )
        _require(0.0 <= self.temperature <= 2.0, "ai.temperature must be between 0 and 2.")
        _require(self.request_timeout_seconds > 0, "ai.request_timeout_seconds must be positive.")
        _require(self.max_retries >= 0, "ai.max_retries cannot be negative.")
        _require(self.retry_backoff_seconds >= 0, "ai.retry_backoff_seconds cannot be negative.")
        _require(self.chunk_max_chars >= 1000, "ai.chunk_max_chars must be at least 1000.")
        _require(
            0 <= self.chunk_overlap_chars < self.chunk_max_chars,
            "ai.chunk_overlap_chars must be >= 0 and smaller than ai.chunk_max_chars.",
        )
        _require(self.max_output_tokens >= 256, "ai.max_output_tokens must be at least 256.")
        _require(
            1 <= self.candidates_per_chunk <= 20,
            "ai.candidates_per_chunk must be between 1 and 20.",
        )
        _require(
            self.min_seconds_between_requests >= 0,
            "ai.min_seconds_between_requests cannot be negative.",
        )
        _require(
            isinstance(self.fallback_models, list)
            and all(isinstance(m, str) and m.strip() for m in self.fallback_models),
            "ai.fallback_models must be a list of model names.",
        )


@dataclass
class DurationConfig:
    """Clip duration limits, in seconds."""

    min_seconds: float = 20.0
    preferred_max_seconds: float = 180.0
    absolute_max_seconds: float = 200.0

    def validate(self) -> None:
        _require(self.min_seconds > 0, "durations.min_seconds must be positive.")
        _require(
            self.min_seconds <= self.preferred_max_seconds,
            "durations.min_seconds cannot exceed durations.preferred_max_seconds.",
        )
        _require(
            self.preferred_max_seconds <= self.absolute_max_seconds,
            "durations.preferred_max_seconds cannot exceed durations.absolute_max_seconds.",
        )
        _require(
            self.absolute_max_seconds <= HARD_MAX_DURATION_SECONDS,
            f"durations.absolute_max_seconds cannot exceed {HARD_MAX_DURATION_SECONDS:.0f}.",
        )


@dataclass
class SelectionConfig:
    """How many clips to keep and how to treat overlaps."""

    num_clips: int = 5
    min_score: float = 60.0  # 0-100 scale
    overlap_threshold: float = 0.5  # fraction of the shorter clip that may overlap
    allow_merge: bool = True

    def validate(self) -> None:
        _require(self.num_clips >= 1, "selection.num_clips must be at least 1.")
        _require(0.0 <= self.min_score <= 100.0, "selection.min_score must be between 0 and 100.")
        _require(
            0.0 <= self.overlap_threshold <= 1.0,
            "selection.overlap_threshold must be between 0 and 1.",
        )


@dataclass
class ScoringWeights:
    """Relative weights for the final 0-100 clip score.

    Weights need not sum to 1; they are normalized before use.
    """

    interestingness: float = 0.30
    hook_strength: float = 0.20
    context_completeness: float = 0.20
    emotional_impact: float = 0.10
    standalone_quality: float = 0.15
    relevance: float = 0.05

    def validate(self) -> None:
        values = self.as_dict()
        _require(all(v >= 0 for v in values.values()), "scoring weights cannot be negative.")
        _require(sum(values.values()) > 0, "At least one scoring weight must be greater than 0.")

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def normalized(self) -> Dict[str, float]:
        """Return the weights scaled so they sum to 1.0."""
        values = self.as_dict()
        total = sum(values.values())
        _require(total > 0, "At least one scoring weight must be greater than 0.")
        return {name: value / total for name, value in values.items()}


@dataclass
class VerticalConfig:
    """Vertical (9:16) formatting settings."""

    width: int = 1080
    height: int = 1920
    crop_mode: str = "auto"  # auto | center_crop | blurred_background
    foreground_scale: float = 1.0  # fraction of frame width used by the foreground video
    background_blur: int = 30  # blur strength for the background
    background_scale: float = 1.0  # extra enlargement of the blurred background (>= 1.0)
    foreground_position: str = "center"  # center | top | bottom
    foreground_offset_px: int = 0  # vertical nudge in pixels

    def validate(self) -> None:
        _require(self.width > 0 and self.height > 0, "vertical width/height must be positive.")
        _require(
            self.width % 2 == 0 and self.height % 2 == 0,
            "vertical width/height must be even numbers (required by H.264).",
        )
        _require(
            abs(self.width / self.height - 9 / 16) < 0.01,
            "vertical output must be 9:16 (for example 1080x1920).",
        )
        _require(self.crop_mode in CROP_MODES, f"vertical.crop_mode must be one of {CROP_MODES}.")
        _require(
            0.1 <= self.foreground_scale <= 1.0,
            "vertical.foreground_scale must be between 0.1 and 1.0.",
        )
        _require(0 <= self.background_blur <= 100, "vertical.background_blur must be 0-100.")
        _require(
            1.0 <= self.background_scale <= 3.0,
            "vertical.background_scale must be between 1.0 and 3.0.",
        )
        _require(
            self.foreground_position in FOREGROUND_POSITIONS,
            f"vertical.foreground_position must be one of {FOREGROUND_POSITIONS}.",
        )


@dataclass
class SubtitleConfig:
    """Burned-in subtitle style."""

    font: str = "Arial"
    font_size: int = 64
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    highlight_color: str = "#FFD700"
    outline_width: int = 4
    position: str = "lower_third"  # top | middle | lower_third
    bottom_margin_px: int = 420  # keeps text above the platform UI area
    max_words_per_line: int = 4
    max_lines: int = 2
    word_highlight: bool = True
    correct_text: bool = False  # keep the original transcript wording by default

    def validate(self) -> None:
        _require(bool(self.font.strip()), "subtitles.font cannot be empty.")
        _require(10 <= self.font_size <= 200, "subtitles.font_size must be between 10 and 200.")
        for name in ("text_color", "outline_color", "highlight_color"):
            _require(
                bool(_HEX_COLOR.match(getattr(self, name))),
                f"subtitles.{name} must be a hex color like #FFFFFF.",
            )
        _require(0 <= self.outline_width <= 20, "subtitles.outline_width must be between 0 and 20.")
        _require(
            self.position in SUBTITLE_POSITIONS,
            f"subtitles.position must be one of {SUBTITLE_POSITIONS}.",
        )
        _require(self.bottom_margin_px >= 0, "subtitles.bottom_margin_px cannot be negative.")
        _require(1 <= self.max_words_per_line <= 12, "subtitles.max_words_per_line must be 1-12.")
        _require(1 <= self.max_lines <= 4, "subtitles.max_lines must be 1-4.")


@dataclass
class ExportConfig:
    """Final encoding settings."""

    container: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18  # lower = higher quality
    preset: str = "medium"
    fps: int = 30
    audio_bitrate: str = "192k"
    pixel_format: str = "yuv420p"

    def validate(self) -> None:
        _require(0 <= self.crf <= 51, "export.crf must be between 0 and 51.")
        _require(
            self.preset
            in (
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow",
            ),
            "export.preset must be a valid x264 preset.",
        )
        _require(1 <= self.fps <= 120, "export.fps must be between 1 and 120.")
        _require(
            bool(re.match(r"^\d+k$", self.audio_bitrate)),
            "export.audio_bitrate must look like 192k.",
        )

@dataclass
class DownloadConfig:
    """Settings for downloading videos from URLs (yt-dlp)."""

    max_height: int = 1080  # highest video resolution to download
    merge_format: str = "mp4"  # container used when merging video and audio
    retries: int = 3
    socket_timeout_seconds: float = 30.0
    cookies_file: str = ""  # optional cookies file for restricted videos
    cache_dir: str = "source_videos"  # folder inside paths.data_dir
    reuse_existing: bool = True  # skip downloading a video already on disk

    def validate(self) -> None:
        _require(144 <= self.max_height <= 4320, "downloads.max_height must be 144-4320.")
        _require(
            self.merge_format in ("mp4", "mkv", "webm"),
            "downloads.merge_format must be mp4, mkv, or webm.",
        )
        _require(self.retries >= 0, "downloads.retries cannot be negative.")
        _require(self.socket_timeout_seconds > 0, "downloads.socket_timeout_seconds must be positive.")
        _require(bool(self.cache_dir.strip()), "downloads.cache_dir cannot be empty.")

@dataclass
class PathConfig:
    """Where files are read from and written to."""

    output_dir: str = "output_clips"
    temp_dir: str = "temp"
    data_dir: str = "data"
    log_dir: str = "logs"
    cleanup_temp: bool = True
    overwrite_existing: bool = False

    @property
    def output_dir_path(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def temp_dir_path(self) -> Path:
        return Path(self.temp_dir).expanduser()

    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir).expanduser()

    def ensure_directories(self) -> None:
        """Create all configured directories if they do not exist."""
        for path in (
            self.output_dir_path,
            self.temp_dir_path,
            self.data_dir_path,
            self.log_dir_path,
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(f"Cannot create directory '{path}': {exc}") from exc

    def validate(self) -> None:
        for name in ("output_dir", "temp_dir", "data_dir", "log_dir"):
            _require(bool(getattr(self, name).strip()), f"paths.{name} cannot be empty.")
        _require(
            self.output_dir_path.resolve() != self.temp_dir_path.resolve(),
            "paths.output_dir and paths.temp_dir must be different folders.",
        )


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
def _build_section(cls: Type[T], data: Any, section_name: str) -> T:
    """Build one dataclass section from a dict, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{section_name}' must be an object.")
    valid = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in '{section_name}': {', '.join(sorted(unknown))}"
        )
    return cls(**data)


@dataclass
class AppConfig:
    """All application settings."""

    ai: AIConfig = field(default_factory=AIConfig)
    durations: DurationConfig = field(default_factory=DurationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    vertical: VerticalConfig = field(default_factory=VerticalConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    downloads: DownloadConfig = field(default_factory=DownloadConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def validate(self) -> None:
        """Validate every section. Raises ConfigError on the first problem."""
        for section in (
            self.ai,
            self.durations,
            self.selection,
            self.scoring_weights,
            self.vertical,
            self.subtitles,
            self.export,
            self.downloads,
            self.paths,
        ):
            section.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict of the settings. Never contains API keys."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the config to a JSON file."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot write config file '{path}': {exc}") from exc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build a config from a dict (missing sections use defaults)."""
        if not isinstance(data, dict):
            raise ConfigError("Config file must contain a JSON object.")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown config section(s): {', '.join(sorted(unknown))}")
        return cls(
            ai=_build_section(AIConfig, data.get("ai"), "ai"),
            durations=_build_section(DurationConfig, data.get("durations"), "durations"),
            selection=_build_section(SelectionConfig, data.get("selection"), "selection"),
            scoring_weights=_build_section(
                ScoringWeights, data.get("scoring_weights"), "scoring_weights"
            ),
            vertical=_build_section(VerticalConfig, data.get("vertical"), "vertical"),
            subtitles=_build_section(SubtitleConfig, data.get("subtitles"), "subtitles"),
            export=_build_section(ExportConfig, data.get("export"), "export"),
            downloads=_build_section(DownloadConfig, data.get("downloads"), "downloads"),
            paths=_build_section(PathConfig, data.get("paths"), "paths"),
        )


def _load_dotenv_if_available() -> None:
    """Load a .env file if python-dotenv is installed. Missing package is fine."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env loading.")
        return
    load_dotenv(override=False)


def _apply_env_overrides(config: AppConfig) -> None:
    """Apply optional SHORTS_* environment variable overrides."""
    overrides = {
        "SHORTS_AI_PROVIDER": lambda v: setattr(config.ai, "provider", v),
        "SHORTS_AI_MODEL": lambda v: setattr(config.ai, "model", v),
        "SHORTS_AI_BASE_URL": lambda v: setattr(config.ai, "base_url", v),
        "SHORTS_OUTPUT_DIR": lambda v: setattr(config.paths, "output_dir", v),
    }
    for name, apply in overrides.items():
        value = os.environ.get(name, "").strip()
        if value:
            apply(value)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load, override, and validate the configuration.

    Args:
        path: Optional JSON config file. If None, defaults are used.

    Raises:
        ConfigError: if the file is unreadable, malformed, or invalid.
    """
    _load_dotenv_if_available()

    if path is None:
        config = AppConfig()
    else:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file '{path}' is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Cannot read config file '{path}': {exc}") from exc
        config = AppConfig.from_dict(raw)

    _apply_env_overrides(config)
    config.validate()
    return config