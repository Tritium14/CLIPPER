"""Checks that required Python packages and external tools are available."""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

from config import AppConfig

logger = logging.getLogger(__name__)

MIN_PYTHON = (3, 9)


@dataclass
class CheckResult:
    """Outcome of one dependency check."""

    name: str
    ok: bool
    detail: str
    required: bool = True


def check_python_version() -> CheckResult:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info >= MIN_PYTHON
    detail = f"Python {version}"
    if not ok:
        detail += f" (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+)"
    return CheckResult("Python version", ok, detail)


def check_python_package(import_name: str, pip_name: str, required: bool, purpose: str) -> CheckResult:
    """Check whether a Python package can be imported."""
    try:
        importlib.import_module(import_name)
    except ImportError:
        return CheckResult(
            f"Package: {pip_name}",
            False,
            f"missing ({purpose}). Install with: pip install {pip_name}",
            required,
        )
    return CheckResult(f"Package: {pip_name}", True, f"installed ({purpose})", required)


def _run_tool(args: List[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Failed to run %s: %s", args[0], exc)
        return None


def check_executable(name: str) -> CheckResult:
    """Check that an executable (ffmpeg or ffprobe) is on PATH and runs."""
    path = shutil.which(name)
    if not path:
        return CheckResult(
            f"Tool: {name}",
            False,
            "not found on PATH. Install FFmpeg and make sure it is on your PATH.",
        )
    result = _run_tool([path, "-version"])
    if result is None or result.returncode != 0:
        return CheckResult(f"Tool: {name}", False, f"found at {path} but failed to run.")
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else name
    return CheckResult(f"Tool: {name}", True, first_line)


def check_ffmpeg_subtitle_support() -> CheckResult:
    """Check that FFmpeg was built with the filters needed for burned-in subtitles."""
    path = shutil.which("ffmpeg")
    if not path:
        return CheckResult("FFmpeg subtitle filter", False, "ffmpeg not found.")
    result = _run_tool([path, "-hide_banner", "-filters"])
    if result is None or result.returncode != 0:
        return CheckResult("FFmpeg subtitle filter", False, "could not list FFmpeg filters.")
    output = result.stdout or ""
    has_filter = any(
        line.split()[1:2] == ["subtitles"] or line.split()[1:2] == ["ass"]
        for line in output.splitlines()
        if len(line.split()) >= 2
    )
    if has_filter:
        return CheckResult("FFmpeg subtitle filter", True, "subtitles/ass filter available (libass)")
    return CheckResult(
        "FFmpeg subtitle filter",
        False,
        "this FFmpeg build lacks libass. Install a full build (for example the "
        "'gyan.dev full' build on Windows).",
    )


def check_ai_provider(config: AppConfig) -> CheckResult:
    """Check that the AI provider settings are complete (does not contact the server)."""
    ai = config.ai
    model = ai.resolved_model()
    if not model:
        return CheckResult("AI provider", False, f"{ai.provider}: no model set (ai.model).")
    try:
        ai.get_api_key()
    except Exception as exc:  # ConfigError for a missing key
        return CheckResult("AI provider", False, str(exc))
    return CheckResult(
        "AI provider", True, f"{ai.provider} -> {ai.resolved_base_url()} (model: {model})"
    )


def run_all_checks(config: Optional[AppConfig] = None) -> List[CheckResult]:
    """Run every dependency check and return the results."""
    results = [
        check_python_version(),
        check_python_package("requests", "requests", True, "AI provider HTTP calls"),
        check_python_package("yt_dlp", "yt-dlp", False, "downloading videos from URLs"),
        check_python_package("dotenv", "python-dotenv", False, "loading a .env file"),
        check_python_package("pytest", "pytest", False, "running the tests"),
        check_python_package("tkinter", "tkinter (bundled with Python)", False, "the graphical interface"),
        check_executable("ffmpeg"),
        check_executable("ffprobe"),
        check_ffmpeg_subtitle_support(),
    ]
    if config is not None:
        results.append(check_ai_provider(config))
    return results


def all_required_ok(results: List[CheckResult]) -> bool:
    """True when every required check passed."""
    return all(r.ok for r in results if r.required)


def format_report(results: List[CheckResult]) -> str:
    """Format check results as a readable text report."""
    lines = ["Dependency check", "================"]
    for r in results:
        status = "OK  " if r.ok else ("FAIL" if r.required else "WARN")
        lines.append(f"[{status}] {r.name}: {r.detail}")
    lines.append("")
    if all_required_ok(results):
        lines.append("All required checks passed.")
    else:
        lines.append("Some required checks failed. Fix the items marked FAIL.")
    return "\n".join(lines)