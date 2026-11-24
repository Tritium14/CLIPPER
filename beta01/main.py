"""Application entry point.

This module only coordinates other modules. It contains no analysis,
video-processing, or configuration logic of its own.

Usage:
    python main.py --write-default-config
    python main.py --check
    python main.py --show-config
    python main.py --test-ai
    python main.py --video "C:\\path\\to\\video.mp4"
    python main.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
    python main.py --transcript "C:\\path\\to\\transcript.srt"
    python main.py --video "URL_OR_PATH" --transcript "transcript.txt" --detect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from analysis.analysis_errors import CandidateStoreError, HighlightDetectionError, LLMError
from analysis.candidate_report import format_candidate_table
from analysis.candidate_store import candidate_file_path, save_candidates
from analysis.highlight_detector import detect_highlights
from analysis.llm_client import LLMClient
from config import AppConfig, ConfigError, load_config
from dependency_checker import all_required_ok, format_report, run_all_checks
from input.input_errors import TranscriptError, VideoInputError
from input.transcript_models import Transcript
from input.transcript_parser import load_transcript
from input.video_source_resolver import resolve_video_source
from logging_setup import setup_logging

logger = logging.getLogger("intelligent_shorts.main")

DEFAULT_CONFIG_PATH = "config.json"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="intelligent_shorts",
        description="AI-powered YouTube to Shorts converter.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the JSON config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write a default config file to --config and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --write-default-config to overwrite an existing file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that Python packages, FFmpeg, and the AI provider setup are usable.",
    )
    parser.add_argument(
        "--test-ai",
        action="store_true",
        help="Send a tiny test request to the configured AI provider and show the reply.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the effective configuration (API keys are never included).",
    )
    parser.add_argument(
        "--video",
        metavar="PATH_OR_URL",
        help="Load a local video file or download a video URL, then show its details.",
    )
    parser.add_argument(
        "--transcript",
        metavar="PATH",
        help="Parse a transcript file (SRT, VTT, or timestamped text) and show a summary. "
        "Combine with --video to also check it against the video length.",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Ask the AI to find highlight candidates in the transcript (needs --transcript).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG-level log messages on the console.",
    )
    return parser


def _write_default_config(path: Path, force: bool) -> int:
    if path.exists() and not force:
        print(f"Config file already exists: {path}\nUse --force to overwrite it.")
        return 1
    AppConfig().save(path)
    print(f"Default config written to: {path}")
    return 0


def _test_ai(config: AppConfig) -> int:
    """Send a small request to the AI provider to confirm the setup works."""
    client = LLMClient(config.ai)
    print(f"Testing {config.ai.provider} with model '{config.ai.resolved_model()}'...")
    try:
        reply = client.chat("You are a connection test.", "Reply with the single word: ok")
    except LLMError as exc:
        logger.error("%s", exc)
        return 5
    print(f"AI provider is working. The model replied: {reply[:200]}")
    return 0


def _run_detection(config: AppConfig, transcript: Transcript, source_name: str, file_stem: str) -> int:
    """Run AI highlight detection, print the candidates, and save them."""
    try:
        result = detect_highlights(transcript, config, source_name=source_name)
    except HighlightDetectionError as exc:
        logger.error("%s", exc)
        return 5

    print()
    print(
        f"AI found {len(result.candidates)} candidate(s) from {result.chunks_total} part(s) "
        f"({result.chunks_failed} failed, {result.rejected_count} suggestion(s) rejected as invalid)."
    )
    for warning in result.warnings:
        print(f"  Warning: {warning}")
    if not result.candidates:
        return 0

    print(format_candidate_table(result.candidates))
    path = candidate_file_path(config.paths.data_dir_path, file_stem)
    try:
        save_candidates(path, result.candidates, source_name)
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 6
    print(f"\nSaved candidates to: {path}")
    return 0


def _inspect_inputs(args: argparse.Namespace, config: AppConfig) -> int:
    """Load the video and/or transcript given on the command line, then optionally detect."""
    resolved = None
    if args.video:
        try:
            resolved = resolve_video_source(args.video, config)
        except VideoInputError as exc:
            logger.error("%s", exc)
            return 3
        print(resolved.describe())

    if not args.transcript:
        return 0

    duration = resolved.info.duration_seconds if resolved else None
    try:
        transcript = load_transcript(args.transcript, video_duration=duration)
    except TranscriptError as exc:
        logger.error("%s", exc)
        return 4
    if resolved:
        print()
    print(transcript.summary())

    if args.detect:
        stem = resolved.path.stem if resolved else (
            transcript.source_path.stem if transcript.source_path else "transcript"
        )
        name = resolved.source_name if resolved else stem
        return _run_detection(config, transcript, name, stem)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the application. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.detect and not args.transcript:
        parser.error("--detect needs --transcript.")
    config_path = Path(args.config)

    if args.write_default_config:
        return _write_default_config(config_path, args.force)

    try:
        config = load_config(config_path if config_path.exists() else None)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    config.paths.ensure_directories()
    setup_logging(
        log_dir=config.paths.log_dir_path,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger.debug("Loaded configuration from %s", config_path if config_path.exists() else "defaults")

    if args.show_config:
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    if args.check:
        results = run_all_checks(config)
        print(format_report(results))
        return 0 if all_required_ok(results) else 1

    if args.test_ai:
        return _test_ai(config)

    if args.video or args.transcript:
        return _inspect_inputs(args, config)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())