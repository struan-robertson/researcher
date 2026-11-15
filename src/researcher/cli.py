import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from researcher.config import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_TIME_LIMIT,
    Settings,
    resolve_settings,
)
from researcher.errors import UserError
from researcher.question import Question, load_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researcher",
        description="Discover literature for a research question.",
        epilog="The research workflow is not implemented yet.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('researcher')}"
    )
    parser.add_argument(
        "target",
        metavar="QUESTION",
        type=Path,
        help="Markdown research brief, or an output directory containing question.md",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        type=Path,
        help="configuration file (default: $XDG_CONFIG_HOME/researcher/config.toml)",
    )
    parser.add_argument(
        "--max-papers",
        metavar="INTEGER",
        type=int,
        help=f"maximum number of papers to report (default: {DEFAULT_MAX_PAPERS})",
    )
    parser.add_argument(
        "--time-limit",
        metavar="DURATION",
        help=f"research time limit such as 45m or 0.5h (default: {DEFAULT_TIME_LIMIT})",
    )
    return parser


def run(question: Question, settings: Settings) -> int:
    """Placeholder for the research workflow, which will receive these inputs."""
    print(f"question ID: {question.question_id}")
    print(f"max papers: {settings.max_papers}")
    print(f"time limit: {settings.time_limit}")
    print("The research workflow is not implemented yet.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    try:
        settings = resolve_settings(
            args.config,
            max_papers=args.max_papers,
            time_limit=args.time_limit,
        )
        question = load_question(args.target)
    except UserError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    return run(question, settings)
