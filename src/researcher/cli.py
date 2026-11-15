import argparse
from collections.abc import Sequence
from importlib.metadata import version


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="researcher",
        description="Discover literature for a research question.",
        epilog="The research workflow is not implemented yet.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('researcher')}"
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0
