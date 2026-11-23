"""Entry point for ``python -m proofreader`` (the only entry point, D10). See :func:`proofreader.cli.main`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
