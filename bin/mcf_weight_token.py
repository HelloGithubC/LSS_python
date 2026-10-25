#!/usr/bin/env python3
"""Command-line entry point for LSS_python.weight_token.

Import token APIs from LSS_python.weight_token, not from this bin script.
"""

from LSS_python.weight_token import main


if __name__ == "__main__":
    raise SystemExit(main())
