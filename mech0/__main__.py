"""python -m mech0 [serve|seed]"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "serve"
    rest = args[1:] if args else []
    if cmd in {"serve", "server", "start"}:
        from .server import main as serve

        return serve(rest)
    if cmd == "seed":
        from .seed import main as seed

        return seed()
    if cmd in {"-h", "--help", "help"}:
        print("usage: python -m mech0 [serve|seed] [--host 127.0.0.1] [--port 8765] [--data-dir .mech0/data]")
        return 0
    print(f"unknown command {cmd!r}; use serve or seed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
