"""python -m echoself [autognosis]"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "autognosis"
    rest = args[1:]
    if cmd in {"autognosis", "self"}:
        from echoself.autognosis.observe import main as autognosis

        return autognosis(rest)
    if cmd in {"-h", "--help", "help"}:
        print("usage: python -m echoself [autognosis] [--remember]")
        return 0
    print(f"unknown command {cmd!r}; use autognosis", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
