#!/usr/bin/env python3
"""Calculatrice CLI — opérations arithmétiques de base via argparse."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calculator.py",
        description="Perform basic arithmetic operations on two numbers.",
    )
    op = parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--add", action="store_true", help="addition")
    op.add_argument("--sub", action="store_true", help="subtraction")
    op.add_argument("--mul", action="store_true", help="multiplication")
    op.add_argument("--div", action="store_true", help="division")
    parser.add_argument("--float", dest="float_div", action="store_true",
                        help="float division (used with --div)")
    parser.add_argument("--int", dest="int_div", action="store_true",
                        help="integer division (used with --div)")
    parser.add_argument("numbers", nargs="*", help="exactly two numeric values")
    return parser


def _parse_numbers(raw: list[str]) -> list[float]:
    if len(raw) != 2:
        print("Error: You must provide exactly two numbers.", file=sys.stderr)
        sys.exit(1)
    try:
        return [float(x) for x in raw]
    except ValueError:
        print("Error: Both arguments must be numeric.", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    a, b = _parse_numbers(args.numbers)

    if args.add:
        print(f"{a} + {b} = {a + b}")
        return 0
    if args.sub:
        print(f"{a} - {b} = {a - b}")
        return 0
    if args.mul:
        print(f"{a} × {b} = {a * b}")
        return 0

    # --div
    use_float = True
    if args.int_div and args.float_div:
        print("Warning: Both --int and --float provided. Using float division by default.")
    elif args.int_div:
        use_float = False
    elif not args.float_div:
        print("Info: No division mode specified. Using float division by default.")

    if b == 0:
        print("Error: Division by zero is not allowed.", file=sys.stderr)
        return 1

    if use_float:
        print(f"{a} / {b} = {a / b}")
    else:
        print(f"{a} // {b} = {int(a // b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
