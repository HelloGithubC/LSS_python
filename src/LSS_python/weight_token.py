"""Canonical mark-method and weight tokens, using only the standard library.

Methods are ``n<k>`` (nearest-neighbor density), ``gn<k>`` (the existing
three-component density gradient), and ``r<radius>`` (fixed-support density).
Weight tokens append ``_weipow<a1>[_<a2>...]`` to one method token. Powers
are ordered, and their evaluation belongs to measurement callers.

Canonical methods use the existing default scientific conventions: include
self, use adaptive h for nearest neighbors, and use h = radius / 2 for the
radius method. Runtime thread and chunk settings do not change the token.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import re
import sys


DEFAULT_MARK = "n30"
_COUNT_PATTERN = re.compile(r"(gn|n)([1-9][0-9]*)")
_RADIUS_PATTERN = re.compile(r"r([0-9]+(?:\.[0-9]+)?)")
_POWER_PATTERN = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_POWER_SEPARATOR = "_weipow"

__all__ = [
    "DEFAULT_MARK", "MarkMethod", "MCFWeightToken", "make_mark_token",
    "parse_mark_token", "normalize_weight_power", "make_weight_token",
    "make_joint_weight_token", "parse_weight_token",
    "add_measurement_weight_arguments", "resolve_measurement_weights",
]


def _decimal(value: object, label: str) -> Decimal:
    """Validate a finite decimal without importing a numerical library."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a real number, got bool")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, (int, float, str)):
        try:
            number = Decimal(str(value).strip())
        except InvalidOperation as error:
            raise ValueError(f"invalid {label}: {value!r}") from error
    else:
        raise ValueError(f"unsupported {label} type: {type(value).__name__}")
    if not number.is_finite():
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _decimal_text(number: Decimal) -> str:
    """Format without scientific notation or context-dependent rounding."""
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def normalize_weight_power(value: object) -> str:
    """Normalize finite powers: -1.0 -> -1, +1 -> 1, and 1e-3 -> 0.001."""
    return _decimal_text(_decimal(value, "weight power"))


@dataclass(frozen=True)
class MarkMethod:
    """A canonical calculation method with an integer count or decimal radius."""

    kind: str
    value: int | Decimal
    token: str


def make_mark_token(kind: str, value: object) -> str:
    """Build one method token, normalizing its positive count or radius."""
    if kind not in ("n", "gn", "r"):
        raise ValueError(f"unknown mark kind {kind!r}; expected n, gn, or r")
    number = _decimal(value, "radius" if kind == "r" else "neighbor count")
    if number <= 0:
        raise ValueError("mark parameter must be positive")
    if kind != "r":
        if number != number.to_integral_value():
            raise ValueError("neighbor count must be an integer")
        return f"{kind}{int(number)}"
    return f"r{_decimal_text(number)}"


def parse_mark_token(token: str) -> MarkMethod:
    """Strictly parse n<k>, gn<k>, or r<radius>; reject alternate spellings."""
    if not isinstance(token, str) or not token:
        raise ValueError("method token must be a non-empty string")
    count = _COUNT_PATTERN.fullmatch(token)
    if count:
        return MarkMethod(count[1], int(count[2]), token)
    radius = _RADIUS_PATTERN.fullmatch(token)
    if radius:
        value = Decimal(radius[1])
        canonical = make_mark_token("r", value)
        if canonical != token:
            raise ValueError(f"method token is not canonical: {token!r} (expected {canonical!r})")
        return MarkMethod("r", value, token)
    raise ValueError(f"invalid method token {token!r}; expected n<k>, gn<k>, or r<radius>")


def make_weight_token(weight_power: object, mark: str = DEFAULT_MARK) -> str:
    """Build <method>_weipow<alpha>; the default preserves n30 token spelling."""
    method = parse_mark_token(mark)
    return f"{method.token}{_POWER_SEPARATOR}{normalize_weight_power(weight_power)}"


def make_joint_weight_token(weight_powers: object, mark: str = DEFAULT_MARK) -> str:
    """Build an ordered same-method power sequence without sorting or deduping."""
    method = parse_mark_token(mark)
    if isinstance(weight_powers, (str, bytes)) or not isinstance(weight_powers, (list, tuple)):
        raise ValueError("weight powers must be an ordered list or tuple")
    if not weight_powers:
        raise ValueError("at least one weight power is required")
    return f"{method.token}{_POWER_SEPARATOR}" + "_".join(
        normalize_weight_power(power) for power in weight_powers
    )


@dataclass(frozen=True)
class MCFWeightToken:
    """Parsed weight identity, preserving the existing mark/powers/token fields."""

    mark: str
    powers: tuple[str, ...]
    token: str

    @property
    def method(self) -> MarkMethod:
        return parse_mark_token(self.mark)

    @property
    def power(self) -> str:
        if len(self.powers) != 1:
            raise ValueError(f"joint token has {len(self.powers)} powers: {self.token}")
        return self.powers[0]

    @property
    def is_joint(self) -> bool:
        return len(self.powers) > 1


def parse_weight_token(token: str) -> MCFWeightToken:
    """Validate a canonical single or ordered joint weight token."""
    if not isinstance(token, str) or not token:
        raise ValueError("weight token must be a non-empty string")
    mark, separator, rest = token.partition(_POWER_SEPARATOR)
    if not separator:
        raise ValueError(f"weight token requires {_POWER_SEPARATOR!r}: {token!r}")
    method = parse_mark_token(mark)
    parts = rest.split("_")
    if any(_POWER_PATTERN.fullmatch(part) is None for part in parts):
        raise ValueError(f"malformed weight power in token: {token!r}")
    powers = tuple(normalize_weight_power(part) for part in parts)
    canonical = make_joint_weight_token(powers, mark=method.token)
    if canonical != token:
        raise ValueError(f"weight token is not canonical: {token!r} (expected {canonical!r})")
    return MCFWeightToken(method.token, powers, canonical)


def add_measurement_weight_arguments(parser: argparse.ArgumentParser) -> None:
    """Register one shared method/power/full-token contract for measurements."""
    parser.add_argument("--weight-method", default=None,
                        help="Raw mark method: n<k>, gn<k>, or r<radius> (default: n30; requires --with_weight)")
    parser.add_argument("--weight-power", "-weipow", dest="weight_pow", default=None,
                        help="Apply this power once to the raw data mark (default: 1)")
    parser.add_argument("--weight-token", default=None,
                        help="Single canonical <method>_weipow<alpha>; requires --with_weight")


def resolve_measurement_weights(args, parser: argparse.ArgumentParser):
    """Resolve defaults and reject conflicting identities before catalog I/O."""
    method_arg, power_arg, token_arg = args.weight_method, args.weight_pow, args.weight_token
    args.weight_method_explicit = method_arg is not None or token_arg is not None
    try:
        if not args.with_weight:
            if args.weight_method_explicit:
                raise ValueError("--weight-method/--weight-token requires --with_weight")
            args.set_weight_to_one = False
            args.weight_method, args.weight_pow = DEFAULT_MARK, 1.0
            args.weight_power_text, args.weight_token = "1", None
            return args
        if token_arg is not None:
            parsed = parse_weight_token(token_arg)
            if parsed.is_joint:
                raise ValueError("one measurement accepts one power; joint tokens describe ordered result sets")
            if method_arg is not None and parse_mark_token(method_arg).token != parsed.mark:
                raise ValueError("--weight-method conflicts with --weight-token")
            if power_arg is not None and normalize_weight_power(power_arg) != parsed.power:
                raise ValueError("--weight-power conflicts with --weight-token")
        else:
            method = parse_mark_token(DEFAULT_MARK if method_arg is None else method_arg)
            power = normalize_weight_power(1 if power_arg is None else power_arg)
            parsed = parse_weight_token(make_weight_token(power, mark=method.token))
        numeric_power = float(parsed.power)
        if not math.isfinite(numeric_power) or (numeric_power == 0 and Decimal(parsed.power) != 0):
            raise ValueError("measurement power is outside the finite nonzero float range")
        args.weight_method, args.weight_pow = parsed.mark, numeric_power
        args.weight_power_text, args.weight_token = parsed.power, parsed.token
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    return args


def build_parser() -> argparse.ArgumentParser:
    """Build the shared token CLI without importing the numerical stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    method = subparsers.add_parser("method-token", help="Build a canonical method token.")
    method.add_argument("kind", choices=("n", "gn", "r"))
    method.add_argument("value", help="Positive neighbor count or support radius")
    parse_method = subparsers.add_parser("parse-method", help="Parse a canonical method token.")
    parse_method.add_argument("token")
    single = subparsers.add_parser("weight-token", help="Build one <method>_weipow<alpha> token.")
    single.add_argument("power", help="Weight power, e.g. -1, 1, -0.3")
    single.add_argument("--method", default=DEFAULT_MARK, help="Canonical method token (default: n30)")
    joint = subparsers.add_parser("joint-weight-token", help="Build ordered powers for one method.")
    joint.add_argument("powers", nargs="+", help="Ordered weight powers")
    joint.add_argument("--method", default=DEFAULT_MARK, help="Canonical method token (default: n30)")
    parse = subparsers.add_parser("parse", help="Parse a canonical weight token as JSON.")
    parse.add_argument("token")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the token CLI; no arguments prints help and succeeds."""
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    try:
        if args.command == "method-token":
            print(make_mark_token(args.kind, args.value))
        elif args.command == "parse-method":
            method = parse_mark_token(args.token)
            print(json.dumps({"kind": method.kind, "value": str(method.value), "token": method.token}))
        elif args.command == "weight-token":
            print(make_weight_token(args.power, mark=args.method))
        elif args.command == "joint-weight-token":
            print(make_joint_weight_token(args.powers, mark=args.method))
        else:
            parsed = parse_weight_token(args.token)
            print(json.dumps({"mark": parsed.mark, "powers": list(parsed.powers), "token": parsed.token}))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
