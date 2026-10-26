"""Protocol and generator checks for canonical method and weight tokens."""

from decimal import Decimal
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from LSS_python.weight_token import (
    main, make_joint_weight_token, make_mark_token, make_weight_token,
    normalize_weight_power, parse_mark_token, parse_weight_token,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind,value,token", [
    ("n", "30.0", "n30"), ("gn", 40, "gn40"),
    ("r", "10.0", "r10"), ("r", "1e-3", "r0.001"),
])
def test_method_build_and_parse(kind, value, token):
    assert make_mark_token(kind, value) == token
    method = parse_mark_token(token)
    assert method.kind == kind
    assert method.value == Decimal(str(value))
    assert method.token == token


@pytest.mark.parametrize("token", [
    "", "n0", "n-1", "n030", "n30.0", "gn0", "gn030", "gn3.5",
    "r0", "r-1", "r10.0", "r01", "r.5", "r1e1", "rnan", "rinf",
    "N30", "n٣٠", " n30", "n30 ", "n30_weipow1", "other30", None,
])
def test_invalid_or_noncanonical_methods(token):
    with pytest.raises(ValueError):
        parse_mark_token(token)


@pytest.mark.parametrize("kind,value", [
    ("n", 0), ("n", 1.5), ("gn", -1), ("r", 0), ("r", "nan"),
    ("r", float("inf")), ("n", True), ("r", ""), ("unknown", 30),
])
def test_invalid_method_parameters(kind, value):
    with pytest.raises(ValueError):
        make_mark_token(kind, value)


@pytest.mark.parametrize("value,canonical", [
    ("-1.0", "-1"), ("+1", "1"), ("1e-3", "0.001"),
    ("-0.000", "0"), (Decimal("2.500"), "2.5"),
])
def test_power_normalization(value, canonical):
    assert normalize_weight_power(value) == canonical


def test_decimal_precision_does_not_merge_distinct_tokens():
    first = "0.123456789012345678901234567891"
    second = "0.123456789012345678901234567892"
    assert normalize_weight_power(first) == first
    assert make_weight_token(first) != make_weight_token(second)


@pytest.mark.parametrize("value", [True, False, "", "nan", "inf", float("nan"), object()])
def test_invalid_powers(value):
    with pytest.raises(ValueError):
        normalize_weight_power(value)


@pytest.mark.parametrize("mark", ["n30", "n40", "gn30", "r10", "r10.5"])
def test_single_and_ordered_joint_weight_tokens(mark):
    token = make_weight_token("-0.50", mark=mark)
    assert token == mark + "_weipow-0.5"
    parsed = parse_weight_token(token)
    assert parsed.mark == mark and parsed.method == parse_mark_token(mark)
    assert parsed.power == "-0.5" and not parsed.is_joint
    joint = make_joint_weight_token([-1, 0, 1, -1], mark=mark)
    assert joint == mark + "_weipow-1_0_1_-1"
    parsed_joint = parse_weight_token(joint)
    assert parsed_joint.powers == ("-1", "0", "1", "-1")
    assert parsed_joint.is_joint
    with pytest.raises(ValueError, match="joint"):
        _ = parsed_joint.power


def test_legacy_default_text_and_fields():
    assert make_weight_token(-1.0) == "n30_weipow-1"
    assert make_joint_weight_token([-1, 1, -0.3]) == "n30_weipow-1_1_-0.3"
    parsed = parse_weight_token("n30_weipow-1")
    assert (parsed.mark, parsed.powers, parsed.token) == ("n30", ("-1",), "n30_weipow-1")


@pytest.mark.parametrize("powers", [[], (), "-1_1", {-1, 1}, None])
def test_joint_requires_nonempty_ordered_sequence(powers):
    with pytest.raises(ValueError):
        make_joint_weight_token(powers)


@pytest.mark.parametrize("token", [
    "", None, "n30", "n30_weipow", "n30_weipow1_", "n30_weipow_1",
    "n30_weipow1__2", "n30_weipow+1", "n30_weipow1.0", "n30_weipow-0",
    "n30_weipow01", "n30_weipow1e-3", "gn30_weipowNaN", "r10.0_weipow1",
    "n30_weipow1_gn30_weipow1", "n30_weipow1\n",
])
def test_invalid_weight_tokens(token):
    with pytest.raises(ValueError):
        parse_weight_token(token)


def test_token_cli_help_and_commands(capsys):
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out
    assert main(["method-token", "r", "10.0"]) == 0
    assert capsys.readouterr().out.strip() == "r10"
    assert main(["weight-token", "-0.5", "--method", "gn30"]) == 0
    assert capsys.readouterr().out.strip() == "gn30_weipow-0.5"
    assert main(["joint-weight-token", "-1", "0", "1", "--method", "r10"]) == 0
    assert capsys.readouterr().out.strip() == "r10_weipow-1_0_1"
    assert main(["parse", "n30_weipow-1"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "mark": "n30", "powers": ["-1"], "token": "n30_weipow-1",
    }
    assert main(["parse-method", "gn30"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "gn"


def test_token_cli_runs_without_site_packages():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-B", "-S", str(ROOT / "bin/mcf_weight_token.py"),
         "weight-token", "-1", "--method", "r10"],
        env=environment, capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "r10_weipow-1"


@pytest.fixture(scope="module")
def generator():
    return runpy.run_path(str(ROOT / "bin/create_rho"))["main"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method,option", [("n6", "--rho-method"), ("gn6", "--method"), ("r2", "-method")])
def test_generator_dispatch_and_raw_output(generator, tmp_path, method, option, dtype):
    from LSS_python.marks import create_drho, create_rho, create_rho_fix_r

    rng = np.random.default_rng(20261025)
    positions = rng.uniform(0, 12, (40, 3)).astype(dtype)
    positions[:2] = [[0.05, 0.1, 0.2], [11.95, 11.9, 11.8]]
    source = tmp_path / "input.npy"
    np.save(source, np.column_stack((positions, np.full(40, 99, dtype=dtype))))
    output = tmp_path / f"marks_{method}"
    args = [str(source), "12", str(output), option, method, "-n", "1"]
    if method == "r2":
        args += ["--chunk-size", "7"]
        expected_mark = create_rho_fix_r(positions, 12.0, r=2.0, chunk_size=7)
    elif method == "gn6":
        rho, gradient = create_drho(positions, 12.0, k=6)
        expected_mark = np.column_stack((rho, gradient))
    else:
        expected_mark = create_rho(positions, 12.0, k=6)
    assert generator(args) == 0
    actual = np.load(output.with_suffix(output.suffix + ".npy"))
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, np.column_stack((positions, expected_mark)))
    assert actual.shape == (40, 7 if method == "gn6" else 4)


def test_generator_default_and_output_protection(generator, tmp_path, capsys, monkeypatch):
    from LSS_python.marks import create_rho

    source = tmp_path / "input.npy"
    pos = np.random.default_rng(23).uniform(0, 12, (40, 3))
    np.save(source, pos)
    output = tmp_path / "input_n30.npy"
    assert generator([str(source), "12", "-n", "1"]) == 0
    np.testing.assert_array_equal(np.load(output), np.column_stack((pos, create_rho(pos, 12.0))))
    previous = output.read_bytes()
    assert generator([str(tmp_path / "missing.npy"), "12", str(output.with_suffix(""))]) == 0
    assert "already exists" in capsys.readouterr().out
    assert output.read_bytes() == previous
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert generator([str(source), "12", str(output), "-method", "n6", "-n", "1", "--force"]) == 0
    assert output.read_bytes() != previous


@pytest.mark.parametrize("options", [
    ["--method", "n6_weipow-1"], ["--method", "r2.0"],
    ["--method", "n1"], ["--method", "n41"], ["--method", "r6"],
    ["--method", "n6", "--chunk-size", "3"],
    ["--method", "r2", "--chunk-size", "0"], ["-n", "0"],
    ["--ignore-self"], ["--h-fixed=1"], ["--no-adaptive-h"],
])
def test_generator_rejects_unsupported_or_invalid_settings(generator, tmp_path, options):
    source = tmp_path / "input.npy"
    np.save(source, np.random.default_rng(24).uniform(0, 12, (40, 3)))
    with pytest.raises(SystemExit) as error:
        generator([str(source), "12", *options])
    assert error.value.code == 2
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("method", ["n6", "gn6", "r0.5"])
def test_generator_default_name_uses_input_directory_and_method(generator, tmp_path, monkeypatch, method):
    source = tmp_path / "snapshot.v2.npy"
    np.save(source, np.random.default_rng(71).uniform(0, 12, (40, 3)))
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("default name must not prompt"))
    assert generator([str(source), "12", "--method", method, "-n", "1"]) == 0
    output = tmp_path / f"snapshot.v2_{method}.npy"
    assert output.exists()
    assert np.load(output).shape == (40, 7 if method == "gn6" else 4)


@pytest.mark.parametrize("name,method", [
    ("custom_n6", "n6"), ("custom-n6.npy", "n6"), ("n6_extra.npy", "n6"),
    ("custom.n6.npy", "n6"), ("custom_gn6.npy", "gn6"), ("custom_r0.5", "r0.5"),
])
def test_generator_explicit_matching_name_does_not_prompt(generator, tmp_path, monkeypatch, name, method):
    source = tmp_path / "input.npy"
    np.save(source, np.random.default_rng(72).uniform(0, 12, (40, 3)))
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("matching method must not prompt"))
    assert generator([str(source), "12", str(tmp_path / name), "--method", method, "-n", "1"]) == 0
    output = tmp_path / (name if name.endswith(".npy") else name + ".npy")
    assert output.exists()


@pytest.mark.parametrize("reply", ["y", "yes", "Y", " YES "])
def test_generator_explicit_missing_method_requires_affirmative(generator, tmp_path, monkeypatch, capsys, reply):
    source = tmp_path / "input.npy"
    np.save(source, np.random.default_rng(73).uniform(0, 12, (40, 3)))
    prompts = []
    def respond(prompt):
        prompts.append(prompt)
        return reply
    monkeypatch.setattr("builtins.input", respond)
    output = tmp_path / "custom"
    assert generator([str(source), "12", str(output), "--method", "n6", "-n", "1"]) == 0
    assert len(prompts) == 1 and "y or yes" in prompts[0]
    assert "does not contain method token 'n6'" in capsys.readouterr().out
    assert (tmp_path / "custom.npy").exists()


@pytest.mark.parametrize("name,method", [
    ("custom.npy", "n6"), ("custom_n60.npy", "n6"), ("custom_gn6.npy", "n6"),
    ("custom_n6.5.npy", "n6"), ("custom_r2.5.npy", "r2"),
])
@pytest.mark.parametrize("force", [False, True])
def test_generator_cancelled_name_is_checked_before_input_io(generator, tmp_path, monkeypatch, name, method, force):
    directory = tmp_path / method
    directory.mkdir()
    output = directory / name
    previous = b"existing output must survive cancellation"
    if force:
        output.write_bytes(previous)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "no")
    args = [str(tmp_path / "missing.npy"), "12", str(output), "--method", method]
    assert generator(args + (["--force"] if force else [])) == 1
    assert len(prompts) == 1
    if force:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()


@pytest.mark.parametrize("reply", ["", "n", "true", "yep"])
def test_generator_other_replies_cancel(generator, tmp_path, monkeypatch, reply):
    monkeypatch.setattr("builtins.input", lambda prompt: reply)
    assert generator([str(tmp_path / "missing.npy"), "12", str(tmp_path / "custom")]) == 1
    assert not (tmp_path / "custom.npy").exists()


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_generator_no_confirmation_cancels_cleanly(generator, tmp_path, monkeypatch, exception):
    def interrupted(prompt):
        raise exception
    monkeypatch.setattr("builtins.input", interrupted)
    assert generator([str(tmp_path / "missing.npy"), "12", str(tmp_path / "custom")]) == 1
    assert not (tmp_path / "custom.npy").exists()


def test_generator_no_argument_help(generator, capsys):
    assert generator([]) == 0
    assert "gn<k>" in capsys.readouterr().out
