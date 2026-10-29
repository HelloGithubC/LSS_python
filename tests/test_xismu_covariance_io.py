"""Round-trip and validation tests for compact covariance sample storage."""

import joblib
import numpy as np
import pytest

from LSS_python.xismu import XismuCovarianceIO, xismu


def make_sample(index=0, dtype=np.float32):
    s, mu = np.meshgrid(np.arange(1, 5, dtype=dtype),
                        np.arange(1, 4, dtype=dtype) / 4, indexing="ij")
    sample = xismu(smax=4, sbin=4, mubin=3, DDnorm=2.0 + index,
                   DRnorm=3.0 + index, RRnorm=4.0, S=s, Mu=mu,
                   DD=np.full((4, 3), 10 + index, dtype=dtype),
                   DR=np.full((4, 3), 5 + index, dtype=dtype),
                   RR=np.full((4, 3), 4, dtype=dtype),
                   sum_wd=12.5 + index, sum_wr=20.5,
                   sum_wd2=14.5 + index, sum_wr2=22.5)
    sample.custom_metadata = {"label": f"sample-{index}", "values": [1, 2]}
    return sample


def assert_state_equal(expected, actual):
    assert type(actual) is type(expected)
    assert actual.__dict__.keys() == expected.__dict__.keys()
    for name, value in expected.__dict__.items():
        result = actual.__dict__[name]
        if isinstance(value, np.ndarray):
            assert isinstance(result, np.ndarray), name
            assert result.dtype == value.dtype, name
            np.testing.assert_array_equal(result, value, err_msg=name)
        else:
            assert result == value, name


@pytest.mark.parametrize("kind", ["jk", "subsample"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_round_trip_preserves_sample_state_and_results(tmp_path, kind, dtype):
    a, b = make_sample(0, dtype), make_sample(1, dtype)
    if kind == "jk":
        b.RR = b.RR * 2
        b.RRnorm = 8.0
    a.get_xis()  # Exercise a populated cache; b retains an empty cache.
    samples = [a, None, b]
    filename = tmp_path / "samples.joblib"
    XismuCovarianceIO.save(filename, samples, kind)
    result = XismuCovarianceIO.load(filename)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == object
    assert result[1] is None
    for original, restored in ((a, result[0]), (b, result[2])):
        assert_state_equal(original, restored)
        np.testing.assert_array_equal(original.get_xis(), restored.get_xis())
        expected = original.intximu(smin=1, smax=4, mumin=0, mumax=1,
                                    remove_last_one=False)
        actual = restored.intximu(smin=1, smax=4, mumin=0, mumax=1,
                                  remove_last_one=False)
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x, y)


def test_load_rejects_legacy_object_array(tmp_path):
    samples = np.array([make_sample(), None, make_sample(1)], dtype=object)
    filename = tmp_path / "legacy.pkl"
    joblib.dump(samples, filename)
    with pytest.raises(ValueError, match="migrate_xismu_covariance"):
        XismuCovarianceIO.load(filename)


def test_refuses_to_overwrite(tmp_path):
    filename = tmp_path / "existing"
    filename.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        XismuCovarianceIO.save(filename, [make_sample()], "jk")
    assert filename.read_bytes() == b"original"


@pytest.mark.parametrize("samples,kind", [
    ([], "jk"), ([None, None], "jk"), ([make_sample()], "invalid"),
    ([[make_sample()]], "jk"), ([make_sample(), "invalid"], "jk"),
])
def test_rejects_invalid_input(tmp_path, samples, kind):
    filename = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError)):
        XismuCovarianceIO.save(filename, samples, kind)
    assert not filename.exists()


@pytest.mark.parametrize("mutation", ["S", "Mu", "shape", "RR", "RRnorm"])
def test_rejects_inconsistent_shared_data(tmp_path, mutation):
    first, second = make_sample(), make_sample(1)
    if mutation in ("S", "Mu"):
        getattr(second, mutation)[0, 0] += 0.25
    elif mutation == "shape":
        second.DD = second.DD[:-1]
    elif mutation == "RR":
        second.RR[0, 0] += 0.25
    else:
        second.RRnorm += 1
    with pytest.raises(ValueError):
        XismuCovarianceIO.save(tmp_path / "bad", [first, second], "subsample")


def test_rejects_unknown_file(tmp_path):
    filename = tmp_path / "unknown.joblib"
    joblib.dump({"unrelated": True}, filename)
    with pytest.raises((ValueError, TypeError)):
        XismuCovarianceIO.load(filename)


@pytest.mark.parametrize("field", ["S", "Mu", "RR"])
def test_rejects_different_shared_array_dtype(tmp_path, field):
    first, second = make_sample(), make_sample(1)
    setattr(second, field, getattr(second, field).astype(np.float64))
    with pytest.raises(ValueError):
        XismuCovarianceIO.save(tmp_path / "bad-dtype", [first, second], "subsample")


@pytest.mark.parametrize("damage", ["version", "missing_samples", "wrong_state", "missing_shared_rr"])
def test_rejects_invalid_compact_payload(tmp_path, damage):
    valid = tmp_path / "valid"
    XismuCovarianceIO.save(valid, [make_sample()], "subsample")
    payload = joblib.load(valid)
    if damage == "version":
        payload["version"] = 999999
    elif damage == "missing_samples":
        del payload["samples"]
    elif damage == "wrong_state":
        payload["samples"][0] = {"sbin": 4}
    else:
        del payload["RR"]
    broken = tmp_path / "broken"
    joblib.dump(payload, broken)
    with pytest.raises(ValueError):
        XismuCovarianceIO.load(broken)


def test_failed_serialization_removes_partial_file(tmp_path, monkeypatch):
    filename = tmp_path / "partial"

    def fail_dump(payload, stream):
        stream.write(b"partial output")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(joblib, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="serialization failed"):
        XismuCovarianceIO.save(filename, [make_sample()], "jk")
    assert not filename.exists()


def test_overwrite_replaces_existing_file(tmp_path):
    filename = tmp_path / "existing"
    filename.write_bytes(b"original")
    sample = make_sample()
    XismuCovarianceIO.save(filename, [sample], "jk", overwrite=True)
    assert_state_equal(sample, XismuCovarianceIO.load(filename)[0])
    assert list(tmp_path.iterdir()) == [filename]


def test_failed_overwrite_preserves_original_and_cleans_temporary_file(tmp_path, monkeypatch):
    filename = tmp_path / "existing"
    filename.write_bytes(b"original")

    def fail_dump(payload, stream):
        stream.write(b"partial output")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(joblib, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="serialization failed"):
        XismuCovarianceIO.save(filename, [make_sample()], "jk", overwrite=True)
    assert filename.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [filename]
