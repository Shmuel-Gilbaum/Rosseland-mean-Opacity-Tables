"""Writing a table pair out and reading it back, in all three formats."""
import os

import numpy as np
import pytest

import rm_tables
from rm_tables import io


@pytest.fixture(scope="module")
def tables():
    return rm_tables.build(n_T=40, n_R=25)


@pytest.mark.parametrize("ext", [".npz", ".txt", ".dat", ".h5"])
def test_a_round_trip_keeps_the_numbers(tables, tmp_path_factory, ext):
    if ext == ".h5":
        pytest.importorskip("h5py")
    p = str(tmp_path_factory.mktemp("io") / f"t{ext}")
    tables.save(p)
    back = rm_tables.load(p)
    for name in ("cold", "hot"):
        want = getattr(tables, name).astype(np.float32).astype(float)
        assert np.abs(getattr(back, name) - want).max() < 1e-5, name
    assert np.allclose(back.cold_log_T, tables.cold_log_T)
    assert np.allclose(back.hot_log_R, tables.hot_log_R)
    assert back.split_log_T == tables.split_log_T


@pytest.mark.parametrize("ext", [".npz", ".txt", ".h5"])
def test_the_provenance_survives(tables, tmp_path_factory, ext):
    if ext == ".h5":
        pytest.importorskip("h5py")
    p = str(tmp_path_factory.mktemp("io") / f"t{ext}")
    tables.save(p)
    got = rm_tables.load(p).provenance
    assert got["cold"] == tables.provenance["cold"]
    assert got["units"] == "cm^2/g"
    assert "Semenov" in got["cold_reference"]
    assert float(got["Z"]) == pytest.approx(tables.provenance["Z"])
    assert int(got["n_T"]) == tables.provenance["n_T"]


def test_the_text_file_is_readable_by_a_person(tables, tmp_path):
    p = str(tmp_path / "t.txt")
    tables.save(p)
    head = open(p).read(2000)
    assert "log10 kappa in cm^2/g" in head
    assert "cold_reference = Semenov" in head
    assert "R = rho / (T / 1e6)**3" in head


def test_the_text_file_has_no_binary_and_can_be_parsed_by_anything(tables,
                                                                   tmp_path):
    p = str(tmp_path / "t.txt")
    tables.save(p)
    rows = [l for l in open(p) if l.strip() and not l.startswith("#")]
    name, nT, nR = rows[0].split()
    assert name == "cold"
    assert int(nT) == tables.cold.shape[0]
    assert int(nR) == tables.cold.shape[1]
    assert len(np.fromstring(rows[1], sep=" ")) == int(nT)


def test_an_unknown_extension_says_what_is_supported(tables, tmp_path):
    with pytest.raises(ValueError) as e:
        tables.save(str(tmp_path / "t.wat"))
    assert ".npz" in str(e.value) and ".h5" in str(e.value)


def test_the_format_can_be_forced_against_the_extension(tables, tmp_path):
    p = str(tmp_path / "no_extension_here")
    tables.save(p, fmt="text")
    back = rm_tables.load(p, fmt="text")
    assert back.cold.shape == tables.cold.shape


def test_text_is_larger_than_numpy_and_that_is_the_trade(tables, tmp_path):
    a, b = str(tmp_path / "t.npz"), str(tmp_path / "t.txt")
    tables.save(a)
    tables.save(b)
    assert os.path.getsize(b) > os.path.getsize(a)


@pytest.mark.parametrize("ext", [".npz", ".txt", ".h5"])
@pytest.mark.parametrize("split", [False, True])
def test_a_single_grid_survives_every_format(tmp_path_factory, ext, split):
    """The writers were built when every table came in pairs. A single grid has
    no hot half, and writing None would either crash or invent an empty array."""
    if ext == ".h5":
        pytest.importorskip("h5py")
    t = rm_tables.build(n_T=30, n_R=20, split=split)
    assert t.is_split is split
    p = str(tmp_path_factory.mktemp("io") / f"t{ext}")
    t.save(p)
    back = rm_tables.load(p)
    assert back.is_split is split
    if not split:
        assert back.hot is None
    assert back.kappa(3000.0, 1e-14) == pytest.approx(
        t.kappa(3000.0, 1e-14), rel=1e-5)


def test_reading_an_opacity_from_a_single_grid_works():
    """`which` returns all False on a single grid, and the lookup must handle
    that rather than reaching for a hot half that is not there."""
    t = rm_tables.build(n_T=30, n_R=20, split=False)
    assert t.kappa(3000.0, 1e-14) > 0.0
    many = t.kappa(np.array([300.0, 3000.0, 1e5]), 1e-14)
    assert many.shape == (3,)
    assert np.isfinite(many).all()
