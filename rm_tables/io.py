"""Writing a table pair to disk, and reading one back.

Three formats, chosen by the file extension:

    .npz    compressed NumPy archive. Compact, Python only.
    .txt    plain text with a readable header. Any language, any editor.
    .h5     HDF5. Any scientific language, needs `h5py` installed.

Each carries the provenance alongside the numbers, so a file found later can be
traced to the papers and the settings that produced it.
"""
import json
import os

import numpy as np

__all__ = ["save", "load", "FORMATS"]

FORMATS = {".npz": "numpy", ".txt": "text", ".dat": "text", ".h5": "hdf5",
           ".hdf5": "hdf5"}
"""File extension to format. `save` and `load` pick from the path."""

_COLD = ("cold_log_T", "cold_log_R", "cold")
_HOT = ("hot_log_T", "hot_log_R", "hot")
_ARRAYS = _COLD + _HOT


def _format_of(path, fmt=None):
    if fmt is not None:
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        raise ValueError(
            f"cannot tell the format from {ext!r}. Use one of "
            f"{sorted(FORMATS)}, or pass fmt=")
    return FORMATS[ext]


def save(tables, path, fmt=None):
    """Write a table pair.

    Parameters
    ----------
    tables : Tables
        What `rm_tables.build` returned.
    path : str
        Destination. Its extension picks the format unless `fmt` says
        otherwise: ``.npz``, ``.txt``, ``.dat``, ``.h5`` or ``.hdf5``.
    fmt : {'numpy', 'text', 'hdf5'}, optional
        Overrides the extension.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    ImportError
        If HDF5 is asked for and `h5py` is not installed.
    """
    kind = _format_of(path, fmt)
    if kind == "numpy":
        _save_npz(tables, path)
    elif kind == "text":
        _save_text(tables, path)
    else:
        _save_hdf5(tables, path)


def load(path, fmt=None):
    """Read a table pair written by `save`.

    Parameters
    ----------
    path : str
        The file. Its extension picks the format unless `fmt` says otherwise.
    fmt : {'numpy', 'text', 'hdf5'}, optional
        Overrides the extension.

    Returns
    -------
    Tables
    """
    from .tables import Tables
    kind = _format_of(path, fmt)
    if kind == "numpy":
        parts = _load_npz(path)
    elif kind == "text":
        parts = _load_text(path)
    else:
        parts = _load_hdf5(path)
    arrays, split, prov = parts
    return Tables(arrays["cold_log_T"], arrays["cold_log_R"], arrays["cold"],
                  arrays["hot_log_T"], arrays["hot_log_R"], arrays["hot"],
                  split, prov)


# --- NumPy ------------------------------------------------------------------

def _save_npz(t, path):
    names = _ARRAYS if t.is_split else _COLD
    np.savez_compressed(
        path, split_log_T=np.array([t.split_log_T]),
        is_split=np.array([t.is_split]),
        provenance=np.array([json.dumps(t.provenance)]),
        **{n: (getattr(t, n).astype(np.float32) if n in ("cold", "hot")
               else getattr(t, n)) for n in names})


def _load_npz(path):
    d = np.load(path)
    arrays = {n: (d[n].astype(float) if n in d else None) for n in _ARRAYS}
    return arrays, float(d["split_log_T"][0]), json.loads(str(d["provenance"][0]))


# --- plain text -------------------------------------------------------------

def _save_text(t, path):
    """One header block, then each table as rows of temperature by density.

    Written so a person can read the provenance and a Fortran program can skip
    the commented header and read the numbers.
    """
    with open(path, "w") as f:
        f.write("# Rosseland mean opacity, log10 kappa in cm^2/g\n")
        f.write("#\n")
        for k in sorted(t.provenance):
            f.write(f"# {k} = {t.provenance[k]}\n")
        f.write(f"# split_log_T = {t.split_log_T}\n")
        f.write("#\n")
        f.write("# Two tables follow. Each begins with a line giving its name\n")
        f.write("# and shape, then its log10 T axis, then its log10 R axis,\n")
        f.write("# then one row of log10 kappa per temperature.\n")
        f.write("# R = rho / (T / 1e6)**3 in g/cm^3.\n")
        for name in (("cold", "hot") if t.is_split else ("cold",)):
            lt = getattr(t, f"{name}_log_T")
            lr = getattr(t, f"{name}_log_R")
            block = getattr(t, name)
            f.write(f"\n{name} {lt.size} {lr.size}\n")
            np.savetxt(f, lt[None, :], fmt="%.8f")
            np.savetxt(f, lr[None, :], fmt="%.8f")
            np.savetxt(f, block, fmt="%.6f")


def _load_text(path):
    prov = {}
    split = None
    with open(path) as f:
        lines = f.read().splitlines()
    body = []
    for line in lines:
        if line.startswith("#"):
            if " = " in line:
                k, v = line[1:].split(" = ", 1)
                k, v = k.strip(), v.strip()
                if k == "split_log_T":
                    split = float(v)
                else:
                    prov[k] = _coerce(v)
        elif line.strip():
            body.append(line)

    arrays = {}
    i = 0
    while i < len(body):
        name, nT, nR = body[i].split()
        nT, nR = int(nT), int(nR)
        arrays[f"{name}_log_T"] = np.fromstring(body[i + 1], sep=" ")
        arrays[f"{name}_log_R"] = np.fromstring(body[i + 2], sep=" ")
        arrays[name] = np.array([np.fromstring(r, sep=" ")
                                 for r in body[i + 3:i + 3 + nT]])
        i += 3 + nT
    for n in _ARRAYS:
        arrays.setdefault(n, None)
    return arrays, split, prov


def _coerce(v):
    """Text back to the type it was written from."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.startswith("(") and v.endswith(")"):
        return tuple(_coerce(p.strip()) for p in v[1:-1].split(",") if p.strip())
    return v


# --- HDF5 -------------------------------------------------------------------

def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "HDF5 output needs h5py. Install it with "
            "'pip install rm-tables[hdf5]' or 'pip install h5py'.") from e
    return h5py


def _save_hdf5(t, path):
    h5py = _h5py()
    with h5py.File(path, "w") as f:
        for n in (_ARRAYS if t.is_split else _COLD):
            a = getattr(t, n)
            f.create_dataset(n, data=(a.astype(np.float32)
                                      if n in ("cold", "hot") else a),
                             compression="gzip")
        f.attrs["split_log_T"] = t.split_log_T
        f.attrs["units"] = "cm^2/g"
        f.attrs["quantity"] = "log10 Rosseland mean opacity"
        for k, v in t.provenance.items():
            f.attrs[k] = v if not isinstance(v, (tuple, list)) else list(v)


def _load_hdf5(path):
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        arrays = {n: (np.array(f[n], dtype=float) if n in f else None)
                  for n in _ARRAYS}
        split = float(f.attrs["split_log_T"])
        prov = {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in f.attrs.items()
                if k not in ("split_log_T", "quantity")}
    return arrays, split, prov
