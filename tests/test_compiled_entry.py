"""The opacity must be reachable from compiled code.

A solver whose residual is compiled cannot call a Python object, so without
this the only route was building a table and fitting an interpolant, which
resamples the sources and costs accuracy.
"""
import numpy as np
import pytest
from numba import njit

import rm_tables

COLD = ["semenov", "ferguson"]


@pytest.mark.parametrize("cold", COLD)
def test_the_compiled_form_agrees_bit_for_bit(cold):
    """A second implementation that merely agrees closely would be a second
    source of truth. These share the same kernel, so they must be identical."""
    k = rm_tables.opacity(cold=cold)
    kc = k.compiled()
    floor = 10.0 ** rm_tables.coverage._BY_NAME[cold].log_T[0]
    T = 10.0 ** np.linspace(np.log10(floor * 1.01), 6.5, 60)
    rho = 10.0 ** np.linspace(-18.0, -7.0, 60) * (T / 1e6) ** 3
    want = k(T, rho)
    got = np.array([kc(float(a), float(b)) for a, b in zip(T, rho)])
    assert np.array_equal(got, want)


@pytest.mark.parametrize("cold", COLD)
def test_it_is_callable_from_inside_a_compiled_function(cold):
    """The whole point. Calling the object itself here fails to compile."""
    kappa = rm_tables.opacity(cold=cold).compiled()

    @njit
    def optical_depth(T, rho, height):
        return kappa(T, rho) * rho * height

    v = optical_depth(3000.0, 1e-14, 1e13)
    assert np.isfinite(v) and v > 0


def test_the_object_itself_still_cannot_be_compiled():
    """If this ever starts passing, the method has stopped earning its place."""
    k = rm_tables.opacity()

    @njit
    def use(T, rho):
        return k(T, rho)

    with pytest.raises(Exception):
        use(3000.0, 1e-14)


def test_a_compiled_loop_gives_the_same_answer_as_the_array_call():
    """The array path and the scalar kernel must not drift apart."""
    k = rm_tables.opacity()
    kappa = k.compiled()

    @njit
    def sweep(Ts, rhos, out):
        for i in range(Ts.size):
            out[i] = kappa(Ts[i], rhos[i])
        return out

    T = 10.0 ** np.linspace(1.0, 6.0, 200)
    rho = 10.0 ** np.linspace(-16.0, -8.0, 200) * (T / 1e6) ** 3
    got = sweep(T, rho, np.empty(200))
    assert np.array_equal(got, k(T, rho))
