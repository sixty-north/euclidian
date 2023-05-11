from math import sqrt

from pytest import approx

from hypothesis import given, assume
from hypothesis.strategies import floats, one_of

from euclidian.cartesian3 import Direction3, Vector3



@given(
    x=one_of(floats(min_value=0.01, max_value=10.0), floats(min_value=-10.0, max_value=-0.01)),
    y=one_of(floats(min_value=0.01, max_value=10.0), floats(min_value=-10.0, max_value=-0.01)),
    z=one_of(floats(min_value=0.01, max_value=10.0), floats(min_value=-10.0, max_value=-0.01)),
)
def test_scalar_projection_on_self_is_unity(x, y, z):
    assume(x != 0 or y != 0 or z != 0)
    u = Direction3(x, y, z)
    print(x, y, z)
    assert u.scalar_projection(u.vector()) == approx(1.0, abs=1e-5)



@given(
    x=floats(min_value=-10.0, max_value=10.0),
    y=floats(min_value=-10.0, max_value=10.0),
    z=floats(min_value=-10.0, max_value=10.0),
)
def test_scalar_projection_of_perpendicular_is_zero(x, y, z):
    assume(x != 0 or y != 0)
    print(x, y, z)
    a = Direction3(x, y, z)
    b = Vector3(-y, x, 0)
    try:
        s = a.scalar_projection(b)
    except ZeroDivisionError:
        pass
    else:
        assert s == approx(0.0)


def test_scalar_projection_positive():
    u = Direction3(1, 0, 0)
    v = Vector3(1, 1, 0)
    assert u.scalar_projection(v) == 1 / sqrt(2)


def test_scalar_projection_negative():
    u = Direction3(1, 0, 0)
    v = Vector3(-1, 1, 0)
    assert u.scalar_projection(v) == -1 / sqrt(2)


def test_scalar_projection_():
    u = Direction3(1, 1, 0)
    v = Vector3(1, 0, 0)
    assert u.scalar_projection(v) == 1 / sqrt(2)
