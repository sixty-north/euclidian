import sys
import unittest

from hypothesis import given, example
from hypothesis.strategies import floats, integers
import math

from euclidian.util import almost_equal
from euclidian.cartesian2 import Transform2, Point2


class TestTransform2(unittest.TestCase):

    @given(px=floats(-1e6, 1e6),
           py=floats(-1e6, 1e6),
           tx=floats(-1e6, 1e6),
           ty=floats(-1e6, 1e6))
    def test_translate_point(self, px, py, tx, ty):
        transform = Transform2.from_translation((tx, ty))
        before = Point2(px, py)
        after = transform(before)
        self.assertEqual(after, Point2(px+tx, py+ty))

    @given(px=floats(-1e6, 1e6),
           py=floats(-1e6, 1e6),
           scale=floats(-1e6, 1e6))
    def test_scale_point(self, px, py, scale):
        transform = Transform2.from_scale(scale)
        before = Point2(px, py)
        after = transform(before)
        self.assertTrue(almost_equal(after.x, px*scale, 1e9))
        self.assertTrue(almost_equal(after.y, py*scale, 1e9))

    @given(px=integers(),
           py=integers(),
           cx=integers(),
           cy=integers())
    def test_rotate_degrees_0(self, px, py, cx, cy):
        center = Point2(cx, cy)
        transform = Transform2.from_rotation_degrees(0, center)
        before = Point2(px, py)
        after = transform(before)

        self.assertEqual(after, before)

    @given(px=integers(),
           py=integers(),
           cx=integers(),
           cy=integers())
    def test_rotate_degrees_90(self, px, py, cx, cy):
        center = Point2(cx, cy)
        transform = Transform2.from_rotation_degrees(90, center)
        before = Point2(px, py)
        after = transform(before)

        d = before - center
        after_check = center + d.perp()
        self.assertEqual(after, after_check)

    @given(px=integers(),
           py=integers(),
           cx=integers(),
           cy=integers())
    def test_rotate_degrees_180(self, px, py, cx, cy):
        center = Point2(cx, cy)
        transform = Transform2.from_rotation_degrees(180, center)
        before = Point2(px, py)
        after = transform(before)

        d = before - center
        after_check = center - d
        self.assertEqual(after, after_check)

    @given(px=integers(),
           py=integers(),
           cx=integers(),
           cy=integers())
    def test_rotate_degrees_270(self, px, py, cx, cy):
        center = Point2(cx, cy)
        transform = Transform2.from_rotation_degrees(270, center)
        before = Point2(px, py)
        after = transform(before)

        d = before - center
        after_check = center - d.perp()
        self.assertEqual(after, after_check)

if __name__ == '__main__':
    unittest.main()
