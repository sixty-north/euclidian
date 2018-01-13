from collections import namedtuple
import operator
import sys
import math

from functools import singledispatch, total_ordering
from numbers import Real
from enum import Enum, unique
import itertools
from euclidian import graycode
from euclidian.cartesian import Cartesian, SpaceMismatchError
from euclidian.graycode import gray
from euclidian.util import sign, all_equal, is_zero


@unique
class OrientedSide(Enum):
    negative = -1
    boundary = 0
    positive = +1


@unique
class Sense(Enum):
    clockwise = -1
    none = 0
    counterclockwise = +1


def determinant_2(d00, d01, d10, d11):
    return d00 * d11 - d10 * d01


class Cartesian2(Cartesian):

    DEFAULT_AXES = ('x', 'y')

    @property
    def dimensionality(self):
        return 2


@total_ordering
class Point2(Cartesian2):
    __slots__ = ['_p']

    @classmethod
    def origin(cls, space=Cartesian2.DEFAULT_AXES):
        return Point2(0, 0, space)

    @classmethod
    def from_vector(cls, vector, space=None):
        s = space if space is not None else vector.space
        return cls(vector[0], vector[1], space=s)

    @classmethod
    def as_midpoint(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are in different spaces".format(p, q))
        return cls((p[0] + q[0]) / 2,
                   (p[1] + q[1]) / 2)

    def __init__(self, *args, **kwargs):
        try:
            space = kwargs.pop('space')
        except KeyError:
            try:
                space = args[self.dimensionality]
            except IndexError:
                space = Cartesian2.DEFAULT_AXES
        super().__init__(space)
        try:
            x = kwargs[self.space[0]] if self.space[0] in kwargs else args[0]
            y = kwargs[self.space[1]] if self.space[1] in kwargs else args[1]
        except IndexError:
            raise TypeError("Exactly {} coordinates must be provided".format(self.dimensionality))
        self._p = (x, y)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._p[i]

    def __getitem__(self, index):
        return self._p[index]

    def __len__(self):
        return len(self._p)

    def __iter__(self):
        return iter(self._p)

    def items(self):
        return zip(self.space, self._p)

    def __sub__(self, rhs):
        if not isinstance(rhs, (Point2, Vector2)):
            return NotImplemented

        if self.space != rhs.space:
            raise SpaceMismatchError("Different spaces")

        if isinstance(rhs, Vector2):
            return Point2(self._p[0] - rhs[0],
                          self._p[1] - rhs[1],
                          space=self.space)

        return Vector2(self._p[0] - rhs._p[0],
                       self._p[1] - rhs._p[1],
                       space=self.space)

    def __add__(self, rhs):
        if not isinstance(rhs, Vector2):
            return NotImplemented
        return Point2(self._p[0] + rhs._d[0],
                      self._p[1] + rhs._d[1],
                      space=self.space)

    def __abs__(self):
        x = self[0]
        y = self[1]
        return math.sqrt(x*x + y*y)

    def __eq__(self, rhs):
        if not isinstance(rhs, Point2):
            return NotImplemented
        return super().__eq__(rhs) and self._p == rhs._p

    def map(self, f, *items, space=None):
        return Point2(*list(itertools.starmap(f, zip(self, *items))),
                      space=space if space is not None else self.space)

    def __ne__(self, rhs):
        if not isinstance(rhs, Point2):
            return NotImplemented
        return not self == rhs

    def __lt__(self, rhs):
        if not isinstance(rhs, Point2):
            return NotImplemented
        if self.space != rhs.space:
            raise SpaceMismatchError("{!r} and {!r} cannot be compared".format(self, rhs))
        return self._p < rhs._p

    def __hash__(self):
        return hash((super().__hash__(), self._p))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(*item) for item in self.items()))

    def bounding_box(self):
        return Box2(self, self)

    def distance_to(self, point):
        dx = point[0] - self[0]
        dy = point[1] - self[1]
        return math.sqrt(dx*dx + dy*dy)

    def vector(self):
        """Returns the position vector."""
        return Vector2(*self._p, space=self.space)

    def intersects(self, obj):
        return _intersects_point2(obj, self)

    def intersection(self, obj):
        return _intersection_with_point2(obj, self)


class Vector2(Cartesian2):
    __slots__ = ['_d']

    @classmethod
    def from_point(cls, point):
        return cls(point[0], point[1], space=point.space)

    def __init__(self, *args, **kwargs):
        try:
            space = kwargs.pop('space')
        except KeyError:
            try:
                space = args[self.dimensionality]
            except IndexError:
                space = Cartesian2.DEFAULT_AXES
        super().__init__(space)
        try:
            x_name = self.space[0]
            y_name = self.space[1]
            dx = kwargs[x_name] if x_name in kwargs else args[0]
            dy = kwargs[y_name] if y_name in kwargs else args[1]
        except IndexError:
            raise TypeError("Exactly {} coordinates must be provided".format(self.dimensionality))
        self._d = (dx, dy)

    def map(self, f, *items, space=None):
        return Vector2(*list(itertools.starmap(f, zip(self, *items))),
                       space=space if space is not None else self.space)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._d[i]

    def __getitem__(self, index):
        return self._d[index]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def __add__(self, rhs):
        if not isinstance(rhs, Vector2):
            return NotImplemented
        return Vector2(self._d[0] + rhs._d[0],
                       self._d[1] + rhs._d[1],
                       space=self.space)

    def __sub__(self, rhs):
        if not isinstance(rhs, Vector2):
            return NotImplemented
        return Vector2(self._d[0] - rhs._d[0],
                       self._d[1] - rhs._d[1],
                       space=self.space)

    def __mul__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector2(self._d[0] * rhs,
                       self._d[1] * rhs,
                       space=self.space)

    def __rmul__(self, lhs):
        if not isinstance(lhs, Real):
            return NotImplemented
        return Vector2(lhs * self._d[0],
                       lhs * self._d[1],
                       space=self.space)

    def __truediv__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector2(self._d[0] / rhs,
                       self._d[1] / rhs,
                       space=self.space)

    def __floordiv__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector2(self._d[0] // rhs,
                       self._d[1] // rhs,
                       space=self.space)

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector2(-self._d[0], -self._d[1])

    def __abs__(self):
        return self.magnitude()

    def magnitude(self):
        return math.sqrt(self.magnitude2())

    def magnitude2(self):
        dx = self._d[0]
        dy = self._d[1]
        return dx*dx + dy*dy

    def unit(self):
        m = self.magnitude()
        if m == 0:
            raise ZeroDivisionError("Cannot produce degenerate unit vector")
        return Vector2(self._d[0] / m,
                       self._d[1] / m,
                       space=self.space)

    def direction(self):
        return Direction2(self._d[0], self._d[1], space=self.space)

    def dot(self, rhs):
        return self._d[0] * rhs._d[0] + self._d[1] * rhs._d[1]

    def perp(self):
        """Anticlockwise perpendicular vector"""
        return Vector2(-self._d[1], self._d[0], space=self.space)

    def det(self, rhs):
        """If det (the determinant) is positive the angle between A (this) and B (rhs) is positive (counter-clockwise).
        If the determinant is negative the angle goes clockwise. Finally, if the determinant is 0, the
        vectors point in the same direction. In Schneider & Eberly this operator is called Kross.
        Is is also often known as PerpDot.
        """
        return determinant_2(self._d[0], self._d[1], rhs._d[0], rhs._d[1])

    def angle(self, rhs):
        return math.atan2(abs(self.determinant(rhs)), self.dot(rhs))

    def atan2(self):
        return math.atan2(self._d[1], self._d[0])

    def components(self):
        """Decompose into two component vectors parallel to the basis vectors.

        Returns:
            A 2-tuple containing two Vector2 instances.
        """
        return (Vector2(self._d[0], 0, space=self.space),
                Vector2(0, self._d[1], space=self.space))

    def __eq__(self, rhs):
        if not isinstance(rhs, Vector2):
            return NotImplemented
        return self.space == rhs.space and self._d == rhs._d

    def __ne__(self, rhs):
        if not isinstance(rhs, Vector2):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        return hash((self.space, self._d))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(*item) for item in self.items()))


class Direction2(Cartesian2):
    __slots__ = ['_d']

    @classmethod
    def between_points(cls, p, q):
        return (q - p).direction()

    def __init__(self, *args, **kwargs):
        try:
            space = kwargs.pop('space')
        except KeyError:
            try:
                space = args[self.dimensionality]
            except IndexError:
                space = Cartesian2.DEFAULT_AXES
        super().__init__(space)
        try:
            x_name = self.space[0]
            y_name = self.space[1]
            dx = kwargs[x_name] if x_name in kwargs else args[0]
            dy = kwargs[y_name] if y_name in kwargs else args[1]
        except IndexError:
            raise TypeError("A least {} coordinates must be provided".format(self.dimensionality))
        if dx == dy == 0:
            raise ValueError("Degenerate {}".format(self.__class__.__name__))
        self._d = (dx, dy)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._d[i]

    def vector(self):
        return Vector2(self._d[0], self._d[1], space=self.space)

    def atan2(self):
        return math.atan2(self._d[1], self._d[0])

    def __eq__(self, rhs):
        if not isinstance(rhs, Direction2):
            return NotImplemented
        return (self.space == rhs.space
                and sign(self._d[0]) == sign(rhs._d[0])
                and sign(self._d[1]) == sign(rhs._d[1])
                and sign(determinant_2(self._d[0], self._d[1], rhs._d[0], rhs._d[1]) == 0))

    def __ne__(self, rhs):
        if not isinstance(rhs, Direction2):
            return NotImplemented
        return not self == rhs

    def __pos__(self):
        return self

    def __neg__(self):
        return Direction2(-self._d[0], -self._d[1], space=self.space)

    def __hash__(self):
        return hash((self.space, self._d))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(axis, coord) for axis, coord in zip(self.space, self._p)))


class Box2(Cartesian2):
    __slots__ = ['_p']

    @classmethod
    def from_points(cls, points, border=0):

        iterator = iter(points)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Iterable series 'points' must contain at least one point")

        space = first.space

        min_x = first[0]
        min_y = first[1]
        max_x = min_x
        max_y = min_y

        for index, point in enumerate(iterator, start=1):
            if point.space != space:
                raise SpaceMismatchError(
                    "Point at index {i} {!r} is not in the same space as first {!r}".format(index, point, first))
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
        return cls.from_extents(min_x - border, min_y - border, max_x + border, max_y + border)

    @classmethod
    def from_extents(cls, min_x, min_y, max_x, max_y, space=Cartesian2.DEFAULT_AXES):
        return cls(Point2(min_x, min_y, space), Point2(max_x, max_y, space))

    @classmethod
    def from_bounded(cls, bounded_objects):
        iterator = iter(bounded_objects)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Iterable series 'bounded_objects' must contain at least one bounded object")

        space = first.space

        first_bbox = first.bounding_box()
        min_x = first_bbox.min[0]
        min_y = first_bbox.min[1]
        max_x = first_bbox.max[0]
        max_y = first_bbox.max[1]

        for index, bounded in enumerate(iterator, start=1):
            if bounded.space != space:
                raise SpaceMismatchError(
                    "Bounded object at index {i} {!r} is not in the same space as first {!r}".format(index, bounded,
                                                                                                     first))

            bbox = bounded.bounding_box()
            min_x = min(min_x, bbox.min[0])
            min_y = min(min_y, bbox.min[1])
            max_x = max(max_x, bbox.max[0])
            max_y = max(max_y, bbox.max[1])
        return cls.from_extents(min_x, min_y, max_x, max_y)

    def __init__(self, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))
        super().__init__(p.space)
        self._p = tuple(sorted((p, q)))

    @property
    def min(self):
        return self._p[0]

    @property
    def max(self):
        return self._p[1]

    def bottom_left_vertex(self):
        return self._p[0]

    def bottom_right_vertex(self):
        return Point2(self.max[0],
                      self.min[1])

    def top_right_vertex(self):
        return self._p[1]

    def top_left_vertex(self):
        return Point2(self.min[0],
                      self.max[1])

    def vertex(self, i, j):
        return Point2(self._p[i][0],
                      self._p[j][1],
                      space=self.space)

    def vertices(self):
        return (self.vertex(*indices) for indices in gray(self.dimensionality))

    def bottom_edge(self):
        return Segment2(self.bottom_left_vertex(),
                        self.bottom_right_vertex())

    def right_edge(self):
        return Segment2(self.bottom_right_vertex(),
                        self.top_right_vertex())

    def top_edge(self):
        return Segment2(self.top_right_vertex(),
                        self.top_left_vertex())

    def left_edge(self):
        return Segment2(self.top_left_vertex(),
                        self.bottom_left_vertex())

    def edges(self):
        yield self.bottom_edge()
        yield self.right_edge()
        yield self.top_edge()
        yield self.left_edge()

    def intersects(self, obj):
        return _intersects_box2(obj, self)

    def vector(self):
        return self._p[1] - self._p[0]

    def __getitem__(self, index):
        return self._p[index]

    def __len__(self):
        return len(self._p)

    def __eq__(self, rhs):
        if not isinstance(rhs, Box2):
            return NotImplemented
        return self.space == rhs.space and self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Box2):
            return NotImplemented
        return not self == rhs

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._p[0], self._p[1])

    def distance_to(self, point):
        if self.intersects(point):
            return 0
        return min(edge.distance_to(point) for edge in self.edges())


class Line2(Cartesian2):
    __slots__ = ['_c']

    @classmethod
    def through_points(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))

        if p == q:
            raise ValueError("Attempt to create a degenerate line")

        if p[1] == q[1]:  # Horizontal line
            a = 0
            if q[0] > p[0]:
                b = 1
                c = -p[1]
            elif q[0] == p[0]:
                b = 0
                c = 0
            else:
                b = -1
                c = p[1]
        elif q[0] == p[0]:  # Vertical line
            b = 0
            if q[1] > p[1]:
                a = -1
                c = p[0]
            elif q[1] == p[1]:
                a = 0
                c = 0
            else:
                a = 1
                c = -p[0]
        else:  # General line
            a = p[1] - q[1]
            b = q[0] - p[0]
            c = -p[0] * a - p[1] * b
        return cls(a, b, c, space=p.space)

    @classmethod
    def perpendicular_to_line_through_point(cls, line, point):
        if line.space != point.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(line, point))
        a = -line.b
        b = line.a
        c = line.b * point[0] - line.a * point[1]
        return Line2(a, b, c, space=line.space)

    @classmethod
    def bisecting_points(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))
        a = 2 * (p[0] - q[0])
        b = 2 * (p[1] - q[1])
        c = q[0] * q[0] + q[1] * q[1] - p[0] * p[0] - p[1] * p[1]
        return cls(a, b, c, space=p.space)

    @classmethod
    def bisecting_lines(cls, m, n):
        if m.space != m.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(m, n))
        n1 = math.sqrt(m.a * m.a + m.b * m.b)
        n2 = math.sqrt(n.a * n.a + n.b * n.b)
        a = n2 * m.a + n1 * n.a
        b = n2 * m.b + n1 * n.b
        c = n2 * m.c + n1 * n.c

        if a == 0 and b == 0:
            a = n2 * m.a - n1 * n.a
            b = n2 * m.b - n1 * n.b
            c = n2 * m.c - n1 * n.c
        return Line2(a, b, c, space=m.space)

    @classmethod
    def through_point_with_direction(cls, p, d):
        if p.space != d.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, d))
        a = -d[1]
        b = d[0]
        c = p[0] * d[1] - p[1] * d[0]
        return cls(a, b, c, space=p.space)

    @classmethod
    def supporting_segment(cls, segment):
        return segment.supporting_line()

    @classmethod
    def supporting_ray(cls, ray):
        raise ray.supporting_line()

    def __init__(self, a, b, c, space=Cartesian2.DEFAULT_AXES):
        super().__init__(space)
        self._c = (a, b, c)

    @property
    def a(self):
        return self._c[0]

    @property
    def b(self):
        return self._c[1]

    @property
    def c(self):
        return self._c[2]

    def is_horizontal(self):
        return self.a == 0.0 and self.b != 0.0

    def is_vertical(self):
        return self.b == 0.0 and self.a != 0.0

    def is_degenerate(self):
        raise self.a == 0 and self.b == 0.0

    def opposite(self):
        return Line2(-self.a, -self.b, -self.c)

    def normal(self):
        return Direction2(self.a, self.b)

    def direction(self):
        raise NotImplemented

    def perpendicular(self):
        return Line2(self.b, -self.a, self.c)

    def point(self, i=0):
        """Generate a point on the line.

        Args:
            i: By providing different values of i (which defaults to 0) distinct
               points on the line can be produced.

        Returns:
            A Point2 on the line.
        """
        if self.b == 0:
            return Point2((-self.b - self.c) / self.a + i * self.b,
                          1 - i * self.a,
                          space=self.space)
        return Point2(1 + i * self.b,
                      -(self.a + self.c) / self.b - i * self.a,
                      space=self.space)

    def _solve_for_0(self, y):
        # TODO: Consider accepting named arguments corresponding to the space axis names
        if self.a == 0:
            raise ZeroDivisionError("Cannot solve horizontal line for _solve_for_0")
        return -self.b * y / self.a - self.c / self.a

    def _solve_for_1(self, x):
        # TODO: Consider accepting named arguments corresponding to the space axis names
        if self.b == 0:
            raise ZeroDivisionError("Cannot solve for vertical line for _solve_for_1")
        return -self.a * x / self.b - self.c / self.b

    def solve(self, axis, value):
        try:
            index = self.space.index(axis)
        except ValueError as e:
            raise ValueError("Unrecognised axis '{}'".format(axis)) from e
        if index == 0:
            return self._solve_for_0(value)
        elif index == 1:
            return self._solve_for_1(value)
        assert False, "We never reach here."

    def __getattr__(self, name):
        if name.startswith('solve_for_'):
            axis = name[10:]
            try:
                index = self.space.index(axis)
            except ValueError:
                pass
            else:
                if index == 0:
                    return self._solve_for_0
                elif index == 1:
                    return self._solve_for_1
                assert False, "We never reach here."
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))

    def distance_to(self, point):
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        a, b, c = self._c
        return (point[0] * a
              + point[1] * b
              + c) / math.sqrt(a*a + b*b)

    def projected_from(self, point):
        """The point on this line closest to the given point.

        Args:
            point: The point for which to find the closest point on this line.

        Returns:
            The point on this line which is closest to 'point'.
        """
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        a, b, c = self._c
        bp0 = b * point[0]
        ap1 = a * point[1]
        a2b2 = a * a + b * b
        return Point2((b * ( bp0 - ap1) - a * self.c) / a2b2,
                      (a * (-bp0 + ap1) - b * self.c) / a2b2,
                      space=self.space)

    def is_parallel_to(self, line):
        if line.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(line, self))
        return sign(determinant_2(self.a, self.b, line.a, line.b)) == 0

    def side(self, point):
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(self, point))
        solution = self.a * point[0] + self.b * point[1] + self.c
        if solution > 0:
            return OrientedSide.positive
        elif solution < 0:
            return OrientedSide.negative
        return OrientedSide.Boundary

    def has_on(self, point):
        return self.side(point) == OrientedSide.boundary

    def has_on_positive_side(self, point):
        return self.side(point) == OrientedSide.positive

    def has_on_negative_side(self, point):
        return self.side(point) == OrientedSide.Negative

    def intersection(self, obj):
        return _intersection_with_line2(obj, self)

    def intersects(self, obj):
        return _intersects_line2(obj, self)

    def __eq__(self, rhs):
        if self is rhs:
            return True

        if not isinstance(rhs, Line2):
            return NotImplemented

        if self.space != rhs.space:
            raise False

        if sign(determinant_2(self.a, self.b, rhs.a, rhs.b)) != 0:
            return False

        s1a = sign(self.a)
        if s1a != 0:
            return s1a == sign(rhs.a) and (sign(determinant_2(self.a, self.c, rhs.a, rhs.c)) == 0)

        return sign(self.b) == sign(rhs.b) and (sign(determinant_2(self.b, self.c, rhs.b, rhs.c)) == 0)

    def __ne__(self, rhs):
        return not self == rhs

    def __hash__(self):
        return hash((self.space, self._c))

    def __repr__(self):
        return '{}({}, {}, {}, space={})'.format(self.__class__.__name__, self.a, self.b, self.c, self.space)


class Ray2(Cartesian2):
    __slots__ = ['_p']

    @classmethod
    def from_source_and_direction(cls, source, direction):
        if source.space != direction.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(source, direction))
        return cls(source, source + direction.vector())

    @classmethod
    def from_source_and_vector(cls, source, vector):
        if source.space != vector.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(source, vector))
        return cls(source, source + vector)

    def __init__(self, source, point):
        if source.space != point.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(source, point))
        super().__init__(source.space)
        self._p = (source, point)

    @property
    def source(self):
        return self._p[0]

    @property
    def point(self):
        return self._p[1]

    def vector(self):
        return self.point - self.source

    def direction(self):
        return self.vector().direction()

    def supporting_line(self):
        return Line2.through_points(*self._p)

    def opposite(self):
        return Ray2.from_source_and_vector(self.source, self.source - self.point)

    def lerp(self, t):
        # TODO: Consider using this version which is more numerically stable lerp(p0, p1, t) = p0 * (1 - t) + p1 * t
        u = (self.point - self.source).unit()
        return self.source + u * t

    def distance_to(self, point):
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        v = self.vector()
        l2 = v.magnitude2()
        if l2 == 0:
            return self.source.distance_to(point)
        t = (point - self.source).dot(v) / l2
        if t < 0:
            return self.source.distance_to(point)
        projection = self.source + t * v
        return projection.distance_to(point)

    def projected_from(self, point):
        """Project a point onto this segment.

        Args:
            point: The point to be projected.

        Return:
            The perpendicular projection of point onto this segment, or None.
        """
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        v = self.vector()
        l2 = v.magnitude2()
        t = (point - self.source).dot(v) / l2
        if t < 0:
            return None
        return self.source + t * v

    def __eq__(self, rhs):
        if not isinstance(rhs, Ray2):
            return NotImplemented
        return self.source == rhs.source and self.direction() == rhs.direction()

    def __ne__(self, rhs):
        return not self == rhs

    def __hash__(self):
        return hash((self.source, self.direction()))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._p[0], self._p[1])


class Segment2(Cartesian2):
    __slots__ = ['_p']

    @classmethod
    def from_point_and_vector(cls, source, vector):
        return cls(source, source + vector)

    def __init__(self, source, target):
        if source.space != target.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(source, target))
        super().__init__(source.space)
        self._p = (source, target)

    @property
    def source(self):
        return self._p[0]

    @property
    def target(self):
        return self._p[1]

    def __getitem__(self, index):
        return self._p[index]

    def __len__(self):
        return len(self._p)

    def vector(self):
        return self.target - self.source

    def direction(self):
        return self.vector().direction()

    def supporting_line(self):
        return Line2.through_points(*self._p)

    def midpoint(self):
        return Point2.as_midpoint(self.source, self.target)

    def length(self):
        dx = self.target[0] - self.source[0]
        dy = self.target[1] - self.source[1]
        return math.sqrt(dx*dx + dy*dy)

    def reversed(self):
        return Segment2(self.target, self.source)

    def lerp(self, t):
        # TODO: Consider using this version which is more numerically stable lerp(p0, p1, t) = p0 * (1 - t) + p1 * t
        return Point2(self.source[0] + t * (self.target[0] - self.source[0]),
                      self.source[1] + t * (self.target[1] - self.source[1]),
                      space=self.space)

    def bounding_box(self):
        return Box2(*self._p)

    def distance_to(self, point):
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        v = self.vector()
        l2 = v.magnitude2()
        if l2 == 0:
            return self.source.distance_to(point)
        t = (point - self.source).dot(v) / l2
        if t < 0:
            return self.source.distance_to(point)
        elif t > 1:
            return self.target.distance_to(point)
        projection = self.source + t * v
        return projection.distance_to(point)

    def projected_from(self, point):
        """Project a point onto this segment.

        Args:
            point: The point to be projected.

        Return:
            The perpendicular projection of point onto this segment, or None.
        """
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        v = self.vector()
        l2 = v.magnitude2()
        t = (point - self.source).dot(v) / l2
        if t < 0:
            return None
        elif t > 1:
            return None
        return self.source + t * v

    def intersects(self, obj):
        return _intersects_segment2(obj, self)

    def __eq__(self, rhs):
        if not isinstance(rhs, Segment2):
            return NotImplemented
        return self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Segment2):
            return NotImplemented
        return self._p != rhs._p

    def __hash__(self):
        return hash(self._p)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._p[0], self._p[1])


class Triangle2(Cartesian2):
    @classmethod
    def from_iterable(cls, i):
        a = next(i)
        b = next(i)
        c = next(i)
        return cls(a, b, c)

    def __init__(self, a, b, c):
        if not (a.space == b.space == c.space):
            raise SpaceMismatchError("{!r}, {!r} and {!r} are not in the same space".format(a, b, c))
        super().__init__(a.space)
        self._p = (a, b, c)

    @property
    def a(self):
        return self._p[0]

    @property
    def b(self):
        return self._p[1]

    @property
    def c(self):
        return self._p[2]

    def vertices(self):
        return iter(self._p)

    def __iter__(self):
        return iter(self._p)

    def __getitem__(self, index):
        return self._p[index]

    def __len__(self):
        return len(self._p)

    def incenter(self):
        if hasattr(self, '_incenter'):
            return self._incenter

        ax1 = self.length_a() * self.a[0]
        bx2 = self.length_b() * self.b[0]
        cx3 = self.length_c() * self.c[0]
        x = (ax1 + bx2 + cx3) / self.perimeter()

        ay1 = self.length_a() * self.a[1]
        by2 = self.length_b() * self.b[1]
        cy3 = self.length_c() * self.c[1]
        y = (ay1 + by2 + cy3) / self.perimeter()
        return Point2(x, y, space=self.space)

    def circumcenter(self):
        if hasattr(self, '_circumcenter'):
            return self._circumcenter

        if abs(self.a[1] - self.b[1]) < sys.float_info.epsilon and abs(self.b[1] - self.c[1]) < sys.float_info.epsilon:
            raise ArithmeticError("Cannot compute circumcenter for degenerate triangle")

        if abs(self.b[1] - self.a[1]) < sys.float_info.epsilon:
            self._circumcenter = self._circumcenter_1()
        elif abs(self.c[1] - self.b[1]) < sys.float_info.epsilon:
            self._circumcenter = self._circumcenter_2()
        else:
            self._circumcenter = self._circumcenter_3()

        return self._circumcenter

    def _circumcenter_3(self):
        a = self.a
        b = self.b
        c = self.c
        m1 = -(b[0] - a[0]) / (b[1] - a[1])
        m2 = -(c[0] - b[0]) / (c[1] - b[1])
        mx1 = (a[0] + b[0]) * 0.5
        mx2 = (b[0] + c[0]) * 0.5
        my1 = (a[1] + b[1]) * 0.5
        my2 = (b[1] + c[1]) * 0.5
        xc = (m1 * mx1 - m2 * mx2 + my2 - my1) / (m1 - m2)
        return Point2(xc, m1 * (xc - mx1) + my1, space=self.space)

    def _circumcenter_2(self):
        a = self.a
        b = self.b
        c = self.c
        m1 = -(b[0] - a[0]) / (b[1] - a[1])
        mx1 = (a[0] + b[0]) * 0.5
        my1 = (a[1] + b[1]) * 0.5
        xc = (c[0] + b[0]) * 0.5
        return Point2(xc, m1 * (xc - mx1) + my1, space=self.space)

    def _circumcenter_1(self):
        a = self.a
        b = self.b
        c = self.c
        m2 = -(c[0] - b[0]) / (c[1] - b[1])
        mx2 = (b[0] + c[0]) * 0.5
        my2 = (b[1] + c[1]) * 0.5
        xc = (b[0] + a[0]) * 0.5
        return Point2(xc, m2 * (xc - mx2) + my2, space=self.space)

    def determinant(self):
        a = self.a
        b = self.b
        c = self.c
        return (a[0] * b[1] - a[0] * c[1] - b[0] * a[1] + b[0]
                * c[1] - c[0] * b[1] + c[0] * a[1])

    def signed_area(self):
        return self.determinant / 2.0

    def area(self):
        return abs(self.signed_area())

    def is_degenerate(self):
        return self.determinant() == 0.0

    def handedness(self):
        det = self.determinant()
        if det < 0.0:
            return Sense.clockwise
        if det > 0.0:
            return Sense.counterclockwise
        return Sense.none

    def length_a(self):
        return self.edge_a().length()

    def length_b(self):
        return self.edge_b().length()

    def length_c(self):
        return self.edge_c().length()

    def perimeter(self):
        return sum(self.edges())

    def angle_a(self):
        return self._angle_from_side_vectors(self.b - self.a, self.c - self.a)

    def angle_b(self):
        return self._angle_from_side_vectors(self.c - self.b, self.a - self.b)

    def angle_c(self):
        return self._angle_from_side_vectors(self.a - self.c, self.b - self.c)

    def edge_a(self):
        return Segment2(self.b, self.c)

    def edge_b(self):
        return Segment2(self.c, self.a)

    def edge_c(self):
        return Segment2(self.a, self.b)

    def edges(self):
        yield self.edge_a()
        yield self.edge_b()
        yield self.edge_c()

    @staticmethod
    def _angle_from_side_vectors(p, q):
        return p.angle(q)

    def is_in_circumcircle(self, p):
        if self.space != p.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(self, p))
            # Return TRUE if the point (xp,yp) lies inside the circumcircle
        # made up by points (x1,y1) (x2,y2) (x3,y3)
        # NOTE: A point on the edge is inside the circumcircle

        center = self.circumcenter()
        bsqr = (self.b - center).Magnitude2
        psqr = (p - center).Magnitude2

        return psqr <= bsqr

    def bounding_box(self):
        return Box2.from_points(self._p)

    def trilinear_to_barycentric(self, r, s, t):
        return self.length_a() * r, self.length_b() * s, self.length_c() * t

    def barycentric_to_cartesian(self, u, v, w):
        a = self.a
        b = self.b
        c = self.c
        return Point2(u * a[0] + v * b[0] + w * c[0],
                      u * a[1] + v * b[1] + w * c[1],
                      space=self.space)

    def trilinear_to_cartesian(self, alpha, beta, gamma):
        u, v, w = self.trilinear_to_barycentric(alpha, beta, gamma)
        return self.barycentric_to_cartesian(u, v, w)

    def cartesian_to_barycentric(self, p):
        if self.space != p.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(self, p))
        a = self.a
        b = self.b
        c = self.c
        dx = p[0] - c[0]
        dy = p[1] - c[1]
        u = (b[1] - c[1]) * dx + (c[0] - b[0]) * dy / self.determinant()
        v = (c[1] - a[1]) * dx + (a[0] - c[0]) * dy / self.determinant()
        w = 1 - u - v
        return u, v, w

    def cartesian_to_trilinear(self, p):
        # p = alpha * a + beta * b
        # Use Cramer's rule here
        # px = alpha*ax + beta*bx
        # py = alpha*ay + beta*by
        a = self.a
        b = self.b
        d = determinant_2(a[0], b[0], a[1], b[1])
        dx = determinant_2(p[0], b[0], p[1], b[1])
        dy = determinant_2(a[0], p[0], a[1], p[1])
        alpha = dx / d
        beta = dy / d
        r = beta / self.length_a()
        s = alpha / self.length_b()
        t = (1 - alpha - beta) / self.length_c()
        return r, s, t

    def intersects(self, obj):
        return _intersects_triangle2(obj, self)

    def __eq__(self, rhs):
        if not isinstance(rhs, Triangle2):
            return NotImplemented
        return self.space == rhs.space and self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Triangle2):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        return hash((self.space, self._p))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__, self.a, self.b, self.c)


class Circle2(Cartesian2):

    def __init__(self, center, radius):
        super().__init__(center.space)
        self._center = center
        self._radius = radius

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def area(self):
        radius = self._radius
        return math.pi * radius * radius

    def bounding_box(self):
        return Box2.from_extents(self._center[0] - self.radius,
                                 self._center[1] - self.radius,
                                 self._center[0] + self.radius,
                                 self._center[1] + self.radius)

    def intersects(self, obj):
        if self.space != obj.space:
            raise SpaceMismatchError("{!r} is not in the same space as {!r}".format(obj, self))
        return _intersects_circle2(obj, self)

    def __repr__(self):
        return '{}({!r}, {})'.format(self.__class__.__name__, self._center, self._radius)

    def __eq__(self, rhs):
        if not isinstance(rhs, Circle2):
            return NotImplemented
        return self.space == rhs.space and self._center == rhs._center and self._radius == rhs._radius

    def __ne__(self, rhs):
        if not isinstance(rhs, Circle2):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        return hash((self.space, self._center, self._radius))


class Quadrilateral2(Cartesian2):

    def __init__(self, a, b, c, d):
        if not (a.space == b.space == c.space == d.space):
            raise SpaceMismatchError("{!r}, {!r}, {!r} and {!r} are not in the same space".format(a, b, c, d))
        super().__init__(a.space)
        self._p = (a, b, c, d)

    @property
    def a(self):
        return self._p[0]

    @property
    def b(self):
        return self._p[1]

    @property
    def c(self):
        return self._p[2]

    @property
    def d(self):
        return self._p[2]

    def vertices(self):
        return iter(self._p)

    def __iter__(self):
        return iter(self._p)

    def __getitem__(self, index):
        return self._p[index]

    def __len__(self):
        return len(self._p)

    def edge_ab(self):
        return Segment2(self.a, self.b)

    def edge_bc(self):
        return Segment2(self.b, self.c)

    def edge_cd(self):
        return Segment2(self.c, self.d)

    def edge_da(self):
        return Segment2(self.d, self.a)

    def length_ab(self):
        return self.edge_ab().length()

    def length_bc(self):
        return self.edge_bc().length()

    def length_cd(self):
        return self.edge_cd().length()

    def length_da(self):
        return self.edge_da().length()

    def perimeter(self):
        return sum(self.edges())

    def edges(self):
        yield self.edge_ab()
        yield self.edge_bc()
        yield self.edge_cd()
        yield self.edge_da()

    def angle_a(self):
        return self._angle_from_side_vectors(self.b - self.a, self.d - self.a)

    def angle_b(self):
        return self._angle_from_side_vectors(self.c - self.b, self.a - self.b)

    def angle_c(self):
        return self._angle_from_side_vectors(self.d - self.c, self.b - self.c)

    def angle_d(self):
        return self._angle_from_side_vectors(self.a - self.d, self.c - self.d)

    def angles(self):
        yield self.angle_a()
        yield self.angle_b()
        yield self.angle_c()
        yield self.angle_d()

    @staticmethod
    def _angle_from_side_vectors(p, q):
        return p.angle(q)

    def diagonal_ac(self):
        return Segment2(self.a, self.c)

    def diagonal_bd(self):
        return Segment2(self.b, self.d)

    def is_convex(self):
        return self.diagonal_ac.intersects(self.diagonal_bd)

    def rectangularity(self):
        return max(self.angles()) - min(self.angles())

    def is_rectangle(self):
        return self.rectangularity() == 0

    def __eq__(self, rhs):
        if not isinstance(rhs, Quadrilateral2):
            return NotImplemented
        return self.space == rhs.space and self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Quadrilateral2):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        return hash((self.space, self._p))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__, self.a, self.b, self.c, self.d)


class TransformDecompositionError(Exception):
    pass


class Transform2:

    @staticmethod
    def identity():
        return IDENTITY_TRANSFORM2

    @classmethod
    def reflection_about(cls, line):
        a = line.a
        b = line.b
        c = line.c
        a2 = a * a
        b2 = b * b
        a2pb2 = a2 + b2
        a2mb2 = a2 - b2
        b2ma2 = b2 - a2
        return cls( b2ma2 / a2pb2,
                   -2*a*b / a2pb2,
                   -2*a*b / a2pb2,
                    a2mb2 / a2pb2,
                   -2*a*c / a2pb2,
                   -2*b*c / a2pb2)

    @classmethod
    def windowing(cls, from_box, to_box):
        """A windowing transformation which transforms from one box to another.

        Args:
            from_box (Box2): The box to transform from.
            to_box (Box2): The box to transform to.

        Returns: A transformation.
        """
        # TODO: Check!
        fv = from_box.vector()
        tv = to_box.vector()
        sv = tv.map(operator.truediv, fv)
        tv = to_box.min - from_box.min
        return cls.identity().scale(sv, from_box.min).translate(tv)

    @classmethod
    def from_rotation(cls, angle_radians, center=None):

        return cls.identity().rotate(angle_radians, center)

    @classmethod
    def from_rotation_degrees(cls, angle_degrees, center=None):
        return cls.identity().rotate_degrees(angle_degrees, center)

    @classmethod
    def from_scale(cls, scale_factor, center=None):
        # TODO: Optimise
        return cls.identity().scale(scale_factor, center)

    @classmethod
    def from_translation(cls, vector):
        return cls(1, 0, 0, 1, vector[0], vector[1])

    def __init__(self, a, c, b, d, tx, ty):
        self._m = ((a, c, tx),
                   (b, d, ty))

    @property
    def a(self):
        return self._m[0][0]

    @property
    def b(self):
        return self._m[1][0]

    @property
    def c(self):
        return self._m[0][1]

    @property
    def d(self):
        return self._m[1][1]

    @property
    def tx(self):
        # TODO: The name should reflect the space
        return self._m[0][2]

    @property
    def ty(self):
        # TODO: The name should reflect the space
        return self._m[1][2]

    def __eq__(self, rhs):
        if not isinstance(rhs, Transform2):
            return NotImplemented
        return self._m == rhs._m

    def __ne__(self, rhs):
        return not self == rhs

    def __repr__(self):
        return '{}(a={}, c={}, b={}, d={}, tx={}, ty={})'.format(self.a, self.c, self.b, self.d, self.tx, self.ty)

    def __mul__(self, rhs):
        if not isinstance(rhs, Transform2):
            return NotImplemented
        a1 = self.a
        b1 = self.b
        c1 = self.c
        d1 = self.d
        tx1 = self.tx
        ty1 = self.ty

        a2 = rhs.a
        b2 = rhs.b
        c2 = rhs.c
        d2 = rhs.d
        tx2 = rhs.tx
        ty2 = rhs.ty

        return Transform2(a=a2 * a1 + c2 * b1,
                          b=b2 * a1 + d2 * b1,
                          c=a2 * c1 + c2 * d1,
                          d=b2 * c1 + d2 * d1,
                          tx=tx1 + (tx2 * a1 + ty2 * b1),
                          ty=ty1 + (tx2 * c1 + ty2 * d1))

    def __rmul__(self, lhs):
        if not isinstance(lhs, Transform2):
            return NotImplemented
        a1 = self.a
        b1 = self.b
        c1 = self.c
        d1 = self.d
        tx1 = self.tx
        ty1 = self.ty

        a2 = lhs.a
        b2 = lhs.b
        c2 = lhs.c
        d2 = lhs.d
        tx2 = lhs.tx
        ty2 = lhs.ty

        return Transform2(a=a2 * a1 + b2 * c1,
                          b=a2 * b1 + b2 * d1,
                          c=c2 * a1 + d2 * c1,
                          d=c2 * b1 + d2 * d1,
                          tx=a2 * tx1 + b2 * ty1 + tx2,
                          ty=c2 * tx1 + d2 * ty1 + ty2)

    def translate(self, vector):
        tx = vector.x * self.a + vector.y * self.b
        ty = vector.x * self.c + vector.y * self.d
        return Transform2(self.a, self.c, self.b, self.d, tx, ty)

    def scale(self, scale_factor, center=None):
        s = Vector2(scale_factor, scale_factor) if isinstance(scale_factor, Real) else scale_factor

        if center is None:
            return self.replace(a=self.a * s.x,
                                c=self.c * s.x,
                                b=self.b * s.y,
                                d=self.d * s.y)

        v = center.vector()
        return self.translate(v).scale(scale_factor).translate(-v)

    def rotate(self, angle_radians, center=None):
        center = Point2(0, 0) if center is None else center
        x = center[0]
        y = center[1]
        cos = math.cos(angle_radians)
        sin = math.sin(angle_radians)
        tx = x - x * cos + y * sin
        ty = y - x * sin - y * cos
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return Transform2(a=cos * a + sin * b,
                          b=-sin * a + cos * b,
                          c=cos * c + sin * d,
                          d=-sin * c + cos * d,
                          tx=self.tx + (tx * a + ty * b),
                          ty=self.ty + (tx * c + ty * d))

    def rotate_degrees(self, angle_degrees, center=None):
        if angle_degrees % 90 == 0:
            quadrant = int(angle_degrees) % 360
            method = getattr(self, 'rotate_{}_degrees'.format(quadrant))
            return method(center)
        return self.rotate(math.radians(angle_degrees), center)

    def rotate_0_degrees(self, center=None):
        return self

    def rotate_90_degrees(self, center=None):
        center = Point2(0, 0) if center is None else center
        x = center[0]
        y = center[1]
        tx = x + y
        ty = y - x
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return Transform2(a=b,
                          b=-a,
                          c=d,
                          d=-c,
                          tx=self.tx + (tx * a + ty * b),
                          ty=self.ty + (tx * c + ty * d))

    def rotate_180_degrees(self, center=None):
        center = Point2(0, 0) if center is None else center
        x = center[0]
        y = center[1]
        tx = x + x
        ty = y + y
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return Transform2(a=-a,
                          b=-b,
                          c=-c,
                          d=-d,
                          tx=self.tx + (tx * a + ty * b),
                          ty=self.ty + (tx * c + ty * d))

    def rotate_270_degrees(self, center=None):
        center = Point2(0, 0) if center is None else center
        x = center[0]
        y = center[1]
        tx = x - y
        ty = y + x
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return Transform2(a=-b,
                          b=a,
                          c=-d,
                          d=c,
                          tx=self.tx + (tx * a + ty * b),
                          ty=self.ty + (tx * c + ty * d))

    def shear(self, shear_factor, center=None):
        if center is None:
            return self.replace(a=self.a + shear_factor.y * self.b,
                                c=self.c + shear_factor.y * self.d,
                                b=self.b + shear_factor.x * self.a,
                                d=self.d + shear_factor.x * self.c)

        v = center.vector()
        return self.translate(v).shear(shear_factor).translate(-v)

    def skew(self, angle0, angle1, center=None):
        shear_factor = Vector2(math.tan(angle0),
                               math.tan(angle1))
        return self.shear(shear_factor, center)

    def replace(self, **kwargs):
        return Transform2(a=kwargs.get('a', self.a),
                          c=kwargs.get('c', self.c),
                          b=kwargs.get('b', self.b),
                          d=kwargs.get('d', self.d),
                          tx=kwargs.get('tx', self.tx),
                          ty=kwargs.get('ty', self.ty))

    def __call__(self, obj):
        """Transform obj"""
        return transform2(obj, self)

    def translation(self):
        return self._decomposition().translation

    def scaling(self):
        return self._decomposition().scaling

    def rotation(self):
        return self._decomposition().rotation

    def shearing(self):
        return self._decomposition().shearing

    def _decomposition(self):
        if not hasattr(self, "_decomp"):
            # http://dev.w3.org/csswg/css3-2d-transforms/#matrix-decomposition
            # http://stackoverflow.com/questions/4361242/
            # https://github.com/wisec/DOMinator/blob/master/layout/style/nsStyleAnimation.cpp#L946
            a = self.a
            b = self.b
            c = self.c
            d = self.d
            if is_zero(a * d - b * c):
                raise TransformDecompositionError("Transformation matrix could not be decomposed.")

            scale_x = math.sqrt(a * a + b * b)
            a /= scale_x
            b /= scale_x

            shear = a * c + b * d
            c -= a * shear
            d -= b * shear

            scale_y = math.sqrt(c * c + d * d)
            c /= scale_y
            d /= scale_y
            shear /= scale_y

            # a * d - b * c should now be 1 or -1
            # TODO: Assert (with tolerance)
            if a * d < b * c:
                a = -a
                b = -b
                # We don't need c & d anymore, but if we did, we'd have to do this
                # c = -c
                # d = -d
                shear = -shear
                scale_x = -scale_x


            self._decomp = Decomposition(translation=Vector2(self.tx, self.ty),
                                         scaling=Vector2(scale_x, scale_y),
                                         rotation=-math.atan2(b, a),
                                         shearing=shear)
        return self._decomp

    def inverse(self):
        decomp = self._decomposition()
        inv_translation = -decomp.translation
        inv_scaling = decomp.scaling.map(lambda c: 1 / c)
        inv_rotation = -decomp.rotation
        inv_shearing = -decomp.shearing
        return Transform2.identity()                 \
                         .shear(inv_shearing)        \
                         .rotate(inv_rotation)       \
                         .scale(inv_scaling)         \
                         .translate(inv_translation)

    def is_identity(self):
        return self.a == 1 and self.c == 0 and self.b == 0 and self.d == 1 and self.tx == 0 and self.tx == 0

IDENTITY_TRANSFORM2 = Transform2(1, 0, 0, 1, 0, 0)

# TODO: Ensure these are in the right order, so we can reduce the transformations.
Decomposition = namedtuple('Decomposition', ['translation', 'scaling', 'rotation', 'shearing'])

@singledispatch
def transform2(obj, transform):
    raise NotImplemented("Transformation of {!r} not implemented".format(obj))


@transform2.register(Point2)
def _(point, transform):
    return Point2(point.x * transform.a + point.y * transform.b + transform.tx,
                  point.x * transform.c + point.y * transform.d + transform.ty)


@transform2.register(Vector2)
def _(vector, transform):
    return Vector2(vector.x * transform.a + vector.y * transform.b,
                   vector.x * transform.c + vector.y * transform.d)


@transform2.register(Direction2)
def _(direction, transform):
    return Vector2(direction.x * transform.a + direction.y * transform.b,
                   direction.x * transform.c + direction.y * transform.d)

@transform2.register(Segment2)
def _(segment, transform):
    return Segment2(transform2(segment.source, transform),
                    transform2(segment.target, transform))


@transform2.register(Line2)
def _(line, transform):
    # Surely there is a more elegant way that this...
    p0 = line.point(0)
    p1 = line.point(1)
    s = transform2(Segment2(p0, p1), transform)
    return s.supporting_line()


@transform2.register(Triangle2)
def _(triangle, transform):
    return Triangle2(transform2(triangle.a, transform),
                     transform2(triangle.b, transform),
                     transform2(triangle.c, transform))


@transform2.register(Ray2)
def _(ray, transform):
    return Ray2(transform2(ray.source, transform),
                transform2(ray.point, transform))


@transform2.register(Box2)
def _(box, transform):
    # Not strictly a transformation of the box as such. Returns
    # A new axis-aligned Box2 which bounds the transformed box
    return Box2.from_points(transform2(p) for p in box.vertices())


def intersection2(a, b):
    return a.intersection(b)


@singledispatch
def _intersection_with_point2(obj, point):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(obj, point))


@_intersection_with_point2.register(Point2)
def _(other_point, point):
    if point == other_point:
        return point
    return None


@_intersection_with_point2.register(Line2)
def _(line, point):
    return _intersection_with_line2(point, line)


@singledispatch
def _intersection_with_line2(obj, line):
    raise NotImplementedError("Intersection between {!r} and {!r} not implemented.".format(obj, line))


@_intersection_with_line2.register(Point2)
def _(point, line):
    if line.has_on(point):
        return point
    return None


@_intersection_with_line2.register(Line2)
def _(other_line, line):
    if line == other_line:
        return line
    if line.is_parallel_to(other_line):
        return None
    a = line.a
    b = line.b
    c = line.c
    d = other_line.a
    e = other_line.b
    f = other_line.c
    x = (c*e - b*f) / (b*d - a*e)
    y = (a*f - c*d) / (b*d - a*e)
    return Point2(x, y)


def intersects2(a, b):
    return a.intersects(b)


@singledispatch
def _intersects_point2(obj, point):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(obj, point))


@_intersects_point2.register(Point2)
def _(other_point, point):
    return point == other_point


@_intersects_point2.register(Line2)
def _(line, point):
    return _intersects_line2(point, line)


@singledispatch
def _intersects_box2(obj, box):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(box, obj))


@_intersects_box2.register(Point2)
def _(point, box):
    if point.space != box.space:
        raise SpaceMismatchError("{!r} is not in the same space as {!r}".format(point, box))
    return all(box.min[c] <= point[c] <= box.max[c] for c in box.dimensionality)


@singledispatch
def _intersects_line2(obj, line):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(line, obj))


@_intersects_line2.register(Line2)
def _(other_line, line):
    return not line.is_parallel_to(other_line)


@_intersects_line2.register(Point2)
def _(point, line):
    return line.has_on(point)


@singledispatch
def _intersects_segment2(obj, segment):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(segment, obj))


@_intersects_segment2.register(Segment2)
def _(segment_p, segment_q):
    if segment_p == segment_q:
        return True
    if segment_p == segment_q.reversed():
        return True
    # TODO Collinear overlapping case

    line_p = segment_p.supporting_line()
    line_q = segment_q.supporting_line()
    return  line_p.side(segment_q.source) != line_p.side(segment_q.target) \
        and line_q.side(segment_p.source) != line_q.side(segment_p.source)


@singledispatch
def _intersects_triangle2(obj, triangle):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(triangle, obj))


@_intersects_triangle2.register(Point2)
def _(point, triangle):
    r, s, t = triangle.cartesian_to_barycentric(point)
    return sign(r) == sign(s) == sign(t) == 1


@_intersects_triangle2.register(Line2)
def _(line, triangle):
    return all_equal(sign(line.a * v[0] + line.b * v[1] + line.c) for v in triangle.vertices())


@_intersects_triangle2.register(Segment2)
def _(segment, triangle):
    return any(segment.intersects(edge) for edge in triangle.edges())


@singledispatch
def _intersects_circle2(obj, circle):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(circle, obj))


@_intersects_circle2.register(Circle2)
def _(circle1, circle2):
    return circle1.radius + circle2.radius >= (circle1.center - circle2.center).magnitude()


@_intersects_circle2.register(Point2)
def _(point, circle):
    return (point - circle.center).magnitude() <= circle.radius