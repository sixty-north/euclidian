from functools import total_ordering, singledispatch
import math
import itertools
from numbers import Real
import sys
from euclidian.cartesian import SpaceMismatchError, Cartesian
from euclidian.cartesian2 import determinant_2, Box2, Point2, Vector2, Direction2
from euclidian.graycode import gray
from euclidian.util import sign, is_zero, all_equal


class Cartesian3(Cartesian):

    DEFAULT_AXES = ('x', 'y', 'z')

    @property
    def dimensionality(self):
        return 3


@total_ordering
class Point3(Cartesian3):
    __slots__ = ['_p']

    @classmethod
    def origin(cls, space=Cartesian3.DEFAULT_AXES):
        return Point3(0, 0, 0, space)

    @classmethod
    def from_vector(cls, vector, space=None):
        s = space if space is not None else vector.space
        return cls(vector[0], vector[1], vector[2], space=s)

    @classmethod
    def as_midpoint(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are in different spaces".format(p, q))
        return cls((p[0] + q[0]) / 2,
                   (p[1] + q[1]) / 2,
                   (p[2] + q[2]) / 2,
                   space=p.space)

    def __init__(self, *args, **kwargs):
        try:
            space = kwargs.pop('space')
        except KeyError:
            try:
                space = args[self.dimensionality]
            except IndexError:
                space = Cartesian3.DEFAULT_AXES
        super().__init__(space)
        try:
            x = kwargs[self.space[0]] if self.space[0] in kwargs else args[0]
            y = kwargs[self.space[1]] if self.space[1] in kwargs else args[1]
            z = kwargs[self.space[2]] if self.space[2] in kwargs else args[2]
        except IndexError:
            raise TypeError("A least {} coordinates must be provided".format(self.dimensionality))
        self._p = (x, y, z)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._p[i]

    def __getitem__(self, index):
        return self._p[index]

    def __sub__(self, rhs):
        if not isinstance(rhs, (Point3, Vector3)):
            return NotImplemented
        if self.space != rhs.space:
            raise SpaceMismatchError("Different spaces")
        if isinstance(rhs, Vector3):
            return Point3(self._p[0] - rhs[0],
                          self._p[1] - rhs[1],
                          self._p[2] - rhs[2])
        return Vector3(self._p[0] - rhs._p[0],
                       self._p[1] - rhs._p[1],
                       self._p[2] - rhs._p[2],
                       space=self.space)

    def __add__(self, rhs):
        if not isinstance(rhs, Vector3):
            return NotImplemented
        return Point3(self._p[0] + rhs._d[0],
                      self._p[1] + rhs._d[1],
                      self._p[2] + rhs._d[2],
                      space=self.space)

    def __abs__(self):
        x = self[0]
        y = self[1]
        z = self[2]
        return math.sqrt(x*x + y*y + z*z)

    def __eq__(self, rhs):
        if not isinstance(rhs, Point3):
            return NotImplemented
        return super().__eq__(rhs) and self._p == rhs._p

    def map(self, f, *items, space=None):
        if not all_equal(item.space for item in itertools.chain([self], items)):
            raise SpaceMismatchError("Not all vectors are in the same space")
        return Point3(*list(itertools.starmap(f, zip(self, *items))),
                      space=space if space is not None else self.space)

    def __ne__(self, rhs):
        if not isinstance(rhs, Point3):
            return NotImplemented
        return not self == rhs

    def __lt__(self, rhs):
        if not isinstance(rhs, Point3):
            return NotImplemented
        if self.space != rhs.space:
            raise SpaceMismatchError("{!r} and {!r} cannot be compared".format(self, rhs))
        return self._p < rhs._p

    def __hash__(self):
        base_hash = super().__hash__()
        return hash((base_hash, self._p))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(axis, coord) for axis, coord in zip(self.space, self._p)))

    def bounding_box(self):
        return Box3(self, self)

    def distance_to(self, point):
        dx = point[0] - self[0]
        dy = point[1] - self[1]
        dz = point[2] - self[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def vector(self):
        """Returns the position vector."""
        return Vector3(*self._p, space=self.space)

    def subspace(self, subspace_axes):
        """Obtain a two-dimensional subspace representation of this vector.

        Args:
            subspace_axes: A 2-sequence of which each element is either and axis names as a string an integer
                axis index.

        Returns:
            A Point2 in the appropriate subspace.
        """
        if len(subspace_axes) != 2:
            raise ValueError("Incompatible subspace axes")

        axis_specs = [self.axis_spec(subspace_axis) for subspace_axis in subspace_axes]

        return Point2(self._p[axis_specs[0].index],
                      self._p[axis_specs[1].index],
                      space=[axis_spec.name for axis_spec in axis_specs])


class Vector3(Cartesian3):
    __slots__ = ['_d']

    @classmethod
    def from_point(cls, point):
        return cls(point[0], point[1], point[2], space=point.space)

    @classmethod
    def orthogonal_to(cls, v, w):
        return v.cross(w)

    def __init__(self, *args, **kwargs):
        try:
            space = kwargs.pop('space')
        except KeyError:
            try:
                space = args[self.dimensionality]
            except IndexError:
                space = Cartesian3.DEFAULT_AXES
        super().__init__(space)
        try:
            x_name = self.space[0]
            y_name = self.space[1]
            z_name = self.space[2]
            dx = kwargs[x_name] if x_name in kwargs else args[0]
            dy = kwargs[y_name] if y_name in kwargs else args[1]
            dz = kwargs[z_name] if z_name in kwargs else args[2]
        except IndexError:
            raise TypeError("A least {} coordinates must be provided".format(self.dimensionality))
        self._d = (dx, dy, dz)

    def map(self, f, *items, space=None):
        if not all_equal(vector.space for vector in itertools.chain([self], items)):
            raise SpaceMismatchError("Not all vectors are in the same space")
        return Vector3(*list(itertools.starmap(f, zip(self, *items))),
                       space=space if space is not None else self.space)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._d[i]

    def __getitem__(self, index):
        return self._d[index]

    def __iter__(self):
        return iter(self._d)

    def items(self):
        return zip(self.space, self._d)

    def __add__(self, rhs):
        if not isinstance(rhs, Vector3):
            return NotImplemented
        return Vector3(self._d[0] + rhs._d[0],
                       self._d[1] + rhs._d[1],
                       self._d[2] + rhs._d[2],
                       space=self.space)

    def __sub__(self, rhs):
        if not isinstance(rhs, Vector3):
            return NotImplemented
        return Vector3(self._d[0] - rhs._d[0],
                       self._d[1] - rhs._d[1],
                       self._d[2] - rhs._d[2],
                       space=self.space)

    def __mul__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector3(self._d[0] * rhs,
                       self._d[1] * rhs,
                       self._d[2] * rhs,
                       space=self.space)

    def __rmul__(self, lhs):
        if not isinstance(lhs, Real):
            return NotImplemented
        return Vector3(lhs * self._d[0],
                       lhs * self._d[1],
                       lhs * self._d[2],
                       space=self.space)

    def __truediv__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector3(self._d[0] / rhs,
                       self._d[1] / rhs,
                       space=self.space)

    def __floordiv__(self, rhs):
        if not isinstance(rhs, Real):
            return NotImplemented
        return Vector3(self._d[0] // rhs,
                       self._d[1] // rhs,
                       self._d[2] // rhs,
                       space=self.space)

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector3(-self._d[0], -self._d[1], -self._d[2])

    def __abs__(self):
        return self.magnitude()

    def magnitude(self):
        return math.sqrt(self.magnitude2())

    def magnitude2(self):
        dx = self._d[0]
        dy = self._d[1]
        dz = self._d[2]
        return dx*dx + dy*dy + dz*dz

    def unit(self):
        m = self.magnitude()
        if m == 0:
            raise ZeroDivisionError("Cannot produce degenerate unit vector")
        return Vector3(self._d[0] / m,
                       self._d[1] / m,
                       self._d[2] / m,
                       space=self.space)

    def direction(self):
        return Direction3(self._d[0], self._d[1], self._d[2], space=self.space)

    def dot(self, rhs):
        return self._d[0] * rhs._d[0] + self._d[1] * rhs._d[1] + self._d[2] * rhs._d[2]

    def cross(self, rhs):
        return Vector3(self._d[1]*rhs._d[2] - self._d[1]*rhs._d[1],
                       self._d[1]*rhs._d[1] - self._d[1]*rhs._d[2],
                       self._d[1]*rhs._d[1] - self._d[1]*rhs._d[1],
                       space=self.space)

    # def det(self, rhs):
    #     """If det (the determinant) is positive the angle between A (this) and B (rhs) is positive (counter-clockwise).
    #     If the determinant is negative the angle goes clockwise. Finally, if the determinant is 0, the
    #     vectors point in the same direction. In Schneider & Eberly this operator is called Kross.
    #     Is is also often known as PerpDot.
    #     """
    #     return determinant_2(self._d[0], self._d[1], rhs._d[0], rhs._d[1])

    # def angle(self, rhs):
    #     return math.atan2(abs(self.determinant(rhs)), self.dot(rhs))

    def angle(self, rhs):
         theta = math.acos(self.dot(rhs) / (self.magnitude() * rhs.magnitude()))
         return theta

    def components(self):
        """Decompose into three component vectors parallel to the basis vectors.

        Returns:
            A 3-tuple containing three Vector3 instances.
        """
        return (Vector3(self._d[0], 0, 0, space=self.space),
                Vector3(0, self._d[1], 0, space=self.space),
                Vector3(0, 0, self._d[2], space=self.space))

    def __eq__(self, rhs):
        if not isinstance(rhs, Vector3):
            return NotImplemented
        return self.space == rhs.space and self._d == rhs._d

    def __ne__(self, rhs):
        if not isinstance(rhs, Vector3):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        base_hash = super().__hash__()
        return hash((base_hash, self._d))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(*item) for item in self.items()))

    def subspace(self, subspace_axes):
        """Obtain a two-dimensional subspace representation of this vector.

        Args:
            subspace_axes: A 2-sequence of which each element is either and axis names as a string an integer
                axis index.

        Returns:
            A Vector2 in the appropriate subspace.
        """
        if len(subspace_axes) != 2:
            raise ValueError("Incompatible subspace axes")

        axis_specs = [self.axis_spec(subspace_axis) for subspace_axis in subspace_axes]

        return Vector2(self._d[axis_specs[0].index],
                       self._d[axis_specs[1].index],
                       space=[axis_spec.name for axis_spec in axis_specs])


class Direction3(Cartesian3):
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
                space = Cartesian3.DEFAULT_AXES
        super().__init__(space)
        try:
            x_name = self.space[0]
            y_name = self.space[1]
            z_name = self.space[2]
            dx = kwargs[x_name] if x_name in kwargs else args[0]
            dy = kwargs[y_name] if y_name in kwargs else args[1]
            dz = kwargs[z_name] if z_name in kwargs else args[2]
        except IndexError:
            raise TypeError("A least {} coordinates must be provided".format(self.dimensionality))
        if dx == dy == dz == 0:
            raise ValueError("Degenerate {}".format(self.__class__.__name__))
        self._d = (dx, dy, dz)

    def __getattr__(self, axis):
        try:
            i = self._space.index(axis)
        except ValueError:
            raise AttributeError("Axis '{}' not recognized.".format(axis))
        return self._d[i]

    def __getitem__(self, index):
        return self._d[index]

    def vector(self):
        return Vector3(self._d[0], self._d[1], self._d[2], space=self.space)

    def __eq__(self, rhs):
        if not isinstance(rhs, Direction3):
            return NotImplemented
        # TODO!
        raise NotImplementedError

    def __ne__(self, rhs):
        if not isinstance(rhs, Direction3):
            return NotImplemented
        return not self == rhs

    def __pos__(self):
        return self

    def __neg__(self):
        return Direction3(-self._d[0], -self._d[1], -self._d[2], space=self.space)

    def __hash__(self):
        base_hash = super().__hash__()
        return hash((base_hash, self._d))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(axis, coord) for axis, coord in zip(self.space, self._d)))

    def angle(self, rhs):
        return self.vector().angle(rhs.vector())

    def scalar_projection(self, vector):
        """Scalar projection onto this direction."""
        return self.vector().unit().dot(vector)

    def vector_projection(self, vector):
        """The projection of a vector onto this direction.
        """
        unit = self.vector().unit()
        return unit.dot(vector) * unit

    def vector_rejection(self, vector):
        return vector - self.vector_projection(vector)

    def subspace(self, subspace_axes):
        """Obtain a two-dimensional subspace representation of this vector.

        Args:
            subspace_axes: A 2-sequence of which each element is either and axis names as a string an integer
                axis index.

        Returns:
            A Direction2 in the appropriate subspace.
        """
        if len(subspace_axes) != 2:
            raise ValueError("Incompatible subspace axes")

        axis_specs = [self.axis_spec(subspace_axis) for subspace_axis in subspace_axes]

        return Direction2(self._c[axis_specs[0].index],
                          self._c[axis_specs[1].index],
                          space=[axis_spec.name for axis_spec in axis_specs])


class Box3(Cartesian3):
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
        min_z = first[2]
        max_x = min_x
        max_y = min_y
        max_z = min_z


        for index, point in enumerate(iterator, start=1):
            if point.space != space:
                raise SpaceMismatchError(
                    "Point at index {i} {!r} is not in the same space as first {!r}".format(index, point, first))
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            min_z = min(min_z, point[2])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
            max_z = max(max_z, point[2])

        return cls.from_extents(min_x - border, min_y - border, min_z - border,
                                max_x + border, max_y + border, max_z + border)

    @classmethod
    def from_extents(cls, min_x, min_y, min_z, max_x, max_y, max_z, space=Cartesian3.DEFAULT_AXES):
        return cls(Point3(min_x, min_y, min_z, space), Point3(max_x, max_y, max_z, space))

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
        min_z = first_bbox.min[2]
        max_x = first_bbox.max[0]
        max_y = first_bbox.max[1]
        max_z = first_bbox.max[2]

        for index, bounded in enumerate(iterator, start=1):
            if bounded.space != space:
                raise SpaceMismatchError(
                    "Bounded object at index {i} {!r} is not in the same space as first {!r}".format(index, bounded,
                                                                                                     first))

            bbox = bounded.bounding_box()
            min_x = min(min_x, bbox.min[0])
            min_y = min(min_y, bbox.min[1])
            min_z = min(min_z, bbox.min[2])
            max_x = max(max_x, bbox.max[0])
            max_y = max(max_y, bbox.max[1])
            max_z = max(max_z, bbox.max[2])
        return cls.from_extents(min_x, min_y, min_z, max_x, max_y, max_z)

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

    def vertex(self, i, j, k):
        return Point3(self._p[i][0],
                      self._p[j][1],
                      self._p[k][2],
                      space=self.space)

    def vertices(self):
        return (self.vertex(*indices) for indices in gray(self.dimensionality))

    def edge(self, i=None, j=None, k=None):
        if (i, j, k).count(None) != 1:
            raise "Overspecified Box3 edge with i={}, j={} and k={}. Exactly two arguments should be provided.".format(i, j, k)
        if i is None:
            return Segment3(self.vertex(0, j, k),
                            self.vertex(1, j, k))
        if j is None:
            return Segment3(self.vertex(i, 0, k),
                            self.vertex(i, 1, k))
        if k is None:
            return Segment3(self.vertex(i, j, 0),
                            self.vertex(i, j, 1))
        assert False, "Programming error"

    def edges(self):
        yield from (self.edge(i=i, j=j) for i, j in gray(self.dimensionality - 1))
        yield from (self.edge(i=i, k=k) for i, k in gray(self.dimensionality - 1))
        yield from (self.edge(j=j, k=k) for j, k in gray(self.dimensionality - 1))

    def intersects(self, obj):
        return _intersects_box3(obj, self)

    def vector(self):
        return self._p[1] - self._p[0]

    def __getitem__(self, index):
        return self._p[index]

    def __eq__(self, rhs):
        if not isinstance(rhs, Box3):
            return NotImplemented
        return self.space == rhs.space and self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Box3):
            return NotImplemented
        return not self == rhs

    def __hash__(self):
        base_hash = super().__hash__()
        return hash((base_hash, self._p))

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._p[0], self._p[1])

    def distance_to(self, point):
        raise NotImplemented
        # TODO: Distance to faces

    def subspace(self, subspace_axes):
        """Obtain a two-dimensional subspace representation of this vector.

        Args:
            subspace_axes: A 2-sequence of which each element is either and axis names as a string an integer
                axis index.

        Returns:
            A Box2 in the appropriate subspace.
        """
        if len(subspace_axes) != 2:
            raise ValueError("Incompatible subspace axes")

        axis_specs = [self.axis_spec(subspace_axis) for subspace_axis in subspace_axes]

        return Box2.from_extents(self.min[axis_specs[0].index], self.min[axis_specs[1].index],
                                 self.max[axis_specs[0].index], self.max[axis_specs[1].index],
                                 space=[axis_spec.name for axis_spec in axis_specs])


@singledispatch
def _intersects_box3(obj, box):
    raise NotImplementedError("Intersection between {!r} and {!r} not supported".format(box, obj))


@_intersects_box3.register(Point3)
def _(point, box):
    if point.space != box.space:
        raise SpaceMismatchError("{!r} is not in the same space as {!r}".format(point, box))
    return all(box.min[c] <= point[c] <= box.max[c] for c in box.dimensionality)


class Line3(Cartesian3):
    __slots__ = ['_point', '_vector']

    @classmethod
    def through_points(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))
        v = q - p
        return cls(p, v)

    @classmethod
    def from_point_and_direction(cls, p, d):
        if p.space != d.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, d))
        v = d.vector()
        return cls(p, v)

    @classmethod
    def supporting_segment(cls, segment):
        return segment.supporting_line()

    @classmethod
    def supporting_ray(cls, ray):
        raise ray.supporting_line()

    def __init__(self, point, vector):
        if point.space != vector.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, vector))
        super().__init__(point.space)
        self._point = point
        self._vector = vector

    def opposite(self):
        return Line3(self._point, -self._vector)

    def perpendicular(self):
        return Plane3.from_point_and_normal(self._point, self._vector)

    def direction(self):
        raise NotImplementedError

    def projection_from(self, p):
        if p.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, self))
        v = self._vector
        w = p - self._point
        c1 = w.dot(v)
        c2 = v.dot(v)
        b = c1 / c2
        return self._point + b * v

    def distance_to(self, p):
        projection = self.projection_from(p)
        return self._point.distance_to(projection)

    def vector(self):
        """Vector parallel the line"""
        return self._vector

    def point(self, i=0):
        """Generate a point on the line.

        Args:
            i: By providing different values of i (which defaults to 0) distinct
               points on the line can be produced.

        Returns:
            A Point3 on the line.
        """
        return self._point + i * self._vector

    def is_parallel_to(self, line):
        v1 = self._vector
        v2 = line._vector
        s = v1.dot(v2) / (v1.magnitude() * v2.magnitude())
        return s >= 1 - sys.float_info.epsilon

    def has_on(self, point):
        return collinear(self._point,
                         self._vector + self._point,
                         point)

    def __eq__(self, rhs):
        if not isinstance(rhs, Line3):
            return NotImplemented
        if self.space != rhs.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(self, rhs))
        return collinear(self.point(0), self.point(1), rhs.point(0)) and collinear(self.point(0), self.point(1), rhs.point(1))

    def __ne__(self, rhs):
        return not self == rhs

    def __hash__(self):
        return hash((self._point, self._vector))

    def __repr__(self):
        return '{}({}, {}, space={})'.format(self.__class__.__name__, self._point, self._vector, self.space)


def collinear(p, q, r):
    px, py, pz = p
    qx, qy, qz = q
    rx, ry, rz = r

    dpx = px-rx
    dqx = qx-rx
    dpy = py-ry
    dqy = qy-ry
    if sign(determinant_2(dpx, dqx, dpy, dqy)) != 0:
        return False

    dpz = pz-rz
    dqz = qz-rz
    return (sign(determinant_2(dpx, dqx, dpz, dqz)) == 0
        and sign(determinant_2(dpy, dqy, dpz, dqz)) == 0)


class Ray3(Cartesian3):
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
        return Line3.through_points(*self._p)

    def opposite(self):
        return Ray3.from_source_and_vector(self.source, self.source - self.point)

    def lerp(self, t):
        # TODO: Consider using this version which is more numerically stable lerp(p0, p1, t) = p0 * (1 - t) + p1 * t
        u = (self.point - self.source).unit()
        return self.source + u * t

    def distance_to(self, p):
        if p.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, self))

        v = self.vector()
        w = p - self._p[0]

        c1 = w.dot(v)
        if c1 <= 0:
            return p.distance_to(self._p[0])

        c2 = v.dot(v)
        b = c1 / c2
        projection = self._p[0] + b * v
        return p.distance_to(projection)

    def projected_from(self, p):
        """Project a point onto this ray.

        Args:
            point: The point to be projected.

        Return:
            The perpendicular projection of point onto this ray, or None.
        """
        if p.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, self))

        v = self.vector()
        w = p - self._p[0]

        c1 = w.dot(v)
        if c1 <= 0:
            return None

        c2 = v.dot(v)
        b = c1 / c2
        return self._p[0] + b * v

    def __eq__(self, rhs):
        if not isinstance(rhs, Ray3):
            return NotImplemented
        return self.source == rhs.source and self.direction() == rhs.direction()

    def __ne__(self, rhs):
        return not self == rhs

    def __hash__(self):
        return hash((self.source, self.direction()))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._p[0], self._p[1])


class Segment3(Cartesian3):
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

    def vector(self):
        return self.target - self.source

    def direction(self):
        return self.vector().direction()

    def supporting_line(self):
        return Line3.through_points(*self._p)

    def midpoint(self):
        return Point3.as_midpoint(self.source, self.target)

    def length(self):
        dx = self.target[0] - self.source[0]
        dy = self.target[1] - self.source[1]
        dz = self.target[2] - self.source[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def reversed(self):
        return Segment3(self.target, self.source)

    def lerp(self, t):
        # TODO: Consider using this version which is more numerically stable lerp(p0, p1, t) = p0 * (1 - t) + p1 * t
        return Point3(self.source[0] + t * (self.target[0] - self.source[0]),
                      self.source[1] + t * (self.target[1] - self.source[1]),
                      self.source[2] + t * (self.target[2] - self.source[2]),
                      space=self.space)

    def bounding_box(self):
        return Box3(*self._p)

    def distance_to(self, p):
        if p.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, self))

        v = self.vector()
        w = p - self._p[0]

        c1 = w.dot(v)
        if c1 <= 0:
            return p.distance_to(self._p[0])

        c2 = v.dot(v)
        if c2 <= c1:
            return p.distance_to(self._p[1])

        b = c1 / c2
        projection = self._p[0] + b * v
        return p.distance_to(projection)

    def projected_from(self, p):
        """Project a point onto this segment.

        Args:
            point: The point to be projected.

        Return:
            The perpendicular projection of point onto this segment, or None.
        """
        if p.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, self))

        v = self.vector()
        w = p - self._p[0]

        c1 = w.dot(v)
        if c1 <= 0:
            return None

        c2 = v.dot(v)
        if c2 <= c1:
            return None

        b = c1 / c2
        return self._p[0] + b * v

    def __eq__(self, rhs):
        if not isinstance(rhs, Segment3):
            return NotImplemented
        return self._p == rhs._p

    def __ne__(self, rhs):
        if not isinstance(rhs, Segment3):
            return NotImplemented
        return self._p != rhs._p

    def __hash__(self):
        return hash(self._p)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._p[0], self._p[1])


class DegeneracyError(Exception):
    pass


class Plane3(Cartesian3):
    __slots__ = ['_c']

    @classmethod
    def through_points(cls, p, q, r):

        if not all_equal(point.space for point in (p, q, r)):
            raise SpaceMismatchError("{!r}, {!r} and {!r} are not in the same space".format(p, q, r))

        if collinear(p, q, r):
            raise DegeneracyError("{}, {} and {} are collinear and cannot uniquely specify a Plane3".format(p, q, r))

        px, py, pz = p
        qx, qy, qz = q
        rx, ry, rz = r

        rpx = px - rx
        rpy = py - ry
        rpz = pz - rz
        rqx = qx - rx
        rqy = qy - ry
        rqz = qz - rz

        a = rpy*rqz - rqy*rpz
        b = rpz*rqx - rqz*rpx
        c = rpx*rqy - rqx*rpy
        d = -a*rx - b*ry - c*rz
        return cls(a, b, c, d, space=p.space)

    @classmethod
    def bisecting_points(cls, p, q):
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))

        px, py, pz = p
        qx, qy, qz = q
        a = 2 * (px - qx)
        b = 2 * (py - qy)
        c = 2 * (pz - qz)
        d = qx*qx + qy*qy + qz*qz - px*px - py*py - pz*pz
        return cls(a, b, c, d, space=p.space)

    @classmethod
    def bisecting_planes(cls, p, q):
        """Determine the bisector of two planes.

        Args:
            p, q: The two planes to be bisected.

        Returns:
            The plane bisecting planes p and q.

        Raises:
            SpaceMismatchError: If planes p and q are not in the same space.
        """
        if p.space != q.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, q))

        pa, pb, pc, pd = p
        qa, qb, qc, qd = q

        pn = math.sqrt(pa*pa + pb*pb + pc*pc)
        qn = math.sqrt(qa*qa + qb*qb + qc*qc)

        a = qn * pa + pn * qa
        b = qn * pb + pn * qb
        c = qn * pc + pn * qc
        d = qn * pd + pn * qd

        if a == b == c == 0:
            a = qn * pa - pn * qa
            b = qn * pb - pn * qb
            c = qn * pc - pn * qc
            d = qn * pd - pn * qd

        return cls(a, b, c, d, space=p.space)

    @classmethod
    def from_point_and_normal(cls, p, n):
        if p.space != n.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(p, n))

        dx = n[0]
        dy = n[1]
        dz = n[2]
        return cls(dx, dy, dz, -dx*p[0] - dy*p[1] - dz*p[2], space=p.space)

    @classmethod
    def through_line_and_point(cls, line, point):
        return cls.through_points(line.point(0),
                                  line.point(1),
                                  point)

    @classmethod
    def through_segment_and_point(cls, segment, point):
        return cls.through_points(segment.source,
                                  segment.target,
                                  point)

    @classmethod
    def through_ray_and_point(cls, ray, point):
        return cls.through_points(ray.source(),
                                  ray.point(),
                                  point)

    def __init__(self, a, b, c, d, space=Cartesian3.DEFAULT_AXES):
        """A plane defined by ax + by + cz + d = 0"""
        super().__init__(space)
        self._c = (a, b, c, d)

    @property
    def a(self):
        return self._c[0]

    @property
    def b(self):
        return self._c[1]

    @property
    def c(self):
        return self._c[2]

    @property
    def d(self):
        return self._c[3]

    def __getitem__(self, index):
        return self._c[index]

    def point(self, i=0, j=0):
        if i == 0 and j == 0:
            return self._point()

        base_i = self.base(0)
        base_j = self.base(1)

        return self._point() + i * base_i + j * base_j

    def _point(self):
        x = y = z = 0
        if not is_zero(self.a):
            x = -self.d / self.a
        elif not is_zero(self.b):
            y = -self.d / self.b
        else:
            z = -self.d / self.c
        return Point3(x, y, z, space=self.space)

    def _parse_solve_args(self, args, kwargs, first_axis_index, second_axis_index):
        """Extract values from the supplied positional and keyword arguments according to axis names.
        """
        first_name = self.space[first_axis_index]
        second_name = self.space[second_axis_index]
        try:
            x = kwargs[first_name] if first_name in kwargs else args[first_axis_index]
            y = kwargs[second_name] if second_name in kwargs else args[second_axis_index]
        except IndexError:
            raise TypeError("At least {} coordinates must be provided for axes '{}' and '{}'".format(
                self.dimensionality - 1, first_name, second_name))
        return x, y

    def _solve_for_2(self, *args, **kwargs):
        """Solve for axis 2 (the z axis) providing coordinates for axis 0 and axis 1 as either positional or
        named arguments consistent with space axis naming."""
        x, y = self._parse_solve_args(args, kwargs, 0, 1)

        a, b, c, d = self._c
        if c == 0:
            raise ZeroDivisionError("Cannot solve for plane parallel to axis")
        z = -(a*x + b*y + d) / c
        return z

    def _solve_for_1(self, *args, **kwargs):
        """Solve for axis 1 (the y axis) providing coordinates for axis 0 and axis 2 as either positional or
        named arguments consistent with space axis naming."""
        x, z = self._parse_solve_args(args, kwargs, 0, 2)
        a, b, c, d = self._c
        if b == 0:
            raise ZeroDivisionError("Cannot solve for plane parallel to axis")
        y = -(a*x + c*z + d) / b
        return y

    def _solve_for_0(self, *args, **kwargs):
        """Solve for axis 0 (the x axis) providing coordinates for axis 1 and axis 2 as either positional or
        named arguments consistent with space axis naming."""
        y, z = self._parse_solve_args(args, kwargs, 1, 2)
        a, b, c, d = self._c
        if b == 0:
            raise ZeroDivisionError("Cannot solve for plane parallel to axis")
        x = -(b*y + c*z + d) / a
        return x

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
                elif index == 2:
                    return self._solve_for_2
                assert False, "We never reach here."
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))

    def distance_to(self, point):
        if point.space != self.space:
            raise SpaceMismatchError("{!r} and {!r} are not in the same space".format(point, self))
        a, b, c, d = self._c
        return (point[0] * a
              + point[1] * b
              + point[2] * c
              + c) / math.sqrt(a*a + b*b + c*c)

    def projected_from(self, point):
        a, b, c, d = self._c
        x, y, z = point
        num = a*x + b*y + c*z + d
        den = a*a + b*b + c*c
        q = num / den

        return Point3(x - q * a,
                      y - q * b,
                      z - q * c,
                      space=self.space)

    def perpendicular(self, point):
        return Line3.from_point_and_direction(point, self.normal_direction())

    def opposite(self):
        raise NotImplementedError

    def normal_vector(self):
        return Vector3(self.a, self.b, self.c, space=self.space)

    def normal_direction(self):
        p = self.point()
        q = p + self.normal_vector()
        return Direction3(p, q, space=self.space)

    def base(self, index):
        """Obtain one of two mutually perpendicular vectors embedded in the plane.

        Args:
            index: Zero or one, to select which vector to return. The zeroth vector will always be
               parallel to the plane formed by the 0 and 1 (i.e. x and y) axes; in other words it will
               be horizontal. The vector associated with index one will be orthogonal to the normal direction and the
               zeroth base vector.

        Returns:
            A base vector. There is no guarantee that the returned vector has unit length.
        """
        if index != 0 or index != 1:
            raise ValueError("Plane base vector index must be 0 or 1")

        if index == 1:
            return Vector3.orthogonal_to(self.normal_vector(), self.base(0))

        if is_zero(self.a):
            return Vector3(1, 0, 0, space=self.space)

        if is_zero(self.b):
            return Vector3(0, 1, 0, space=self.space)

        return Vector3(-self.b, self.a, 0, space=self.space)  # Horizontal

    def is_degenerate(self):
        return self.a == 0.0 and self.b == 0.0 and self.c == 0.0

    def __repr__(self):
        return '{}(a={}, b={}, c={}, d={})'.format(self.__class__.__name__, *self._c)


class Transform3:

    @staticmethod
    def identity():
        return IDENTITY_TRANSFORM3

    # @classmethod
    # def from_rotation(cls, angle_radians, axis):
    #      cos_theta = math.cos(angle_radians)
    #      sin_theta = math.sin(angle_radians)
    #
    #      unit_axis = axis.unit()
    #      l = unit_axis[0]
    #      m = unit_axis[1]
    #      n = unit_axis[2]
    #
    #      a = l*l*(1 - cos_theta) + cos_theta
    #      b = l*m*(1 - cos_theta) + n*sin_theta
    #      c = l*n*(1 - cos_theta) - m*sin_theta
    #
    #      d = m*l*(1 - cos_theta) - n*sin_theta
    #      e = m*m*(1 - cos_theta) + cos_theta
    #      f = m*n*(1 - cos_theta) + l*sin_theta
    #
    #      g = n*l*(1 - cos_theta) + m*sin_theta
    #      h = n*m*(1 - cos_theta) - l*sin_theta
    #      i = n*n*(1 - cos_theta) + cos_theta
    #
    #      return cls(a, d, g, b, e, h, c, f, i, 0, 0, 0)


    @classmethod
    def from_rotation(cls, angle_radians, axis):
         cos_theta = math.cos(angle_radians)
         sin_theta = math.sin(angle_radians)

         unit_axis = axis.unit()
         u = unit_axis[0]
         v = unit_axis[1]
         w = unit_axis[2]

         a = u*u + (1 - u*u)*cos_theta
         b = u*v*(1 - cos_theta) + w*sin_theta
         c = u*w*(1 - cos_theta) - v*sin_theta

         d = u*v*(1 - cos_theta) - w*sin_theta
         e = v*v + (1 - v*v)*cos_theta
         f = v*w*(1 - cos_theta) + u*sin_theta

         g = u*w*(1 - cos_theta) + v*sin_theta
         h = v*w*(1 - cos_theta) - u*sin_theta
         i = w*w + (1 - w*w)*cos_theta

         return cls(a, d, g, b, e, h, c, f, i, 0, 0, 0)

    @classmethod
    def from_rotation_degrees(cls, angle_degrees, axis):
        return cls.from_rotation(math.radians(angle_degrees), axis)

    @classmethod
    def from_rotation_x(cls, angle_radians):
        return cls.identity().rotate_x(angle_radians)

    @classmethod
    def from_rotation_y(cls, angle_radians):
        return cls.identity().rotate_y(angle_radians)

    @classmethod
    def from_rotation_z(cls, angle_radians):
        return cls.identity().rotate_z(angle_radians)

    @classmethod
    def from_scale(cls, scale_factor, center=None):
       # TODO: Optimise
       return cls.identity().scale(scale_factor, center)

    @classmethod
    def from_translation(cls, vector):
        return cls(1, 0, 0, 0, 1, 0, 0, 0, 1, vector[0], vector[1], vector[2])

    def __init__(self, a=0, d=0, g=0, b=0, e=0, h=0, c=0, f=0, i=0, tx=0, ty=0, tz=0):
        self._m = ((a, d, g, tx),
                   (b, e, h, ty),
                   (c, f, i, tz))

    @property
    def a(self):
        return self._m[0][0]

    @property
    def b(self):
        return self._m[1][0]

    @property
    def c(self):
        return self._m[2][0]

    @property
    def d(self):
        return self._m[0][1]

    @property
    def e(self):
        return self._m[1][1]

    @property
    def f(self):
        return self._m[2][1]

    @property
    def g(self):
        return self._m[0][2]

    @property
    def h(self):
        return self._m[1][2]

    @property
    def i(self):
        return self._m[2][2]

    @property
    def tx(self):
        # TODO: The name should reflect the space
        return self._m[0][3]

    @property
    def ty(self):
        # TODO: The name should reflect the space
        return self._m[1][3]

    @property
    def tz(self):
        # TODO: The name should reflect the space
        return self._m[2][3]

    def __eq__(self, rhs):
        if not isinstance(rhs, Transform3):
            return NotImplemented
        return self._m == rhs._m

    def __ne__(self, rhs):
        return not self == rhs

    def __repr__(self):
        return '{}(a={}, d={}, g={}, b={}, e={}, h={}, c={}, f={}, i={}, tx={}, ty={}, tz={})'.format(
            self.__class__.__name__,
            self.a, self.d, self.g, self.b, self.e, self.h, self.c, self.f, self.i, self.tx, self.ty, self.tz)

    def __mul__(self, rhs):
        if not isinstance(rhs, Transform3):
            return NotImplemented

        a1 = self.a
        b1 = self.b
        c1 = self.c

        d1 = self.d
        e1 = self.e
        f1 = self.f

        g1 = self.g
        h1 = self.h
        i1 = self.i

        tx1 = self.tx
        ty1 = self.ty
        tz1 = self.tz


        a2 = rhs.a
        b2 = rhs.b
        c2 = rhs.c

        d2 = rhs.d
        e2 = rhs.e
        f2 = rhs.f

        g2 = rhs.g
        h2 = rhs.h
        i2 = rhs.i

        tx2 = rhs.tx
        ty2 = rhs.ty
        tz2 = rhs.tz

        a = a1 * a2 + d1 * b2 + g1 * c2
        b = b1 * a2 + e1 * b2 + h1 * c2
        c = c1 * a2 + f1 * b2 + i1 * c2

        d = a1 * d2 + d1 * e2 + g1 * f2
        e = b1 * d2 + e1 * e2 + h1 * f2
        f = c1 * d2 + f1 * e2 + i1 * f2

        g = a1 * g2 + d1 * h2 + g1 * i2
        h = b1 * g2 + e1 * h2 + h1 * i2
        i = c1 * g2 + f1 * h2 + i1 * i2

        tx = a1 * tx2 + d1 * ty2 + g1 * tz2 + tx1
        ty = b1 * tx2 + e1 * ty2 + h1 * tz2 + ty1
        tz = c1 * tx2 + f1 * ty2 + i1 * tz2 + tz1

        return Transform3(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i, tx=tx, ty=ty, tz=tz)

    # TODO: __rmul__

    def translate(self, vector):
        tx = vector.x * self.a + vector.y * self.b + vector.z * self.c
        ty = vector.x * self.d + vector.y * self.e + vector.z * self.f
        tz = vector.x * self.g + vector.y * self.h + vector.z * self.i
        return self.replace(tx=tx, ty=ty, tz=tz)

    def scale(self, scale_factor, center=None):
        s = Vector3(scale_factor, scale_factor, scale_factor) if isinstance(scale_factor, Real) else scale_factor

        if center is None:
            return self.replace(a=self.a * s.x,
                                d=self.d * s.x,
                                g=self.g * s.x,
                                b=self.b * s.y,
                                e=self.e * s.y,
                                h=self.h * s.y,
                                c=self.c * s.z,
                                f=self.f * s.z,
                                i=self.i * s.z)

        v = center.vector()
        return self.translate(v).scale(scale_factor).translate(-v)


    def replace(self, **kwargs):
        return Transform3(
            a=kwargs.get('a', self.a),
            d=kwargs.get('d', self.d),
            g=kwargs.get('g', self.g),
            b=kwargs.get('b', self.b),
            e=kwargs.get('e', self.e),
            h=kwargs.get('h', self.h),
            c=kwargs.get('c', self.c),
            f=kwargs.get('f', self.f),
            i=kwargs.get('i', self.i),
            tx=kwargs.get('tx', self.tx),
            ty=kwargs.get('ty', self.ty),
            tz=kwargs.get('tz', self.tz))

    # def __call__(self, obj):
    #     """Transform obj"""
    #     return transform3(obj, self)

    def rotate(self, angle_radians, axis_vector=None):
        return self.from_rotation(angle_radians, axis_vector) * self

    # TODO: Use axis names
    def rotate_z(self, psi):
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)
        return Transform3(
            a=cos_psi,
            d=sin_psi,
            b=-sin_psi,
            e=cos_psi,
            i=1
        ) * self

    def rotate_x(self, theta):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return Transform3(
            a=1,
            e=cos_theta,
            f=-sin_theta,
            h=sin_theta,
            i=cos_theta
        ) * self

    def rotate_y(self, phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return Transform3(
            a=cos_phi,
            c=-sin_phi,
            e=1,
            g=sin_phi,
            i=cos_phi
        ) * self

    def is_identity(self):
        return self.a == 1 and self.d == 0 and self.c == 0 and self.b == 0 and self.e == 1 and self.h == 0 and self.c == 0 and self.f == 0 and self.i == 1 and self.tx == 0 and self.tx == 0 and self.tz == 0

IDENTITY_TRANSFORM3 = Transform3(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)


@singledispatch
def transform3(obj, transform):
    raise NotImplemented("Transformation of {!r} not implemented".format(obj))


@transform3.register(Point3)
def _(point, transform):
    return Point3(point.x * transform.a + point.y * transform.b + point.z * transform.c + transform.tx,
                  point.x * transform.d + point.y * transform.e + point.z * transform.f + transform.ty,
                  point.x * transform.g + point.y * transform.h + point.z * transform.i + transform.tz)


@transform3.register(Vector3)
def _(vector, transform):
    return Vector3(vector.x * transform.a + vector.y * transform.b + vector.z * transform.c,
                   vector.x * transform.d + vector.y * transform.e + vector.z * transform.f,
                   vector.x * transform.g + vector.y * transform.h + vector.z * transform.i)


@transform3.register(Direction3)
def _(direction, transform):
    return Direction3(direction.x * transform.a + direction.y * transform.b + direction.z * transform.c,
                      direction.x * transform.d + direction.y * transform.e + direction.z * transform.f,
                      direction.x * transform.g + direction.y * transform.h + direction.z * transform.i)


@transform3.register(Segment3)
def _(segment, transform):
    return Segment3(transform3(segment.source, transform),
                    transform3(segment.target, transform))