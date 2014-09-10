from functools import singledispatch

from euclidian.cartesian2 import Box2, Point2, Segment2, Line2, Ray2


def clip_segment_to_box(segment, box):
    """Clip a line segment to a box.

    Args:
        segment (Segment2): The segment to be clipped.
        box (Box2): The clipping window.

    Returns:
        A segment clipped to the box, or None if no portion of the
        provided segment lay within the box.
    """
    t0 = 0.0
    t1 = 1.0

    d = segment.vector()

    edges = ((-d[0], (segment.source[0] - box.min[0])),
             ( d[0], (box.max[0] - segment.source[0])),
             (-d[1], (segment.source[1] - box.min[1])),
             ( d[1], (box.max[1] - segment.source[1])))

    for p, q in edges:
        r = q / p
        if p == 0 and q < 0:
            # Parallel line outside the box
            return None

        if p < 0:
            if r > t1:
                return None
            t0 = max(r, t0)
        elif p > 0:
            if r < t0:
                return None
            t1 = min(r, t1)

    if t0 == 0.0 and t1 == 1.0:
        return segment

    s = segment.lerp(t0)
    t = segment.lerp(t1)

    return type(segment)(s, t)


def clip_line_to_box(line, box):
    """Clip a line to a box.

    Args:
        line (Line2): The line to be clipped.
        box (Box2): The clipping window.

    Returns:
        A segment of the line clipped to the box, or None if no portion of the
        provided line lay within the box.
    """
    intersections = sorted((Point2(box.min[0],
                                   line.solve_for_y(box.min[0])),

                            Point2(box.max[0],
                                   line.solve_for_y(box.max[0])),

                            Point2(line.solve_for_x(box.min[1]),
                                   box.min[1]),

                            Point2(line.solve_for_x(box.max[1]),
                                   box.max[1])))
    return clip_segment_to_box(Segment2(intersections[0], intersections[3]), box)


def clip_ray_to_box(ray, box):
    """Clip a ray to a box.

    Args:
        ray (Ray2): The line to be clipped.
        box (Box2): The clipping window.

    Returns:
        A segment of the ray clipped to the box, or None if no portion of the
        provided ray lay within the box.
    """
    points = sorted((ray.source,
                     ray.point,
                     Point2(box.min[0],
                            line.solve_for_y(box.min[0])),

                     Point2(box.max[0],
                            line.solve_for_y(box.max[0])),

                     Point2(line.solve_for_x(box.min[1]),
                            box.min[1]),

                     Point2(line.solve_for_x(box.max[1]),
                            box.max[1])))

    s = find_by_identity(ray.source, points)
    p = find_by_identity(ray.point, points)

    segment = Segment2(ray.source, points[-1]) if s < p else Segment2(ray.source, points[0])

    return clip_segment_to_box(segment, box)


def find_by_identity(item, iterable):
    """Finds an the index of an item using equality of identity.

    Args:
        item: The item to find.
        iterable: An iterable series potentially containing item.

    Returns:
        The index of item in iterable.

    Raises:
        ValueError: If item is not present in iterable.
    """
    for index, current in enumerate(iterable):
        if current is item:
            return index
    raise ValueError("{} is not in the iterable series".format(item))


def clip_point_to_box(point, box):
    """Clip a line to a box.

    Args:
        point (Line2): The line to be clipped.
        box (Box2): The clipping window.

    Returns:
        The point within the box, or None if the point is outside the box.
    """
    return point if ((box.min[0] <= point[0] <= box.max[0])
                 and (box.min[1] <= point[1] <= box.max[0])) else None


@singledispatch
def clip_to_box(obj, box):
    raise NotImplementedError("{!r} to {!r} clipping not implemented".format(obj, box))

@clip_to_box.register(Point2)
def _(point, box):
    return clip_point_to_box(point, box)

@clip_to_box.register(Line2)
def _(line, box):
    return clip_line_to_box(line, box)

@clip_to_box.register(Segment2)
def _(segment, box):
    return clip_segment_to_box(segment, box)

@clip_to_box.register(Ray2)
def _(ray, box):
    return clip_ray_to_box(ray, box)

if __name__ == '__main__':



    box = Box2(Point2(0.0, 0.0), Point2(1.0, 1.0))
    a = Point2(0.25, -0.1)
    b = Point2(0.75, 1.1)
    segment = Segment2(a, b)
    clipped = clip_segment_to_box(segment, box)
    print(clipped)

    line = Line2.through_points(a, b)

    clipped2 = clip_line_to_box(line, box)
    print(clipped2)

    ray = Ray2(a, b)
    clipped3 = clip_ray_to_box(ray, box)
    print(clipped3)

    clipped4 = clip_ray_to_box(ray.opposite(), box)
    print(clipped4)





