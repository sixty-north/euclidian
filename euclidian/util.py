import sys


def sign(x):
    return (x > 0) - (x < 0)

def is_zero(x):
    # future proofing for introduction of tolerance
    return x == 0


def all_equal(iterable):
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("all_equal() cannot be used on an empty iterable series")

    for item in iterator:
        if item != first:
            return False
    return True


def normalise(*args):
    s = sum(args)
    return tuple(arg / s for arg in args)


def almost_equal(x, y, epsilon=sys.float_info.epsilon):
    max_xy_one = max(1.0, abs(x), abs(y))
    e = epsilon * max_xy_one
    delta = abs(x - y)
    return delta <= e