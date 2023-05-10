import operator
from euclidian.graycode import signed_gray


def flips(vector):
    for code in signed_gray(vector.dimensionality):
        yield vector.map(operator.mul, code)
