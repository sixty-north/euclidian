from abc import ABCMeta, abstractmethod
from collections import namedtuple
from numbers import Integral


class Cartesian(metaclass=ABCMeta):
    __slots__ = ['_space']

    def __init__(self, axes):
        if len(axes) != self.dimensionality:
            raise ValueError("{} cannot have {} space".format(self.__class__.__name__, len(axes)))
        self._space = tuple(axes)

    @property
    @abstractmethod
    def dimensionality(self):
        raise NotImplementedError("Cartesian subclasses must override the dimensionality property")

    @property
    def space(self):
        return self._space

    def __eq__(self, rhs):
        if not isinstance(rhs, Cartesian):
            return NotImplemented
        return self._space == rhs._space

    def __ne__(self, rhs):
        return not self == rhs

    def __repr__(self):
        if all(len(axis) == 1 for axis in self._space):
            return '{}({})'.format(self.__class__.__name__, ''.join(self._space))
        return '{}({!r})'.format(self.__class__.__name__, self._space)

    def __hash__(self):
        return hash(self._space)

    def axis_spec(self, axis):
        """Given an axis as either a name or index.

        Args:
            axis: Either an integer axis index or a string axis name.

        Returns:
            An AxisSpec containing both the axis index and the axis name.
        """
        if isinstance(axis, Integral):
            index = axis
            try:
                name = self._space[index]
            except ValueError as e:
                raise ValueError("Axis index {} out of range".format(index)) from e
        else:
            name = axis
            try:
                index = self._space.index(name)
            except ValueError as e:
                raise ValueError("Unrecognised axis name {!r}".format(name)) from e
        return AxisSpec(index, name)


class SpaceMismatchError(Exception):
    pass


AxisSpec = namedtuple('AxisSpec', ['index', 'name'])