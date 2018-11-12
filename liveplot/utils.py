from numbers import Real
from typing import Union, Sequence


__all__ = ["check_valid_color"]


ValidColor = Union[str, Real, Sequence[Real], None]


def _is_valid_color(c: ValidColor) -> bool:
    """ Checks if `c` is a valid color argument for matplotlib.

        Parameters
        ----------
        c : Union[str, Real, Sequence[Real], NoneType]

        Returns
        -------
        bool """
    from matplotlib import colors
    from collections import Sequence
    from numbers import Real

    if c is None:
        return True

    if isinstance(c, Sequence) and not isinstance(c, str):
        return 3 <= len(c) <= 4 and all(0 <= i <= 1 for i in c)

    # greyscale
    if isinstance(c, Real):
        return 0 <= c <= 1

    if isinstance(c, str):
        # greyscale
        if c.isdecimal():
            return 0 <= float(c) <= 1
        # color cycle value: C0
        if c.startswith("C"):
            return len(c) > 1 and c[1:].isdigit()
        # html hex
        if c.startswith("#"):
            return len(c) == 7

        return c.lower() in colors.BASE_COLORS or c.lower() in colors.CSS4_COLORS
    return False


def check_valid_color(c: ValidColor) -> bool:
    """
    Checks if `c` is a valid color argument for matplotlib. Raises
    `ValueError` if `c` is not a valid color.

    Parameters
    ----------
    c : Union[str, Real, Sequence[Real], NoneType]

    Returns
    -------
    bool

    Raises
    ------
    ValueError"""
    if not _is_valid_color(c):
        raise ValueError("{} is not a valid matplotlib color".format(repr(c)))
    else:
        return True
