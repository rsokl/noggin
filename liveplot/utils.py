def is_valid_color(c):
    """ Checks if `c` is a valid color argument for matplotlib.

        Parameters
        ----------
        c : Union[str, Real, Sequence[Real], NoneType]

        Returns
        -------
        bool"""
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


def check_valid_color(c):
    if not is_valid_color(c):
        raise TypeError("{} is not a valid matplotlib color".format(repr(c)))
    else:
        return True
