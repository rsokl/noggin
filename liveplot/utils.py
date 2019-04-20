from liveplot.typing import ValidColor


__all__ = ["check_valid_color"]


def check_valid_color(c: ValidColor) -> bool:
    """
    Checks if `c` is a valid color argument for matplotlib or `None`.
    Raises `ValueError` if `c` is not a valid color.

    Parameters
    ----------
    c : Union[str, Real, Sequence[Real], NoneType]

    Returns
    -------
    bool

    Raises
    ------
    ValueError"""
    from matplotlib.colors import is_color_like

    if c is not None and not is_color_like(c):
        raise ValueError("{} is not a valid matplotlib color".format(repr(c)))
    else:
        return True
