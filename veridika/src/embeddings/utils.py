from collections.abc import Iterable


def batch(iterable: Iterable, n=1) -> Iterable:
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]
