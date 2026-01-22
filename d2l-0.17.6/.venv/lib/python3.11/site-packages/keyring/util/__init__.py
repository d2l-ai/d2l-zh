import contextlib


def suppress_exceptions(callables, exceptions=Exception):
    """
    yield the results of calling each element of callables, suppressing
    any indicated exceptions.
    """
    for callable in callables:
        with contextlib.suppress(exceptions):
            yield callable()
