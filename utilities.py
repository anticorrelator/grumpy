class Bunch:

    """
    Bunch is a utility class used to group named attributes into a single
    object. Bunch can be called with keyword arguments to set these
    attributes or they can be assigned after instantiation.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    pass


def iterfy(x):
    """
    Checks to see if the input is iterable. If not, the input is nested in a
    list so that it is. Useful for sanitizing inputs to functions that get
    looped over.
    """
    if isinstance(x, str):
        x = [x]
    try:
        iter(x)
    except TypeError:
        x = [x]
    return x
