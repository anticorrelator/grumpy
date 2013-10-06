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
