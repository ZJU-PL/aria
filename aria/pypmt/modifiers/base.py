
class Modifier:
    """
    Modifier class.
    """
    def __init__(self, name):
        self.name = name
        self.mutexes = []

    def encode(self, encoder, actions) -> set:
        raise NotImplementedError('Modifier.encode() not implemented.')
