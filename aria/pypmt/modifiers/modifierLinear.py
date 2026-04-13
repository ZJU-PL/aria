import z3
from .base import Modifier

class LinearModifier(Modifier):
    """
    Linear modifier, contains method to implement sequential execution semantics.
    """
    def __init__(self):
        super().__init__("LinearModifier")
        
    def encode(self, encoder, actions) -> set:
        return set([z3.PbEq([(var, 1) for var in actions], 1)])

# we could also do the clique, but a priori seems more expensive
# TODO : run some simple tests
#       mutexes = []
#       actions=variables.values()
#       for a1 in actions:
#           for a2 in actions:
#               # Skip same action
#               if a1 != a2:
#                   continue
#               mutexes.append((a1,a2))
#       return mutexes
