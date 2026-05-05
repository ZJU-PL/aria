import z3

class BasePropagator(z3.UserPropagateBase):
    """!
    A base class that receives an encoder and implements a custom propagator.
    It already includes the basic structure for a propagator and the trail
    management.
    """

    def __init__(self, encoder, s=None, ctx=None):
        # We need to call the parent constructor for Z3 to do its magic.
        super().__init__(s, ctx)

        # The encoder holds the semantic mappings between UP and Z3 variables.
        self.encoder = encoder

        # keep track of changes/assignments made during the decision process
        # each entry is an action or change that can be undone.
        self.trail = []

        # Not all elements in the trail are decisions. This list that marks
        # decision levels in the previous trail. Here we store the length
        # of the trail at a specific decision level. Helps to identify where
        # to backtrack to when popping decision levels.
        self.decision_levels = []

        # We add the custom fixed and final methods to the propagator
        self.add_fixed(lambda x, v: self._fixed(x, v))
        self.add_final(lambda: self._final())
        self.encoder = encoder

    # The push and pop methods take care of the trail.
    # For example, this is how it can evolve during the decision process:
    # we start by deciding on x1=T
    # Trail: ['D: x1=T', 'P: x2=F']
    # Decision Levels: [0]

    # Now we decide on x3=T
    # Trail: ['D: x1=T', 'P: x2=F', 'D: x3=T', 'P: x4=F']
    # Decision Levels: [0, 2]
    
    # Now we decide on x5=T
    # Trail: ['D: x1=T', 'P: x2=F', 'D: x3=T', 'P: x4=F', 'D: x5=T', 'P: x6=F']
    # Decision Levels: [0, 2, 4]

    # Now we backtrack to the previous decision level
    # Trail: ['D: x1=T', 'P: x2=F', 'D: x3=T', 'P: x4=F']
    # Decision Levels: [0, 2]
    
    # Now we backtrack to the first decision level
    # Trail: ['D: x1=T', 'P: x2=F']
    # Decision Levels: [0]
    def push(self): # type: ignore
        # mark a new decision level in the trail.
        self.decision_levels.append(len(self.trail))

    def pop(self, num_scopes) -> None: # type: ignore
        # head is the position of the trail that we have to backtrack to
        head = self.decision_levels[len(self.decision_levels) - num_scopes]
        # while we have not reached the head, undo the last action
        while len(self.trail) > head:
            self.trail.pop()
        self.decision_levels = self.decision_levels[:-num_scopes]

    def add_action_variables(self, t):
        # Add the variables of the action to the propagator
        for a in self.encoder.task.actions:
            action = self.encoder.get_action_var(a.name, t)
            self.add(action)

    def _fixed(self, x, v):
        # Implement logic for when a variable is fixed to a value
        pass

    def _final(self):
        # Implement final check logic
        pass