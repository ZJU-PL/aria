
class Encoder:
    """!
    The main role of an Encoder is to receive a Unified Planning task and
    produce a SMT encoding.  This class is an interface for all encodings to
    implement.
    
    Typical encodings iteratively add layers (or timesteps) until the solver
    proves its satisfiability.  That is, finds a plan of a given number of
    steps. Therefore, encoders generally keep a record of which timesteps have
    been encoded.  Note that requirements of some encodings will leave some
    methods unimplemented if they do not need them.
    """

    def __iter__(self):
        """!
        The iterator goes through the raw actions, allowing a clean interface
        when for example extracting a plan from a model.
        """
        raise NotImplementedError
    
    def __len__(self):
        """!
        @returns the number of timesteps that have been encoded
        """
        raise NotImplementedError

    def create_variables(self, t):
        """!
        Creates the Z3 variables needed for a given timestep

        @param t: The timestep for which the variables need to be created
        """
        raise NotImplementedError

    def encode(self, t):
        """!
        Encodes and returns the formula for a single transition step (from t to t+1).

        @param t: the timestep to consider when encoding the single transition step
        """
        raise NotImplementedError

    def encode_initial_state(self):
        """!
        Encodes the initial state.
        @returns the encoded formula/s
        """
        raise NotImplementedError

    def encode_goal_state(self, t):
        """!
        Encodes the goal state.
        @returns the encoded formula/s
        """
        raise NotImplementedError

    def encode_actions(self, t):
        """!
        Encodes the transition function.
        @returns the encoded formula/s
        """
        raise NotImplementedError

    def encode_frame(self, t):
        """!
        Encodes the frame axioms.
        @returns the encoded formula/s
        """
        raise NotImplementedError

    def encode_execution_semantics(self):
        """!
        Encodes the possible needed mutexes between actions
        @returns the encoded formula/s
        """
        raise NotImplementedError
