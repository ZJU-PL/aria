from collections import defaultdict

import z3


class ForallPropagator(z3.UserPropagateBase):
    def __init__(self, s, ctx=None, e=None):
        z3.UserPropagateBase.__init__(self, s, ctx)
        self.name = "forall-lazy"
        self.add_fixed(lambda x, v: self._fixed(x, v))
        self.encoder = e
        self.graph = self.encoder.modifier.graph
        self.current = [set()]  # Use a set instead of a NetworkX graph
        self.trail = []  # Trail to record changes
        self.levels = []
        self.consistent = True
        self.propagated = defaultdict(set) # Tracks propagated clauses to prevent repropagation
        self.nots = defaultdict(dict) # Stores negated actions for Not-Caching

    def push(self):
        self.levels.append(len(self.trail))

    def pop(self, n):
        for _ in range(n):
            if self.levels:
                # Find the start of the current decision level
                level_start = self.levels.pop()
                # Undo all changes recorded after this level
                while len(self.trail) > level_start:
                    step, action = self.trail.pop()
                    self.current[step].remove(action)
        self.consistent = True

    def _fixed(self, action, value):
        if value and self.consistent:
            # Parse action name and step
            actions = str(action).split('_')
            step = int(actions[-1])
            action_name = '_'.join(actions[:-1])
            if step >= len(self.current):
                while step >= len(self.current):
                    self.current.append(set())
                self.current[step].add(action_name)
                self.trail.append((step, action_name))
                # There cannot be any interference: no other actions in step are True
                return
            literals = set()
            self.trail.append((step, action_name))
            self.current[step].add(action_name)
            for dest in self.current[step] & set(self.graph.neighbors(action_name)):
                literals.add(self.encoder.get_action_var(dest, step))
                self.consistent = False
            for dest in set(self.graph.neighbors(action_name)) - self.current[step] - self.propagated[step]:
                if (dest, action_name) not in self.propagated[step]:
                    if dest not in self.nots[step]:
                        self.nots[step][dest] = z3.Not(self.encoder.get_action_var(dest, step))
                    self.propagate(
                        e=self.nots[step][dest],
                        ids=[action],
                        eqs=[]
                    )
                    self.propagated[step].add((dest, action_name))
                    self.propagated[step].add((action_name, dest))

            # Checking and adding in nodes using set intersection
            for source in self.current[step] & set(self.graph.predecessors(action_name)):
                literals.add(self.encoder.get_action_var(source, step))
                self.consistent = False
            for source in set(self.graph.neighbors(action_name)) - self.current[step]:
                if (source, action_name) not in self.propagated[step]:
                    if source not in self.nots[step]:
                        self.nots[step][source] = z3.Not(self.encoder.get_action_var(source, step))
                    self.propagate(
                        e=self.nots[step][source],
                        ids=[action],
                        eqs=[]
                    )
                    self.propagated[step].add((source, action_name))
                    self.propagated[step].add((action_name, source))
            # Check if anything has caused interference
            if literals:
                literals.add(action)  # New action itself is only added once
                self.conflict(deps=list(literals), eqs=[])
