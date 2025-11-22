"""
Concise Role-Based Access Control (RBAC) Demo with Z3 Verification
"""
from z3 import *

class RBACModel:
    def __init__(self):
        self.users, self.roles, self.permissions, self.resources = set(), set(), set(), set()
        self.user_role_assignments, self.role_permission_assignments = {}, {}
        self.permission_resource_assignments, self.role_hierarchy = {}, {}
        self.solver = Solver()
        self.z3_users, self.z3_roles, self.z3_permissions, self.z3_resources = {}, {}, {}, {}
        self.z3_user_role, self.z3_role_perm, self.z3_perm_resource, self.z3_role_hierarchy = {}, {}, {}, {}

    def add_user(self, user):
        self.users.add(user)
        self.user_role_assignments[user] = set()
    def add_role(self, role):
        self.roles.add(role)
        self.role_permission_assignments[role] = set()
        self.role_hierarchy[role] = set()
    def add_permission(self, perm):
        self.permissions.add(perm)
        self.permission_resource_assignments[perm] = set()
    def add_resource(self, res):
        self.resources.add(res)
    def assign_user_to_role(self, user, role):
        if user not in self.users or role not in self.roles:
            raise ValueError("User or role does not exist")
        self.user_role_assignments[user].add(role)
    def assign_permission_to_role(self, perm, role):
        if perm not in self.permissions or role not in self.roles:
            raise ValueError("Permission or role does not exist")
        self.role_permission_assignments[role].add(perm)
    def assign_permission_to_resource(self, perm, res):
        if perm not in self.permissions or res not in self.resources:
            raise ValueError("Permission or resource does not exist")
        self.permission_resource_assignments[perm].add(res)
    def add_role_hierarchy(self, senior, junior):
        if senior not in self.roles or junior not in self.roles:
            raise ValueError("Role does not exist")
        self.role_hierarchy[senior].add(junior)
    def get_all_permissions_for_role(self, role):
        if role not in self.roles:
            raise ValueError("Role does not exist")
        perms = set(self.role_permission_assignments[role])
        for jr in self.role_hierarchy[role]:
            perms.update(self.get_all_permissions_for_role(jr))
        return perms
    def get_all_roles_for_user(self, user):
        if user not in self.users:
            raise ValueError("User does not exist")
        roles = set(self.user_role_assignments[user])
        all_roles = set(roles)
        for r in roles:
            for sr in self.roles:
                if r in self.get_all_junior_roles(sr):
                    all_roles.add(sr)
        return all_roles
    def get_all_junior_roles(self, role):
        if role not in self.roles:
            raise ValueError("Role does not exist")
        juniors = set(self.role_hierarchy[role])
        for jr in list(juniors):
            juniors.update(self.get_all_junior_roles(jr))
        return juniors
    def check_user_permission(self, user, perm):
        if user not in self.users or perm not in self.permissions:
            raise ValueError("User or permission does not exist")
        return any(perm in self.get_all_permissions_for_role(r) for r in self.get_all_roles_for_user(user))
    def check_user_access_to_resource(self, user, res):
        if user not in self.users or res not in self.resources:
            raise ValueError("User or resource does not exist")
        for r in self.get_all_roles_for_user(user):
            for p in self.get_all_permissions_for_role(r):
                if res in self.permission_resource_assignments[p]:
                    return True
        return False
    # Z3 formal verification
    def initialize_z3_variables(self):
        for user in self.users:
            self.z3_users[user] = Bool(f"user_{user}")
            self.z3_user_role[user] = {role: Bool(f"user_{user}_has_role_{role}") for role in self.roles}
        for role in self.roles:
            self.z3_roles[role] = Bool(f"role_{role}")
            self.z3_role_perm[role] = {perm: Bool(f"role_{role}_has_permission_{perm}") for perm in self.permissions}
            self.z3_role_hierarchy[role] = {jr: Bool(f"role_{role}_inherits_{jr}") for jr in self.roles}
        for perm in self.permissions:
            self.z3_permissions[perm] = Bool(f"permission_{perm}")
            self.z3_perm_resource[perm] = {res: Bool(f"permission_{perm}_on_resource_{res}") for res in self.resources}
        for res in self.resources:
            self.z3_resources[res] = Bool(f"resource_{res}")
    def encode_rbac_model(self):
        self.initialize_z3_variables()
        self.solver = Solver()
        # Assignments
        for user in self.users:
            for role in self.roles:
                self.solver.add(self.z3_user_role[user][role] if role in self.user_role_assignments[user] else Not(self.z3_user_role[user][role]))
        for role in self.roles:
            for perm in self.permissions:
                self.solver.add(self.z3_role_perm[role][perm] if perm in self.role_permission_assignments[role] else Not(self.z3_role_perm[role][perm]))
        for perm in self.permissions:
            for res in self.resources:
                self.solver.add(self.z3_perm_resource[perm][res] if res in self.permission_resource_assignments[perm] else Not(self.z3_perm_resource[perm][res]))
        for sr in self.roles:
            for jr in self.roles:
                self.solver.add(self.z3_role_hierarchy[sr][jr] if jr in self.role_hierarchy[sr] else Not(self.z3_role_hierarchy[sr][jr]))
        # Hierarchy transitivity
        for r1 in self.roles:
            for r2 in self.roles:
                for r3 in self.roles:
                    self.solver.add(Implies(And(self.z3_role_hierarchy[r1][r2], self.z3_role_hierarchy[r2][r3]), self.z3_role_hierarchy[r1][r3]))
        # Permission inheritance (no-op, for extensibility)
        # for sr in self.roles:
        #     for jr in self.roles:
        #         for perm in self.permissions:
        #             self.solver.add(Implies(And(self.z3_role_hierarchy[sr][jr], self.z3_role_perm[jr][perm]), True))
    def verify_property(self, prop):
        if not self.solver.assertions():
            self.encode_rbac_model()
        self.solver.push()
        self.solver.add(Not(prop))
        result = self.solver.check()
        if result == sat:
            model = self.solver.model()
            ce = self._extract_counterexample(model)
            self.solver.pop()
            return False, ce
        self.solver.pop()
        return True, None
    def _extract_counterexample(self, model):
        ce = {"user_roles": {}, "role_permissions": {}, "permission_resources": {}, "role_hierarchy": {}}
        for user in self.users:
            ce["user_roles"][user] = [role for role in self.roles if is_true(model.evaluate(self.z3_user_role[user][role]))]
        for role in self.roles:
            ce["role_permissions"][role] = [perm for perm in self.permissions if is_true(model.evaluate(self.z3_role_perm[role][perm]))]
        for perm in self.permissions:
            ce["permission_resources"][perm] = [res for res in self.resources if is_true(model.evaluate(self.z3_perm_resource[perm][res]))]
        for sr in self.roles:
            ce["role_hierarchy"][sr] = [jr for jr in self.roles if is_true(model.evaluate(self.z3_role_hierarchy[sr][jr]))]
        return ce
    def verify_separation_of_duty(self, r1, r2):
        if r1 not in self.roles or r2 not in self.roles:
            raise ValueError("Roles must exist")
        prop = And([Not(And(self.z3_user_role[u][r1], self.z3_user_role[u][r2])) for u in self.users])
        return self.verify_property(prop)
    def verify_least_privilege(self, role, req_perms):
        if role not in self.roles or any(p not in self.permissions for p in req_perms):
            raise ValueError("Role or permission does not exist")
        has_req = And([self.z3_role_perm[role][p] for p in req_perms])
        not_others = And([Not(self.z3_role_perm[role][p]) for p in self.permissions if p not in req_perms])
        return self.verify_property(And(has_req, not_others))
    def verify_role_containment(self, r1, r2):
        if r1 not in self.roles or r2 not in self.roles:
            raise ValueError("Roles must exist")
        prop = And([Implies(self.z3_role_perm[r2][p], self.z3_role_perm[r1][p]) for p in self.permissions])
        return self.verify_property(prop)
    def verify_no_access(self, user, res):
        if user not in self.users or res not in self.resources:
            raise ValueError("User or resource does not exist")
        prop = Not(Or([And(self.z3_user_role[user][role], self.z3_role_perm[role][perm], self.z3_perm_resource[perm][res]) for role in self.roles for perm in self.permissions]))
        return self.verify_property(prop)
    def verify_resource_isolation(self, r1, r2):
        if r1 not in self.resources or r2 not in self.resources:
            raise ValueError("Resources must exist")
        prop = And([
            Not(And(
                Or([And(self.z3_user_role[u][role1], self.z3_role_perm[role1][p1], self.z3_perm_resource[p1][r1]) for role1 in self.roles for p1 in self.permissions]),
                Or([And(self.z3_user_role[u][role2], self.z3_role_perm[role2][p2], self.z3_perm_resource[p2][r2]) for role2 in self.roles for p2 in self.permissions])
            )) for u in self.users
        ])
        return self.verify_property(prop)

