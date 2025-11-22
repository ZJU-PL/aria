"""
ABAC Demo with Formal Verification (Z3)
"""

from z3 import *
import itertools

class AttributeBasedAccessControl:
    def __init__(self):
        self.subjects, self.resources, self.actions, self.environments = set(), set(), set(), set()
        self.subject_attributes, self.resource_attributes = {}, {}
        self.action_attributes, self.environment_attributes = {}, {}
        self.policies = []
    def add_subject(self, sid, attrs): self.subjects.add(sid); self.subject_attributes[sid] = attrs
    def add_resource(self, rid, attrs): self.resources.add(rid); self.resource_attributes[rid] = attrs
    def add_action(self, aid, attrs): self.actions.add(aid); self.action_attributes[aid] = attrs
    def add_environment(self, eid, attrs): self.environments.add(eid); self.environment_attributes[eid] = attrs
    def add_policy(self, policy): self.policies.append(policy)
    def check_access(self, sid, rid, aid, eid):
        if sid not in self.subjects or rid not in self.resources or aid not in self.actions or eid not in self.environments:
            return False
        s, r, a, e = self.subject_attributes[sid], self.resource_attributes[rid], self.action_attributes[aid], self.environment_attributes[eid]
        return all(policy(s, r, a, e) for policy in self.policies)

class ABACVerifier:
    def __init__(self, abac):
        self.abac = abac
        self.solver = Solver()
        self.SubjectSort, self.ResourceSort = DeclareSort('Subject'), DeclareSort('Resource')
        self.ActionSort, self.EnvSort = DeclareSort('Action'), DeclareSort('Environment')
        self.subject_constants = {sid: Const(f"subject_{sid}", self.SubjectSort) for sid in abac.subjects}
        self.resource_constants = {rid: Const(f"resource_{rid}", self.ResourceSort) for rid in abac.resources}
        self.action_constants = {aid: Const(f"action_{aid}", self.ActionSort) for aid in abac.actions}
        self.env_constants = {eid: Const(f"env_{eid}", self.EnvSort) for eid in abac.environments}
        self.subject_attr_funcs = {attr: Function(f"subject_{attr}", self.SubjectSort, StringSort()) for attrs in abac.subject_attributes.values() for attr in attrs}
        self.resource_attr_funcs = {attr: Function(f"resource_{attr}", self.ResourceSort, StringSort()) for attrs in abac.resource_attributes.values() for attr in attrs}
        self.action_attr_funcs = {attr: Function(f"action_{attr}", self.ActionSort, StringSort()) for attrs in abac.action_attributes.values() for attr in attrs}
        self.env_attr_funcs = {attr: Function(f"env_{attr}", self.EnvSort, StringSort()) for attrs in abac.environment_attributes.values() for attr in attrs}
    def define_attribute_constraints(self):
        c = []
        for sid, attrs in self.abac.subject_attributes.items():
            s = self.subject_constants[sid]
            for k, v in attrs.items():
                if k in self.subject_attr_funcs: c.append(self.subject_attr_funcs[k](s) == StringVal(v))
        for rid, attrs in self.abac.resource_attributes.items():
            r = self.resource_constants[rid]
            for k, v in attrs.items():
                if k in self.resource_attr_funcs: c.append(self.resource_attr_funcs[k](r) == StringVal(v))
        for aid, attrs in self.abac.action_attributes.items():
            a = self.action_constants[aid]
            for k, v in attrs.items():
                if k in self.action_attr_funcs: c.append(self.action_attr_funcs[k](a) == StringVal(v))
        for eid, attrs in self.abac.environment_attributes.items():
            e = self.env_constants[eid]
            for k, v in attrs.items():
                if k in self.env_attr_funcs: c.append(self.env_attr_funcs[k](e) == StringVal(v))
        return c
    def access_function(self):
        return Function('access', self.SubjectSort, self.ResourceSort, self.ActionSort, self.EnvSort, BoolSort())
    def policy_to_z3(self, policy):
        access = self.access_function()
        c = []
        for s_id, r_id, a_id, e_id in itertools.product(self.abac.subjects, self.abac.resources, self.abac.actions, self.abac.environments):
            s, r, a, e = self.subject_constants[s_id], self.resource_constants[r_id], self.action_constants[a_id], self.env_constants[e_id]
            s_attrs, r_attrs, a_attrs, e_attrs = self.abac.subject_attributes[s_id], self.abac.resource_attributes[r_id], self.abac.action_attributes[a_id], self.abac.environment_attributes[e_id]
            c.append(access(s, r, a, e) == policy(s_attrs, r_attrs, a_attrs, e_attrs))
        return c
    def verify_property(self, property_formula):
        self.solver.push()
        for constraint in self.define_attribute_constraints(): self.solver.add(constraint)
        for policy in self.abac.policies:
            for constraint in self.policy_to_z3(policy): self.solver.add(constraint)
        self.solver.add(Not(property_formula))
        result = self.solver.check()
        model = self.solver.model() if result == sat else None
        self.solver.pop()
        return result, model

# Example usage
if __name__ == "__main__":
    abac = AttributeBasedAccessControl()
    abac.add_subject("alice", {"role": "manager", "department": "finance", "clearance": "high"})
    abac.add_subject("bob", {"role": "employee", "department": "engineering", "clearance": "medium"})
    abac.add_subject("charlie", {"role": "contractor", "department": "engineering", "clearance": "low"})
    abac.add_resource("financial_report", {"type": "document", "sensitivity": "high", "department": "finance"})
    abac.add_resource("code_repo", {"type": "repository", "sensitivity": "medium", "department": "engineering"})
    abac.add_resource("company_wiki", {"type": "knowledge_base", "sensitivity": "low", "department": "all"})
    abac.add_action("read", {"impact": "low"})
    abac.add_action("write", {"impact": "high"})
    abac.add_action("execute", {"impact": "medium"})
    abac.add_environment("office_hours", {"time": "work_hours", "location": "office", "connection": "secure"})
    abac.add_environment("remote", {"time": "any", "location": "remote", "connection": "vpn"})
    abac.add_policy(lambda s, r, a, e: r["department"] == s["department"] or r["department"] == "all")
    abac.add_policy(lambda s, r, a, e: {"low": 1, "medium": 2, "high": 3}[s["clearance"]] >= {"low": 1, "medium": 2, "high": 3}[r["sensitivity"]])
    abac.add_policy(lambda s, r, a, e: s["clearance"] == "high" if a["impact"] == "high" else True)
    abac.add_policy(lambda s, r, a, e: e["connection"] == "vpn" if e["location"] == "remote" else True)
    print("Testing access control decisions:\n" + "-" * 40)
    print(f"Alice writing financial report during office hours: {abac.check_access('alice', 'financial_report', 'write', 'office_hours')}")
    print(f"Bob reading code repo remotely: {abac.check_access('bob', 'code_repo', 'read', 'remote')}")
    print(f"Charlie writing to code repo during office hours: {abac.check_access('charlie', 'code_repo', 'write', 'office_hours')}")
    print(f"Bob reading financial report during office hours: {abac.check_access('bob', 'financial_report', 'read', 'office_hours')}")
    print("\nFormal Verification Example:\n" + "-" * 40)
    verifier = ABACVerifier(abac)
    access = verifier.access_function()
    subject = Const("subject_var", verifier.SubjectSort)
    resource = Const("resource_var", verifier.ResourceSort)
    action = Const("action_var", verifier.ActionSort)
    env = Const("env_var", verifier.EnvSort)
    property_formula = ForAll(
        [subject, resource, action, env],
        Implies(
            And(
                verifier.subject_attr_funcs["clearance"](subject) == StringVal("low"),
                verifier.resource_attr_funcs["sensitivity"](resource) == StringVal("high")
            ),
            Not(access(subject, resource, action, env))
        )
    )
    result, model = verifier.verify_property(property_formula)
    print(f"Property verification result: {result}")
    print("Property violated. Counterexample found:" if result == sat else "Property holds. No violations possible.")
    if result == sat: print(model)


