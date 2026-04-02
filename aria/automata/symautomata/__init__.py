import importlib

__all__ = [
    "Predicate",
    "SetPredicate",
    "SFA",
    "SFAArc",
    "SFAState",
    "Z3Predicate",
]


def __getattr__(name):
    if name in __all__:
        sfa_module = importlib.import_module(".sfa", __name__)
        return getattr(sfa_module, name)
    raise AttributeError(name)
