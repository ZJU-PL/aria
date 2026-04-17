from aria.bool.dissolve.dissolve import (
    DilemmaEngine,
    DilemmaQuery,
    DilemmaTriple,
    DissolveConfig,
    Scheduler,
    UBTree,
)


def test_dissolve_demo_import_surface() -> None:
    assert DilemmaEngine is not None
    assert DilemmaQuery is not None
    assert DilemmaTriple is not None
    assert DissolveConfig is not None
    assert Scheduler is not None
    assert UBTree is not None
