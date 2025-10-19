from chapkit import ArtifactHierarchy


def test_label_lookup_returns_configured_value() -> None:
    hierarchy = ArtifactHierarchy(name="ml_flow", level_labels={0: "train", 1: "predict"})
    assert hierarchy.label_for(0) == "train"
    assert hierarchy.label_for(1) == "predict"
    assert hierarchy.label_for(2) == "level_2"


def test_describe_returns_metadata() -> None:
    hierarchy = ArtifactHierarchy(name="ml_flow", level_labels={0: "train"})
    metadata = hierarchy.describe(0)
    assert metadata == {"hierarchy": "ml_flow", "level_depth": 0, "level_label": "train"}


def test_describe_uses_fallback_label() -> None:
    hierarchy = ArtifactHierarchy(name="ml_flow", level_labels={})
    metadata = hierarchy.describe(3)
    assert metadata["level_label"] == "level_3"
