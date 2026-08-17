from pathlib import Path

from hovr_sg.utils.ontology import Ontology


def test_ontology_mapping():
    root = Path(__file__).parents[1]
    ontology = Ontology(root / "ontology" / "ontology_v1.json")
    assert ontology.canonical_leaf("male") == "man"
    assert ontology.canonical_leaf("coffee cup") == "cup"
    assert ontology.canonical_predicate("grasping") == "holding"
    assert "person" in ontology.parent_groups("girl")
    assert "chair" in ontology.object_leaves["table"].siblings
