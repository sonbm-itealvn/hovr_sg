import json

from hovr_sg.utils.vg_ontology import generate_vg_ontology


def test_generate_vg_ontology_from_vg_records(tmp_path):
    objects = [
        {"image_id": 1, "objects": [
            {"object_id": 1, "names": ["Bikes", "bike"], "x": 0, "y": 0, "w": 10, "h": 10},
            {"object_id": 2, "names": ["cars"], "x": 10, "y": 0, "w": 10, "h": 10},
        ]},
        {"image_id": 2, "objects": [
            {"object_id": 3, "names": ["bike"], "x": 0, "y": 0, "w": 10, "h": 10},
        ]},
    ]
    relationships = [
        {"image_id": 1, "relationships": [
            {"predicate": "next to", "subject": {"object_id": 1}, "object": {"object_id": 2}},
            {"predicate": "behind", "subject": {"object_id": 2}, "object": {"object_id": 1}},
        ]},
    ]
    objects_path = tmp_path / "objects.json"
    relationships_path = tmp_path / "relationships.json"
    objects_path.write_text(json.dumps(objects), encoding="utf-8")
    relationships_path.write_text(json.dumps(relationships), encoding="utf-8")
    ontology, report = generate_vg_ontology(
        objects_path, relationships_path,
        {
            "version": "ontology_test",
            "min_object_count": 1,
            "min_predicate_count": 1,
            "aliases": {"bike": "bicycle"},
            "symmetric_predicates": ["next to"],
        },
    )
    object_ids = {item["id"] for item in ontology["object_leaves"]}
    predicate_ids = {item["id"] for item in ontology["predicates"]}
    assert object_ids == {"bicycle", "car"}
    assert predicate_ids == {"next_to", "behind"}
    bicycle = next(item for item in ontology["object_leaves"] if item["id"] == "bicycle")
    assert bicycle["parents"] == ["vehicle"]
    assert "car" in bicycle["siblings"]
    next_to = next(item for item in ontology["predicates"] if item["id"] == "next_to")
    assert next_to["symmetric"] is True
    assert report["object"]["coverage"] == 1.0
    assert report["predicate"]["coverage"] == 1.0
