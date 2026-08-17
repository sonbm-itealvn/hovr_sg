import csv
import json

from hovr_sg.utils.gqa_ontology import generate_gqa_ontology
from hovr_sg.utils.open_images_ontology import generate_open_images_ontology


def test_open_images_ontology_from_csv(tmp_path):
    boxes = tmp_path / "boxes.csv"
    with boxes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "ClassName"])
        writer.writeheader()
        writer.writerows([
            {"ImageID": "img1", "ClassName": "/m/bike"},
            {"ImageID": "img2", "ClassName": "/m/bike"},
            {"ImageID": "img1", "ClassName": "/m/car"},
        ])
    descriptions = tmp_path / "class-descriptions.csv"
    with descriptions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["LabelName", "DisplayName"])
        writer.writeheader()
        writer.writerows([
            {"LabelName": "/m/bike", "DisplayName": "Bicycle"},
            {"LabelName": "/m/car", "DisplayName": "Car"},
        ])
    relations = tmp_path / "relations.csv"
    with relations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "RelationshipLabel"])
        writer.writeheader()
        writer.writerow({"ImageID": "img1", "RelationshipLabel": "next to"})
    ontology, report = generate_open_images_ontology(
        boxes, relations, descriptions,
        {"min_object_count": 1, "min_predicate_count": 1},
    )
    assert {item["id"] for item in ontology["object_leaves"]} == {"bicycle", "car"}
    assert {item["id"] for item in ontology["predicates"]} == {"next_to"}
    assert report["object"]["coverage"] == 1.0
    assert report["predicate"]["coverage"] == 1.0


def test_gqa_ontology_from_scene_graphs(tmp_path):
    scene_graphs = tmp_path / "sceneGraphs.json"
    scene_graphs.write_text(json.dumps({
        "img1": {"objects": {
            "1": {"name": "bikes", "relations": [{"name": "next to", "object": "2"}]},
            "2": {"name": "car", "relations": []},
        }},
        "img2": {"objects": [
            {"name": "bike", "relationships": [{"predicate": "behind", "target": "0"}]},
        ]},
    }), encoding="utf-8")
    ontology, report = generate_gqa_ontology(
        scene_graphs,
        {
            "min_object_count": 1,
            "min_predicate_count": 1,
            "aliases": {"bike": "bicycle"},
        },
    )
    assert {item["id"] for item in ontology["object_leaves"]} == {"bicycle", "car"}
    assert {item["id"] for item in ontology["predicates"]} == {"next_to", "behind"}
    assert report["object"]["raw_instances"] == 3
    assert report["predicate"]["raw_instances"] == 2
