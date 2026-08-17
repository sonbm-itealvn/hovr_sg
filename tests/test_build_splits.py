import json

from tools.build_splits import main, read_records, split_image_ids


def test_split_image_ids_has_no_overlap():
    train_ids, val_ids = split_image_ids(["a", "b", "c", "d"], 0.25, 42)
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {"a", "b", "c", "d"}


def test_build_splits_writes_train_val_and_novel_splits(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    ontology_path = tmp_path / "ontology.json"
    output_dir = tmp_path / "splits"
    records = [
        {"image_id": "a", "objects": [{"label": "person"}], "relations": [{"predicate": "on"}]},
        {"image_id": "b", "objects": [{"label": "person"}], "relations": [{"predicate": "on"}]},
        {"image_id": "c", "objects": [{"label": "dog"}], "relations": [{"predicate": "holding"}]},
        {"image_id": "d", "objects": [{"label": "person"}], "relations": [{"predicate": "on"}]},
    ]
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    ontology_path.write_text(json.dumps({
        "version": "test",
        "object_groups": [{"id": "person", "name": "person", "parents": []}, {"id": "animal", "name": "animal", "parents": []}],
        "object_leaves": [
            {"id": "person", "name": "person", "parents": ["person"], "aliases": [], "siblings": ["dog"]},
            {"id": "dog", "name": "dog", "parents": ["animal"], "aliases": [], "siblings": ["person"]},
        ],
        "predicate_groups": [{"id": "spatial", "name": "spatial", "parents": []}, {"id": "contact", "name": "contact", "parents": []}],
        "predicates": [
            {"id": "on", "name": "on", "parents": ["spatial"], "aliases": []},
            {"id": "holding", "name": "holding", "parents": ["contact"], "aliases": []},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "build_splits", "--input", str(records_path), "--ontology", str(ontology_path),
        "--output-dir", str(output_dir), "--object-novel-ratio", "0.5",
        "--relation-novel-ratio", "0.5", "--val-ratio", "0.5", "--train-val-source", "all", "--seed", "7",
    ])
    main()
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert all((output_dir / f"{name}.jsonl").exists() for name in ("ss", "ns", "sn", "nn"))
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    assert manifest["train_val_source"] == "all"
    assert manifest["train_val_image_overlap"] == 0
    assert manifest["train_images"] + manifest["val_images"] == manifest["train_val_pool_images"]
