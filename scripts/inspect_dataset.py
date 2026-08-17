from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hovr_sg.utils.ontology import Ontology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    ontology = Ontology(args.ontology)
    objects, groups, relations, scopes = Counter(), Counter(), Counter(), Counter()
    images = 0
    for path in args.jsonl:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line); images += 1
            scopes[rec.get("annotation_scope", "unknown")] += 1
            for obj in rec.get("objects", []):
                objects[obj.get("label", "unknown")] += 1
                for group in obj.get("group_labels", []):
                    groups[group] += 1
            for rel in rec.get("relations", []):
                relations[rel.get("predicate", "unknown")] += 1
    report = {
        "ontology": ontology.summary(), "images": images,
        "objects": objects, "groups": groups, "relations": relations, "annotation_scopes": scopes,
        "unseen_object_labels": sorted(set(objects) - set(ontology.leaf_names())),
        "unseen_predicates": sorted(set(relations) - set(ontology.predicate_names())),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=dict)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
