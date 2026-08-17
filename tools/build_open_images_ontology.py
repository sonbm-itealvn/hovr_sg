from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from hovr_sg.utils.open_images_ontology import generate_open_images_ontology


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ontology from Open Images V6 CSV files")
    parser.add_argument("--boxes-csv", required=True)
    parser.add_argument("--relations-csv", default=None)
    parser.add_argument("--class-descriptions", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--ontology-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--min-object-count", type=int, default=None)
    parser.add_argument("--min-predicate-count", type=int, default=None)
    parser.add_argument("--max-object-leaves", type=int, default=None)
    parser.add_argument("--max-predicates", type=int, default=None)
    parser.add_argument("--max-aliases-per-label", type=int, default=None)
    args = parser.parse_args()
    policy = {}
    if args.policy:
        policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {}
    overrides = {
        "min_object_count": args.min_object_count,
        "min_predicate_count": args.min_predicate_count,
        "max_object_leaves": args.max_object_leaves,
        "max_predicates": args.max_predicates,
        "max_aliases_per_label": args.max_aliases_per_label,
    }
    policy.update({key: value for key, value in overrides.items() if value is not None})
    ontology, report = generate_open_images_ontology(
        args.boxes_csv, args.relations_csv, args.class_descriptions, policy
    )
    ontology_path, report_path = Path(args.ontology_output), Path(args.report_output)
    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ontology_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ontology_output": str(ontology_path), "report_output": str(report_path),
        "version": ontology["version"],
        "object_leaves": len(ontology["object_leaves"]),
        "predicates": len(ontology["predicates"]),
        "object_coverage": report["object"]["coverage"],
        "predicate_coverage": report["predicate"]["coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
