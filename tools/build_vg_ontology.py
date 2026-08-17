from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from hovr_sg.utils.vg_ontology import generate_vg_ontology


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Visual Genome ontology and coverage report"
    )
    parser.add_argument("--objects", required=True, help="Visual Genome objects.json")
    parser.add_argument("--relationships", required=True, help="Visual Genome relationships.json")
    parser.add_argument("--policy", default=None, help="Optional JSON/YAML ontology policy")
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
        with Path(args.policy).open("r", encoding="utf-8") as handle:
            policy = yaml.safe_load(handle) or {}
    overrides = {
        "min_object_count": args.min_object_count,
        "min_predicate_count": args.min_predicate_count,
        "max_object_leaves": args.max_object_leaves,
        "max_predicates": args.max_predicates,
        "max_aliases_per_label": args.max_aliases_per_label,
    }
    policy.update({key: value for key, value in overrides.items() if value is not None})
    ontology, report = generate_vg_ontology(args.objects, args.relationships, policy)

    ontology_path = Path(args.ontology_output)
    report_path = Path(args.report_output)
    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ontology_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ontology_output": str(ontology_path),
        "report_output": str(report_path),
        "version": ontology["version"],
        "object_leaves": len(ontology["object_leaves"]),
        "object_groups": len(ontology["object_groups"]),
        "predicates": len(ontology["predicates"]),
        "object_coverage": report["object"]["coverage"],
        "predicate_coverage": report["predicate"]["coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
