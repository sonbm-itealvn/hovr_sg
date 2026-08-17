from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an HOVR-SG official checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ontology", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {"checkpoint_type", "format_version", "encoder", "model", "prototypes", "config", "resolved"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    if checkpoint["checkpoint_type"] != "hovr_sg_official_checkpoint":
        raise ValueError("Checkpoint does not have the official HOVR-SG checkpoint type")
    prototypes = checkpoint["prototypes"]
    dims = {name: tuple(value.shape) for name, value in prototypes.items()}
    if not all(name in prototypes for name in ("leaf", "groups", "relations")):
        raise ValueError("Checkpoint must contain leaf, groups and relations prototypes")
    text_dims = {int(prototypes[name].shape[-1]) for name in ("leaf", "groups", "relations")}
    if len(text_dims) != 1 or next(iter(text_dims)) != int(checkpoint["resolved"]["text_dim"]):
        raise ValueError(f"Prototype dimensions do not match checkpoint resolved text_dim: {dims}")
    ontology_hash = sha256_file(args.ontology)
    if checkpoint.get("ontology_sha256") != ontology_hash:
        raise ValueError("Ontology hash differs from the ontology used to create this checkpoint")
    report = {
        "valid": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_type": checkpoint["checkpoint_type"],
        "format_version": checkpoint["format_version"],
        "epoch": checkpoint.get("epoch"),
        "stage": checkpoint.get("stage"),
        "resolved": checkpoint["resolved"],
        "val_metrics": checkpoint.get("val_metrics", {}),
        "git_commit": checkpoint.get("git_commit", "unknown"),
        "ontology_sha256": ontology_hash,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
