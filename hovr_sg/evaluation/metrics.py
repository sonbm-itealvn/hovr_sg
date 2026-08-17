"""Self-contained COCO-style object and scene-graph evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-8)


def _average_precision(scores: List[float], matches: List[bool], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positives = 0.0
    false_positives = 0.0
    precisions = []
    recalls = []
    for index in order:
        if matches[index]:
            true_positives += 1.0
        else:
            false_positives += 1.0
        precisions.append(true_positives / max(true_positives + false_positives, 1e-8))
        recalls.append(true_positives / total_gt)
    if not recalls:
        return 0.0
    recall_levels = [index / 101.0 for index in range(101)]
    return sum(
        max((precision for precision, recall in zip(precisions, recalls) if recall >= level), default=0.0)
        for level in recall_levels
    ) / len(recall_levels)


def coco_style_object_metrics(records: Iterable[dict], predictions: Iterable[dict], ontology) -> Dict[str, float]:
    """Compute class-aware AP at IoU .50 and .75 plus COCO-style AP@[.50:.95]."""
    records = list(records)
    predictions = list(predictions)
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    per_threshold = []
    for threshold in thresholds:
        scores = []
        matched_flags = []
        total_gt = 0
        for record, prediction in zip(records, predictions):
            gt_boxes = record["boxes"].tolist()
            gt_labels = record["leaf_indices"].tolist()
            total_gt += len(gt_boxes)
            used = set()
            for candidate in prediction.get("objects", []):
                scores.append(float(candidate.get("score", 0.0)))
                label = candidate.get("label")
                label_index = ontology.leaf_index(label) if label in ontology.leaf_to_idx else -1
                best = -1.0
                best_index = None
                for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_index in used or int(gt_label) != label_index:
                        continue
                    overlap = box_iou(candidate["box"], gt_box)
                    if overlap > best:
                        best, best_index = overlap, gt_index
                if best_index is not None and best >= threshold:
                    used.add(best_index)
                    matched_flags.append(True)
                else:
                    matched_flags.append(False)
        per_threshold.append(_average_precision(scores, matched_flags, total_gt))
    return {
        "object_AP50": per_threshold[0],
        "object_AP75": per_threshold[5],
        "object_mAP50_95": sum(per_threshold) / len(per_threshold),
    }


def _ground_truth_triplets(record: dict, ontology) -> set[tuple[int, int, int]]:
    object_index = {int(object_id): index for index, object_id in enumerate(record["object_ids"])}
    triplets = set()
    for relation in record["relations"]:
        subject = object_index.get(int(relation["subject_id"]))
        object_ = object_index.get(int(relation["object_id"]))
        if subject is not None and object_ is not None:
            triplets.add((subject, int(relation["predicate_index"]), object_))
    return triplets


def scene_graph_metrics(records: Iterable[dict], predictions: Iterable[dict], ontology, ks=(20, 50, 100)) -> Dict[str, float]:
    """Compute relation Recall@K and mean Recall over predicate classes."""
    recalls = {k: [] for k in ks}
    predicate_hits = defaultdict(int)
    predicate_total = defaultdict(int)
    for record, prediction in zip(records, predictions):
        gt_triplets = _ground_truth_triplets(record, ontology)
        object_ids = list(range(len(record["object_ids"])))
        pred_objects = prediction.get("objects", [])
        slot_to_gt = {}
        for candidate in pred_objects:
            slot = int(candidate.get("slot", -1))
            best = (-1.0, None)
            for gt_index, gt_box in enumerate(record["boxes"].tolist()):
                overlap = box_iou(candidate["box"], gt_box)
                if overlap > best[0] and candidate.get("label") == ontology.leaf_names()[int(record["leaf_indices"][gt_index])]:
                    best = (overlap, gt_index)
            if best[1] is not None and best[0] >= 0.5:
                slot_to_gt[slot] = best[1]
        candidate_triplets = []
        for relation in prediction.get("relations", []):
            subject = slot_to_gt.get(int(relation.get("subject_slot", -1)))
            object_ = slot_to_gt.get(int(relation.get("object_slot", -1)))
            predicate = ontology.predicate_index(relation["predicate"]) if relation.get("predicate") in ontology.predicate_to_idx else -1
            if subject is not None and object_ is not None and predicate >= 0:
                candidate_triplets.append((float(relation.get("score", 0.0)), (subject, predicate, object_)))
        candidate_triplets.sort(key=lambda item: item[0], reverse=True)
        for _, predicate, _ in gt_triplets:
            predicate_total[predicate] += 1
        for k in ks:
            hits = {triplet for _, triplet in candidate_triplets[:k]}
            recalls[k].append(len(hits & gt_triplets) / max(len(gt_triplets), 1))
        for predicate in predicate_total:
            gt_for_predicate = {triplet for triplet in gt_triplets if triplet[1] == predicate}
            predicted_for_predicate = {triplet for _, triplet in candidate_triplets[:max(ks)]}
            predicate_hits[predicate] += len(gt_for_predicate & predicted_for_predicate)
    result = {f"relation_Recall@{k}": sum(values) / max(len(values), 1) for k, values in recalls.items()}
    per_predicate = [predicate_hits[index] / total for index, total in predicate_total.items() if total]
    result["relation_mean_Recall"] = sum(per_predicate) / max(len(per_predicate), 1)
    return result


def evaluate_scene_graph(records: Iterable[dict], predictions: Iterable[dict], ontology) -> Dict[str, float]:
    records, predictions = list(records), list(predictions)
    output = coco_style_object_metrics(records, predictions, ontology)
    output.update(scene_graph_metrics(records, predictions, ontology))
    return output
