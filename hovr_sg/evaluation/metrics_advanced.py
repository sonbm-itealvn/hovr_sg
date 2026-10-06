"""Advanced Evaluation Metrics for HOVR-SG (CVPR/ECCV/ICCV Standard)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import torch
from .metrics import box_iou, coco_style_object_metrics

def evaluate_object_detection_splits(
    records: Iterable[dict], 
    predictions: Iterable[dict], 
    ontology,
    novel_objects: Set[int]
) -> Dict[str, float]:
    """Bảng 2: Open-Vocabulary Object Detection (Base AP, Novel AP, Harmonic Mean)."""
    # Use existing metrics function but modify it to compute Base and Novel separately
    records = list(records)
    predictions = list(predictions)
    
    threshold = 0.50
    base_scores, base_matched, base_gt = [], [], 0
    novel_scores, novel_matched, novel_gt = [], [], 0
    
    for record, prediction in zip(records, predictions):
        gt_boxes = record["boxes"].tolist()
        gt_labels = record["leaf_indices"].tolist()
        
        for gt_label in gt_labels:
            if gt_label in novel_objects:
                novel_gt += 1
            else:
                base_gt += 1
                
        used = set()
        for candidate in prediction.get("objects", []):
            score = float(candidate.get("score", 0.0))
            label = candidate.get("label")
            label_index = ontology.leaf_index(label) if label in ontology.leaf_to_idx else -1
            
            is_novel = label_index in novel_objects
            
            best = -1.0
            best_index = None
            for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_index in used or int(gt_label) != label_index:
                    continue
                overlap = box_iou(candidate["box"], gt_box)
                if overlap > best:
                    best, best_index = overlap, gt_index
                    
            matched = False
            if best_index is not None and best >= threshold:
                used.add(best_index)
                matched = True
                
            if is_novel:
                novel_scores.append(score)
                novel_matched.append(matched)
            else:
                base_scores.append(score)
                base_matched.append(matched)
                
    def _ap(scores, matches, total_gt):
        if total_gt == 0: return 0.0
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        tp = 0.0
        fp = 0.0
        precisions, recalls = [], []
        for index in order:
            if matches[index]: tp += 1.0
            else: fp += 1.0
            precisions.append(tp / max(tp + fp, 1e-8))
            recalls.append(tp / total_gt)
        if not recalls: return 0.0
        recall_levels = [i / 101.0 for i in range(101)]
        return sum(
            max((p for p, r in zip(precisions, recalls) if r >= level), default=0.0)
            for level in recall_levels
        ) / len(recall_levels)

    base_ap = _ap(base_scores, base_matched, base_gt) * 100
    novel_ap = _ap(novel_scores, novel_matched, novel_gt) * 100
    hm = (2 * base_ap * novel_ap) / max(base_ap + novel_ap, 1e-8)
    
    return {
        "Base_AP": base_ap,
        "Novel_AP": novel_ap,
        "Harmonic_Mean": hm
    }


def evaluate_sgg_splits(
    records: Iterable[dict], 
    predictions: Iterable[dict], 
    ontology,
    novel_objects: Set[int],
    novel_predicates: Set[int],
    ks=(50, 100)
) -> Dict[str, Dict[str, float]]:
    """Bảng 1: Fully Open-Vocabulary Scene Graph Generation (SGDet) mR@K trên 4 split."""
    records = list(records)
    predictions = list(predictions)
    
    splits = {"SS": [], "NS": [], "SN": [], "NN": []}
    
    results = {}
    for k in ks:
        results[f"mR@{k}"] = {}
        for split in splits:
            p_hits = defaultdict(int)
            p_total = defaultdict(int)
            
            for record, prediction in zip(records, predictions):
                object_index = {int(object_id): index for index, object_id in enumerate(record["object_ids"])}
                gt_triplets = set()
                gt_splits = {}
                for relation in record["relations"]:
                    subj_id = object_index.get(int(relation["subject_id"]))
                    obj_id = object_index.get(int(relation["object_id"]))
                    if subj_id is not None and obj_id is not None:
                        pred_idx = int(relation["predicate_index"])
                        triplet = (subj_id, pred_idx, obj_id)
                        gt_triplets.add(triplet)
                        subj_label = int(record["leaf_indices"][subj_id])
                        obj_label = int(record["leaf_indices"][obj_id])
                        has_novel_obj = (subj_label in novel_objects) or (obj_label in novel_objects)
                        has_novel_rel = pred_idx in novel_predicates
                        if not has_novel_obj and not has_novel_rel: s = "SS"
                        elif has_novel_obj and not has_novel_rel: s = "NS"
                        elif not has_novel_obj and has_novel_rel: s = "SN"
                        else: s = "NN"
                        gt_splits[triplet] = s
                        if s == split:
                            p_total[pred_idx] += 1
                
                slot_to_gt = {}
                for candidate in prediction.get("objects", []):
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
                    subj_idx = slot_to_gt.get(int(relation.get("subject_slot", -1)))
                    obj_idx = slot_to_gt.get(int(relation.get("object_slot", -1)))
                    pred_name = relation.get("predicate")
                    pred_idx = ontology.predicate_index(pred_name) if pred_name in ontology.predicate_to_idx else -1
                    if subj_idx is not None and obj_idx is not None and pred_idx >= 0:
                        candidate_triplets.append((float(relation.get("score", 0.0)), (subj_idx, pred_idx, obj_idx)))
                
                candidate_triplets.sort(key=lambda item: item[0], reverse=True)
                hits = {t for _, t in candidate_triplets[:k]}
                
                for t in hits & gt_triplets:
                    if gt_splits[t] == split:
                        p_hits[t[1]] += 1
            
            per_predicate = [p_hits[idx] / total for idx, total in p_total.items() if total > 0]
            mR = (sum(per_predicate) / len(per_predicate)) * 100 if per_predicate else 0.0
            results[f"mR@{k}"][split] = mR
            
    return results


def evaluate_killer_metrics(
    records: Iterable[dict], 
    predictions: Iterable[dict], 
    ontology,
    train_triplets: Set[Tuple[int, int, int]]
) -> Dict[str, float]:
    """Bảng 3: zsR@100, Sibling Confusion, Group Consistency."""
    records = list(records)
    predictions = list(predictions)
    
    # 1. Zero-shot Triplet Recall (zsR@100)
    zs_total = 0
    zs_hits = 0
    
    # 2. Sibling Confusion
    total_incorrect = 0
    sibling_errors = 0
    
    # 3. Group Consistency
    total_objects = 0
    consistent_objects = 0
    
    for record, prediction in zip(records, predictions):
        gt_boxes = record["boxes"].tolist()
        gt_labels = record["leaf_indices"].tolist()
        
        # --- Group Consistency & Sibling Confusion ---
        for candidate in prediction.get("objects", []):
            label_name = candidate.get("label")
            label_idx = ontology.leaf_index(label_name) if label_name in ontology.leaf_to_idx else -1
            if label_idx < 0:
                continue
                
            leaf_prob = candidate.get("score", 0.0)
            group_scores = candidate.get("group_scores", [])
            
            # Group Consistency: P(Leaf) <= P(Group) for ALL ancestor groups
            ancestors = ontology.leaf_to_groups.get(label_idx, [])
            if ancestors and group_scores:
                total_objects += 1
                is_consistent = True
                for g_idx in ancestors:
                    if g_idx < len(group_scores) and leaf_prob > group_scores[g_idx]:
                        is_consistent = False
                        break
                if is_consistent:
                    consistent_objects += 1

            # Sibling Confusion
            best_overlap = -1.0
            best_gt_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                overlap = box_iou(candidate["box"], gt_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt_idx = gt_idx
                    
            if best_gt_idx is not None and best_overlap >= 0.5:
                gt_label_idx = gt_labels[best_gt_idx]
                if label_idx != gt_label_idx:
                    total_incorrect += 1
                    siblings = ontology.sibling_map.get(gt_label_idx, [])
                    if label_idx in siblings:
                        sibling_errors += 1

        # --- Zero-shot Triplet Recall (zsR@100) ---
        object_index = {int(object_id): index for index, object_id in enumerate(record["object_ids"])}
        gt_triplets = set()
        for relation in record["relations"]:
            subj_id = object_index.get(int(relation["subject_id"]))
            obj_id = object_index.get(int(relation["object_id"]))
            if subj_id is not None and obj_id is not None:
                pred_idx = int(relation["predicate_index"])
                subj_label = int(record["leaf_indices"][subj_id])
                obj_label = int(record["leaf_indices"][obj_id])
                triplet_semantic = (subj_label, pred_idx, obj_label)
                
                if triplet_semantic not in train_triplets:
                    gt_triplets.add((subj_id, pred_idx, obj_id))
                    
        zs_total += len(gt_triplets)
        
        pred_objects = prediction.get("objects", [])
        slot_to_gt = {}
        for candidate in pred_objects:
            slot = int(candidate.get("slot", -1))
            best = (-1.0, None)
            for gt_index, gt_box in enumerate(record["boxes"].tolist()):
                overlap = box_iou(candidate["box"], gt_box)
                gt_label_name = ontology.leaf_names()[int(record["leaf_indices"][gt_index])]
                if overlap > best[0] and candidate.get("label") == gt_label_name:
                    best = (overlap, gt_index)
            if best[1] is not None and best[0] >= 0.5:
                slot_to_gt[slot] = best[1]

        candidate_triplets = []
        for relation in prediction.get("relations", []):
            subj_idx = slot_to_gt.get(int(relation.get("subject_slot", -1)))
            obj_idx = slot_to_gt.get(int(relation.get("object_slot", -1)))
            pred_name = relation.get("predicate")
            pred_idx = ontology.predicate_index(pred_name) if pred_name in ontology.predicate_to_idx else -1
            if subj_idx is not None and obj_idx is not None and pred_idx >= 0:
                candidate_triplets.append((float(relation.get("score", 0.0)), (subj_idx, pred_idx, obj_idx)))
                
        candidate_triplets.sort(key=lambda item: item[0], reverse=True)
        hits = {t for _, t in candidate_triplets[:100]}
        zs_hits += len(hits & gt_triplets)

    return {
        "zsR@100": (zs_hits / max(zs_total, 1)) * 100,
        "Sibling_Confusion": (sibling_errors / max(total_incorrect, 1)) * 100,
        "Group_Consistency": (consistent_objects / max(total_objects, 1)) * 100
    }
