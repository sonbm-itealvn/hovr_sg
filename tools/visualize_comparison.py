"""Script to visualize side-by-side Scene Graph comparisons (GT vs Baseline vs HOVR-SG)."""

import argparse
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_scene_graph(image: np.ndarray, objects: List[Dict], relations: List[Dict], title: str, highlight_errors: bool = False) -> np.ndarray:
    """
    Draw bounding boxes and relations on an image.
    objects: list of dicts with 'box': [x1,y1,x2,y2], 'label': str, 'error': bool (optional)
    relations: list of dicts with 'sub_idx', 'obj_idx', 'predicate', 'error': bool (optional)
    """
    img_draw = image.copy()
    
    # Draw objects
    centers = []
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = map(int, obj['box'])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centers.append((cx, cy))
        
        # Color coding: Green if correct, Red if error and highlight is ON
        color = (0, 0, 255) if (highlight_errors and obj.get('error')) else (0, 255, 0)
        thickness = 4 if (highlight_errors and obj.get('error')) else 2
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background for visibility
        label = obj['label']
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_draw, (x1, max(0, y1 - 25)), (x1 + w, y1), color, -1)
        cv2.putText(img_draw, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0) if color == (0,255,0) else (255,255,255), 2)
        
    # Draw relations
    for rel in relations:
        s_idx, o_idx = rel['sub_idx'], rel['obj_idx']
        if s_idx >= len(centers) or o_idx >= len(centers): continue
        
        pt1 = centers[s_idx]
        pt2 = centers[o_idx]
        
        color = (0, 0, 255) if (highlight_errors and rel.get('error')) else (255, 165, 0) # Red if error, Orange otherwise
        thickness = 3 if (highlight_errors and rel.get('error')) else 2
        
        # Draw line
        cv2.arrowedLine(img_draw, pt1, pt2, color, thickness, tipLength=0.05)
        
        # Draw text in the middle
        mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        label = rel['predicate']
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Offset slightly so it doesn't cross the line directly
        cv2.rectangle(img_draw, (mx - w//2 - 2, my - h - 5), (mx + w//2 + 2, my + 5), (0,0,0), -1)
        cv2.putText(img_draw, label, (mx - w//2, my), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img_draw

def plot_comparison(image_path: str, output_path: str):
    """Plot the 3-panel comparison figure."""
    image = cv2.imread(image_path)
    if image is None:
        # Create a dummy image for the template if file not found
        image = np.ones((600, 800, 3), dtype=np.uint8) * 200
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # -------------------------------------------------------------
    # DUMMY DATA FOR TEMPLATE
    # Replace these with data extracted from your evaluation outputs
    # -------------------------------------------------------------
    
    # 1. Ground Truth
    gt_objects = [
        {'box': [100, 150, 300, 500], 'label': 'man'},
        {'box': [350, 400, 450, 550], 'label': 'cup'}
    ]
    gt_relations = [
        {'sub_idx': 0, 'obj_idx': 1, 'predicate': 'holding'}
    ]
    
    # 2. Baseline (Errors highlighted)
    base_objects = [
        {'box': [100, 150, 300, 500], 'label': 'man'},
        {'box': [350, 400, 450, 550], 'label': 'bottle', 'error': True} # Sibling confusion error
    ]
    base_relations = [
        {'sub_idx': 0, 'obj_idx': 1, 'predicate': 'near', 'error': True} # Relation hallucination
    ]
    
    # 3. HOVR-SG (Ours)
    ours_objects = [
        {'box': [102, 148, 298, 505], 'label': 'man'},
        {'box': [345, 402, 452, 545], 'label': 'cup'}
    ]
    ours_relations = [
        {'sub_idx': 0, 'obj_idx': 1, 'predicate': 'holding'}
    ]
    # -------------------------------------------------------------
    
    img_gt = draw_scene_graph(image, gt_objects, gt_relations, "Ground Truth")
    img_base = draw_scene_graph(image, base_objects, base_relations, "Baseline", highlight_errors=True)
    img_ours = draw_scene_graph(image, ours_objects, ours_relations, "HOVR-SG (Ours)")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(img_gt)
    axes[0].set_title("Ground Truth", fontsize=20, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_base)
    axes[1].set_title("Baseline (with errors)", fontsize=20, fontweight='bold', color='red')
    axes[1].axis('off')
    
    axes[2].imshow(img_ours)
    axes[2].set_title("HOVR-SG (Ours)", fontsize=20, fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Qualitative Comparison to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize HOVR-SG vs Baseline Comparison.")
    parser.add_argument("--image", type=str, default="sample.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="qualitative_comparison.png", help="Output path")
    args = parser.parse_args()
    
    plot_comparison(args.image, args.output)

if __name__ == "__main__":
    main()
