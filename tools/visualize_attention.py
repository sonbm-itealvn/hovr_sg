"""Script to visualize Union-Region boxes and Attention Heatmaps for HOVR-SG."""

import argparse
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def get_union_box(box_s: List[float], box_o: List[float]) -> List[float]:
    """Calculate the union bounding box of subject and object."""
    return [
        min(box_s[0], box_o[0]),
        min(box_s[1], box_o[1]),
        max(box_s[2], box_o[2]),
        max(box_s[3], box_o[3])
    ]

def draw_boxes(image: np.ndarray, box_s: List[float], box_o: List[float]) -> np.ndarray:
    """Draw subject (red), object (blue), and union (dashed yellow) boxes."""
    img_draw = image.copy()
    
    # Coordinates (assuming absolute pixel values [x1, y1, x2, y2])
    x1_s, y1_s, x2_s, y2_s = map(int, box_s)
    x1_o, y1_o, x2_o, y2_o = map(int, box_o)
    x1_u, y1_u, x2_u, y2_u = map(int, get_union_box(box_s, box_o))
    
    # Draw Subject (Red)
    cv2.rectangle(img_draw, (x1_s, y1_s), (x2_s, y2_s), (255, 0, 0), 3)
    cv2.putText(img_draw, "Subject", (x1_s, max(0, y1_s - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Draw Object (Blue)
    cv2.rectangle(img_draw, (x1_o, y1_o), (x2_o, y2_o), (0, 0, 255), 3)
    cv2.putText(img_draw, "Object", (x1_o, max(0, y1_o - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Draw Union (Yellow)
    # Using a simple line for illustration. For dashed, multiple segments are needed.
    cv2.rectangle(img_draw, (x1_u, y1_u), (x2_u, y2_u), (255, 255, 0), 2)
    cv2.putText(img_draw, "Union Region", (x1_u, y2_u + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return img_draw

def overlay_attention(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay a 2D attention heatmap on the image."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_norm = np.uint8(255 * (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8))
    
    colormap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    # Convert BGR to RGB
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image, 0.5, colormap, 0.5, 0)
    return overlay

def visualize_prediction(image_path: str, box_s: List[float], box_o: List[float], heatmap: np.ndarray, output_path: str, predicate_label: str):
    """Generate the qualitative output figure."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_boxes = draw_boxes(image, box_s, box_o)
    img_attention = overlay_attention(image, heatmap)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(img_boxes)
    ax1.set_title("Bounding Boxes & Union Region", fontsize=16)
    ax1.axis("off")
    
    ax2.imshow(img_attention)
    ax2.set_title(f"Attention Heatmap (Predicate: '{predicate_label}')", fontsize=16)
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Qualitative Visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize HOVR-SG Attention and Regions.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="attention_map.png", help="Output path")
    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # DUMMY DATA: Replace this section with actual model outputs
    # ---------------------------------------------------------
    print("WARNING: Using dummy bounding boxes and heatmaps for the template.")
    # Assuming image is at least 500x500 for dummy coords
    box_s = [50, 100, 200, 400]   # [x1, y1, x2, y2]
    box_o = [250, 300, 450, 480]
    
    # Dummy heatmap: 16x16 grid simulating cross-attention weights
    heatmap = np.zeros((16, 16), dtype=np.float32)
    # Simulate high attention near the intersection/interaction area
    heatmap[8:12, 6:10] = 1.0 
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    predicate = "holding"
    # ---------------------------------------------------------

    visualize_prediction(args.image, box_s, box_o, heatmap, args.output, predicate)

if __name__ == "__main__":
    main()
