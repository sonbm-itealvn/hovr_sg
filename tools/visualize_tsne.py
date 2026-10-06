"""Script to visualize the Dual Latent Space (Text Prototypes vs Visual Embeddings)."""

import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Try importing umap, fallback to TSNE if not available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def extract_embeddings(
    model, 
    encoder, 
    prototypes, 
    dataloader, 
    device: torch.device, 
    max_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất Text Prototypes và Visual Embeddings từ mô hình bằng cách chạy inference.
    """
    model.eval()
    encoder.eval()
    
    # 1. Trích xuất Text Prototypes trực tiếp từ tham số
    with torch.no_grad():
        leaf_protos, group_protos, _ = prototypes()
        leaf_protos_np = leaf_protos.cpu().numpy()
        group_protos_np = group_protos.cpu().numpy()
    
    # 2. Trích xuất Visual Embeddings bằng cách chạy qua dataloader
    all_z_leaf = []
    all_z_group = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size > max_samples:
                break
                
            images = batch["images"].to(device)
            visual = encoder(images)
            
            # Forward pass lấy raw embeddings (có thể thay đổi tuỳ implementation thực tế của bạn)
            out = model(visual, prototypes.leaf, prototypes.groups, prototypes.relations)
            
            # z_leaf và z_group thường là [B, num_queries, D]
            # Ta flatten nó thành [B * num_queries, D]
            batch_z_leaf = out.z_leaf.view(-1, out.z_leaf.shape[-1]).cpu().numpy()
            batch_z_group = out.z_group.view(-1, out.z_group.shape[-1]).cpu().numpy()
            
            # Lấy predicted labels hoặc GT labels để tô màu
            # Ở đây dùng predicted label cho tiện (max cosine similarity)
            leaf_logits = out.leaf_logits.view(-1, out.leaf_logits.shape[-1])
            pred_labels = leaf_logits.argmax(dim=-1).cpu().numpy()
            
            # Chỉ lấy các query có objectness score cao (loại bỏ background/noise)
            obj_scores = out.objectness_logits.sigmoid().view(-1)
            valid_mask = (obj_scores > 0.5).cpu().numpy()
            
            all_z_leaf.append(batch_z_leaf[valid_mask])
            all_z_group.append(batch_z_group[valid_mask])
            all_labels.append(pred_labels[valid_mask])
            
    if all_z_leaf:
        visual_z_leaf = np.concatenate(all_z_leaf, axis=0)
        visual_z_group = np.concatenate(all_z_group, axis=0)
        visual_labels = np.concatenate(all_labels, axis=0)
    else:
        # Fallback dummy nếu không có mẫu hợp lệ
        visual_z_leaf = np.random.randn(100, leaf_protos_np.shape[-1])
        visual_z_group = np.random.randn(100, group_protos_np.shape[-1])
        visual_labels = np.random.randint(0, len(leaf_protos_np), 100)
        
    return leaf_protos_np, group_protos_np, visual_z_leaf, visual_z_group, visual_labels


def plot_dual_latent_space(
    leaf_protos, group_protos, 
    visual_z_leaf, visual_z_group, visual_labels,
    ontology, output_path: str, method: str = 'umap',
    target_classes: List[str] = None
):
    """
    Project và vẽ Plot kết hợp (Visual Embeddings xoay quanh Text Prototypes).
    Để tránh nhiễu, chúng ta chỉ nên vẽ một vài classes tiêu biểu (target_classes).
    """
    print(f"Projecting embeddings using {method.upper()}...")
    
    # 1. Chọn lọc các class cần hiển thị (ví dụ: dog, cat, car, person...)
    if target_classes is None:
        target_classes = ["person", "car", "dog", "cup", "chair"]
        
    target_indices = [ontology.leaf_index(c) for c in target_classes if c in ontology.leaf_to_idx]
    
    # Lọc visual embeddings thuộc về các classes đó
    mask = np.isin(visual_labels, target_indices)
    filt_z_leaf = visual_z_leaf[mask]
    filt_labels = visual_labels[mask]
    
    # Lọc Text Prototypes của các classes đó
    filt_leaf_protos = leaf_protos[target_indices]
    
    # Gộp tất cả lại để chạy UMAP/t-SNE cùng một không gian
    # Thứ tự: [Filtered Visual Leaf, Filtered Leaf Protos, Group Protos]
    all_embs = np.concatenate([filt_z_leaf, filt_leaf_protos, group_protos], axis=0)
    
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        proj = reducer.fit_transform(all_embs)
    else:
        if method == 'umap':
            print("UMAP not found (pip install umap-learn). Falling back to t-SNE.")
        reducer = TSNE(n_components=2, metric='cosine', perplexity=min(30, len(all_embs)-1), random_state=42)
        proj = reducer.fit_transform(all_embs)
        
    # Tách ngược lại
    n_vis = len(filt_z_leaf)
    n_lproto = len(filt_leaf_protos)
    
    proj_vis_leaf = proj[:n_vis]
    proj_lproto = proj[n_vis:n_vis+n_lproto]
    proj_gproto = proj[n_vis+n_lproto:]
    
    # ---------------- PLOTTING ----------------
    plt.figure(figsize=(14, 10))
    
    # Màu sắc cho từng class
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_indices)))
    
    # 1. Vẽ Visual Embeddings (Các chấm mờ)
    for idx, (cls_idx, color) in enumerate(zip(target_indices, colors)):
        cls_mask = (filt_labels == cls_idx)
        cls_name = ontology.leaf_names()[cls_idx]
        plt.scatter(
            proj_vis_leaf[cls_mask, 0], proj_vis_leaf[cls_mask, 1], 
            c=[color], alpha=0.3, s=40, edgecolors='none', label=f'Visual: {cls_name}'
        )
        
    # 2. Vẽ Text Leaf Prototypes (Các ngôi sao đậm)
    for idx, (cls_idx, color) in enumerate(zip(target_indices, colors)):
        cls_name = ontology.leaf_names()[cls_idx]
        plt.scatter(
            proj_lproto[idx, 0], proj_lproto[idx, 1], 
            c=[color], alpha=1.0, s=300, marker='*', edgecolors='black', label=f'Proto: {cls_name}'
        )
        # Gắn text
        plt.annotate(cls_name, (proj_lproto[idx, 0], proj_lproto[idx, 1] + 0.5), 
                     fontsize=12, fontweight='bold', ha='center')
                     
    # 3. Vẽ Group Prototypes (Các hình thoi đen)
    # Chỉ vẽ các group có liên quan đến target_classes
    target_groups = set()
    for cls_idx in target_indices:
        target_groups.update(ontology.leaf_to_groups.get(cls_idx, []))
        
    target_groups = list(target_groups)
    for g_idx in target_groups:
        g_name = ontology.group_names()[g_idx]
        plt.scatter(
            proj_gproto[g_idx, 0], proj_gproto[g_idx, 1], 
            c='black', alpha=0.8, s=150, marker='D', label='Group Proto' if g_idx == target_groups[0] else ""
        )
        plt.annotate(f"[{g_name}]", (proj_gproto[g_idx, 0], proj_gproto[g_idx, 1] - 0.8), 
                     fontsize=13, fontweight='bold', ha='center', color='black')

    plt.title("Latent Space: Visual Features clustering around Text Prototypes", fontsize=18)
    
    # Fix legend trùng lặp
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved qualitative visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize HOVR-SG Dual Latent Space with Visual Features.")
    parser.add_argument("--model", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="latent_space_alignment.png", help="Output image path")
    parser.add_argument("--method", type=str, choices=['tsne', 'umap'], default='umap', help="Projection method")
    args = parser.parse_args()

    print("Note: This template uses dummy extracted visual data. To run on real data, you must wire up your Encoder/Model and DataLoader in main().")

    # -------------------------------------------------------------
    # DUMMY DATA MOCKUP CHO TEMPLATE
    # -------------------------------------------------------------
    class DummyOntology:
        def __init__(self):
            self.leaves = ["person", "car", "dog", "cup", "chair", "tree", "bird"]
            self.groups = ["animal", "vehicle", "furniture", "object"]
            self.leaf_to_idx = {name: i for i, name in enumerate(self.leaves)}
            # Map dog(2)->animal(0), person(0)->animal(0), car(1)->vehicle(1), cup(3)->object(3), chair(4)->furniture(2)
            self.leaf_to_groups = {0: [0], 1: [1], 2: [0], 3: [3], 4: [2]}
        def leaf_names(self): return self.leaves
        def group_names(self): return self.groups
        def leaf_index(self, name): return self.leaf_to_idx[name]
        
    ontology = DummyOntology()
    
    # Mock Text Prototypes (D = 256)
    np.random.seed(42)
    leaf_protos_np = np.random.randn(len(ontology.leaves), 256) * 5
    group_protos_np = np.random.randn(len(ontology.groups), 256) * 5
    
    # Mock Visual Embeddings quần tụ quanh các leaf protos (Thêm gaussian noise)
    visual_z_leaf = []
    visual_labels = []
    for i, proto in enumerate(leaf_protos_np):
        samples = proto + np.random.randn(100, 256) * 2.0  # Thêm noise
        visual_z_leaf.append(samples)
        visual_labels.extend([i] * 100)
        
    visual_z_leaf = np.concatenate(visual_z_leaf, axis=0)
    visual_labels = np.array(visual_labels)
    visual_z_group = visual_z_leaf.copy() # Dummy for template

    plot_dual_latent_space(
        leaf_protos=leaf_protos_np, 
        group_protos=group_protos_np, 
        visual_z_leaf=visual_z_leaf, 
        visual_z_group=visual_z_group, 
        visual_labels=visual_labels,
        ontology=ontology, 
        output_path=args.output, 
        method=args.method,
        target_classes=["person", "car", "dog", "cup"] # Hiển thị 4 class để plot nhìn sạch đẹp
    )


if __name__ == "__main__":
    main()
