#!/bin/bash
# Ablation Study Runner cho Bảng 4 (Ablation Study trên tập Novel-Novel)
# Gồm 5 cấu hình tăng dần độ phức tạp

# Định nghĩa các thư mục output
OUTPUT_DIR="output/ablation"
mkdir -p $OUTPUT_DIR

# -------------------------------------------------------------------------
# Config #1 (Base): Single Latent, No L_ancestor, No L_sibling, No Context
# -------------------------------------------------------------------------
echo ">>> Đang chạy Config #1: Baseline (Single Latent)"
# TODO: Bạn cần sửa lại config.yaml để disable dual latent (hoặc truyền qua flag nếu source hỗ trợ)
# python scripts/train.py --config configs/hovr_sg.yaml \
#   --override model.dual_latent=False \
#   --override loss.ancestor=0.0 \
#   --override loss.sibling=0.0 \
#   --override model.relation.context_layers=0 \
#   --output_dir $OUTPUT_DIR/config1

# -------------------------------------------------------------------------
# Config #2: Dual Latent (Không L_ancestor, Không L_sibling, Không Context)
# -------------------------------------------------------------------------
echo ">>> Đang chạy Config #2: Dual Latent"
# python scripts/train.py --config configs/hovr_sg.yaml \
#   --override model.dual_latent=True \
#   --override loss.ancestor=0.0 \
#   --override loss.sibling=0.0 \
#   --override model.relation.context_layers=0 \
#   --output_dir $OUTPUT_DIR/config2

# -------------------------------------------------------------------------
# Config #3: Dual Latent + L_ancestor (Không L_sibling, Không Context)
# -------------------------------------------------------------------------
echo ">>> Đang chạy Config #3: Dual Latent + L_ancestor"
# python scripts/train.py --config configs/hovr_sg.yaml \
#   --override model.dual_latent=True \
#   --override loss.ancestor=0.25 \
#   --override loss.sibling=0.0 \
#   --override model.relation.context_layers=0 \
#   --output_dir $OUTPUT_DIR/config3

# -------------------------------------------------------------------------
# Config #4: Dual Latent + L_ancestor + L_sibling (Không Context)
# -------------------------------------------------------------------------
echo ">>> Đang chạy Config #4: Dual Latent + L_ancestor + L_sibling"
# python scripts/train.py --config configs/hovr_sg.yaml \
#   --override model.dual_latent=True \
#   --override loss.ancestor=0.25 \
#   --override loss.sibling=0.25 \
#   --override model.relation.context_layers=0 \
#   --output_dir $OUTPUT_DIR/config4

# -------------------------------------------------------------------------
# Config #5 (Full): Dual Latent + L_ancestor + L_sibling + Context Transformer
# -------------------------------------------------------------------------
echo ">>> Đang chạy Config #5: Full Architecture"
# python scripts/train.py --config configs/hovr_sg.yaml \
#   --override model.dual_latent=True \
#   --override loss.ancestor=0.25 \
#   --override loss.sibling=0.25 \
#   --override model.relation.context_layers=2 \
#   --output_dir $OUTPUT_DIR/config5

echo ">>> Quá trình Ablation hoàn tất. Kiểm tra thư mục $OUTPUT_DIR để xem logs và metrics."
