# HOVR-SG: Google Colab training và checkpoint release

Tài liệu này mô tả quy trình tạo một checkpoint HOVR-SG có thể phát hành từ dữ liệu scene-graph đã chuẩn hóa. Pipeline không xem `TinyImageEncoder` hoặc pseudo-text embedding là model release; cấu hình release mặc định dùng CLIP ViT-B/32 pretrained, CLIP text prototypes, Hungarian matching, validation và lựa chọn `best.pt` theo metric đã khai báo.

Nếu dùng Visual Genome, có thể mở trực tiếp notebook [`notebooks/hovr_sg_visual_genome_colab.ipynb`](../notebooks/hovr_sg_visual_genome_colab.ipynb) trên Google Colab. Notebook thực hiện cả các bước sinh ontology, convert raw annotations, tạo train/val và novelty splits, training, validate, evaluation và copy artifact về Google Drive.

## 1. Chuẩn bị runtime Colab

Trong Colab, chọn **GPU runtime** rồi chạy:

```bash
!git clone https://github.com/sonbm-itealvn/hovr_sg.git
%cd hovr_sg
!pip install -e .
```

Kiểm tra GPU và package:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Chuẩn bị dữ liệu

Mỗi dòng JSONL phải tuân theo unified scene-graph schema. `image_path` có thể là đường dẫn tương đối; khi đó truyền thư mục ảnh qua `--image-root`. Với Visual Genome, dùng `tools/convert_visual_genome.py` rồi `tools/build_splits.py`; tool split sẽ sinh `train.jsonl`, `val.jsonl`, `ss.jsonl`, `ns.jsonl`, `sn.jsonl` và `nn.jsonl`. Mặc định train/val lấy từ pool `ss` để giữ strict zero-shot; dùng `--train-val-source all` chỉ cho pilot không strict.

Ví dụ kiểm tra dataset:

```bash
!python scripts/inspect_dataset.py \
  --jsonl /content/data/train.jsonl \
  --ontology ontology/ontology_v1.json
```

Với dữ liệu Visual Genome, Open Images hoặc GQA, chạy converter tương ứng trước khi training và tạo split train/validation riêng. Không đưa ảnh validation vào train split.

## 3. Training model release

Lệnh dưới đây dùng CLIP pretrained mặc định, augmentation train-only, bốn training stages, AMP và validation. `best.pt` được chọn theo `validation.selection_metric`, mặc định là `object_mAP50_95`.

```bash
!python scripts/train.py \
  --config configs/hovr_sg.yaml \
  --train-jsonl /content/data/train.jsonl \
  --val-jsonl /content/data/val.jsonl \
  --ontology ontology/ontology_v1.json \
  --image-root /content/data/images \
  --output-dir /content/runs/hovr_sg_release \
  --device cuda
```

### Cấu hình ngắn cho Colab T4

Đoạn cấu hình trong ảnh không đặt số epoch. `validation.frequency: 1` chỉ có nghĩa là validation sau mỗi epoch; `selection_metric: relation_Recall@50` chỉ quyết định metric dùng để chọn `best.pt`; `training.amp: true` bật mixed precision trên CUDA/T4.

Nếu bạn chỉ có khoảng 4 giờ 30 phút, đặt trong cell tạo config:

```python
EPOCHS = 2
config['training']['epochs'] = EPOCHS
config['training']['amp'] = True
config['validation']['frequency'] = 1
config['validation']['selection_metric'] = 'object_mAP50_95'
```

Với stage schedule mặc định `2 + 2 + 3 + 3`, chạy 2 epoch sẽ chỉ hoàn tất `detector_warmup`; relation head và joint stage chưa được huấn luyện đầy đủ. Đây là checkpoint thử nghiệm/khởi đầu, chưa nên xem là model scene-graph hoàn chỉnh. Nếu muốn 2 epoch có cả một bước joint nhanh, có thể dùng schedule rút gọn:

```yaml
stages:
  detector_warmup_epochs: 1
  hierarchical_epochs: 0
  relation_epochs: 0
  joint_epochs: 1
```

Cấu hình rút gọn này phù hợp để kiểm tra toàn pipeline trên T4, nhưng chất lượng thường thấp hơn training đủ 10 epoch. Có thể chạy tiếp từ `last.pt` bằng `--resume` sau khi còn thời gian hoặc phiên Colab mới.

Các artifact quan trọng sau khi hoàn tất:

| File | Ý nghĩa |
|---|---|
| `last.pt` | Checkpoint mới nhất, chứa model, encoder, prototypes, optimizer, scaler và stage state để resume. |
| `best.pt` | Checkpoint được chọn theo validation metric; đây là file dùng cho release/inference. |
| `best_manifest.json` | Metric chọn best, epoch, commit code và hash ontology. |
| `training_summary.json` | Stage schedule, lịch sử train/validation, config và best metrics. |

Nếu muốn chọn model theo scene-graph metric thay vì object detection, sửa config:

```yaml
validation:
  frequency: 1
  selection_metric: relation_Recall@50
```

Để bật AMP, đặt:

```yaml
training:
  amp: true
```

AMP chỉ được kích hoạt khi runtime có CUDA; CPU sẽ tự chạy FP32.

## 4. Resume khi Colab bị ngắt

Colab nên lưu output vào Google Drive hoặc tải `last.pt` lên storage bền vững. Khi tiếp tục, dùng đúng config, ontology và dataset split:

```bash
!python scripts/train.py \
  --config configs/hovr_sg.yaml \
  --train-jsonl /content/data/train.jsonl \
  --val-jsonl /content/data/val.jsonl \
  --ontology ontology/ontology_v1.json \
  --image-root /content/data/images \
  --output-dir /content/runs/hovr_sg_resume \
  --resume /content/runs/hovr_sg_release/last.pt \
  --device cuda
```

Checkpoint resume khôi phục model state, prototypes, optimizer, AMP scaler, epoch/stage progress, lịch sử validation và best score. Không đổi ontology hoặc `model.backbone_name` giữa các lần resume.

Để chạy tiếp mà không sửa `training.epochs` trong YAML, dùng `--additional-epochs`. Ví dụ checkpoint đang ở epoch 10 và muốn chạy thêm 10 epoch:

```bash
!python scripts/train.py \
  --config configs/hovr_sg.yaml \
  --train-jsonl /content/data/train.jsonl \
  --val-jsonl /content/data/val.jsonl \
  --ontology ontology/ontology_vg_v1.json \
  --image-root /content/data/images \
  --output-dir /content/runs/hovr_sg_release \
  --resume /content/runs/hovr_sg_release/last.pt \
  --additional-epochs 10 \
  --lr 1e-5 \
  --device cuda
```

CLI sẽ tự tính tổng epoch mới, mở rộng stage cuối, khôi phục lịch sử cũ và áp dụng learning rate override cho optimizer. Không truyền đồng thời `--epochs` và `--additional-epochs`.

## 5. Kiểm tra checkpoint trước khi phát hành

Chạy validator sau khi training:

```bash
!python scripts/validate_checkpoint.py \
  --checkpoint /content/runs/hovr_sg_release/best.pt \
  --ontology ontology/ontology_v1.json
```

Validator kiểm tra checkpoint type, format version, đủ encoder/model/prototype state, đồng nhất text dimension và hash ontology. Chỉ dùng `best.pt` nếu validator trả về `"valid": true`.

## 6. Evaluation và inference

```bash
!python scripts/evaluate.py \
  --checkpoint /content/runs/hovr_sg_release/best.pt \
  --jsonl /content/data/test.jsonl \
  --ontology ontology/ontology_v1.json \
  --image-root /content/data/images \
  --output /content/runs/hovr_sg_release/test_predictions.json \
  --device cuda
```

Output chứa raw predictions, object `AP50`, `AP75`, `mAP@[.50:.95]`, relation `Recall@20/50/100` và `mean Recall`. Với model release thực tế, cần lưu kèm test split manifest, ontology hash, config và commit code để kết quả có thể tái lập.

## 7. Tải checkpoint về máy

```python
from google.colab import files
files.download('/content/runs/hovr_sg_release/best.pt')
files.download('/content/runs/hovr_sg_release/best_manifest.json')
files.download('/content/runs/hovr_sg_release/training_summary.json')
```

`best.pt` là checkpoint chính thức **của lần training và dataset cụ thể đó**. Repository không thể tự tạo chất lượng pretrained chỉ bằng việc sửa source; chất lượng cuối cùng phụ thuộc dataset, split, số epoch, GPU, seed, pretrained backbone và protocol đánh giá. Điểm khác biệt của pipeline mới là sau khi bạn chạy training trên Colab, kết quả có lifecycle đầy đủ để kiểm tra, resume, chọn best và phát hành có provenance thay vì chỉ là `last.pt` từ một research smoke run.
