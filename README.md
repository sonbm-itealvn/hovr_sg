# HOVR-SG

**HOVR-SG** là một skeleton nghiên cứu PyTorch cho **Hierarchical Open-Vocabulary Relational Scene Graph Detection**. Package hiện thực hóa blueprint đã đề xuất: object query kiểu DETR, hai không gian latent `leaf/group`, open-vocabulary prototype scoring, sparse relation decoder, ontology mapping và dataset adapters cho Visual Genome, Open Images V6 và GQA.

> Đây là research source có thể mở rộng, không phải checkpoint pretrained. Fallback encoder trong package chỉ dùng để smoke test; thực nghiệm mở vocabulary nghiêm túc nên dùng backbone VLM/grounded detector như Grounding DINO, CLIP/SigLIP hoặc encoder tương đương.

## Cấu trúc

```text
hovr_sg_package/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── configs/
│   ├── hovr_sg.yaml
│   └── dataset_paths.example.yaml
├── ontology/
│   ├── ontology_v1.json
│   ├── object_leaf.txt
│   ├── object_group.txt
│   └── predicates.txt
├── hovr_sg/
│   ├── models/
│   │   ├── backbone.py
│   │   ├── hovr_sg.py
│   │   └── prototypes.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── unified_dataset.py
│   │   └── adapters.py
│   ├── losses/
│   │   └── hovr_losses.py
│   └── utils/
│       ├── config.py
│       ├── ontology.py
│       └── seed.py
├── tools/
│   ├── convert_visual_genome.py
│   ├── convert_open_images.py
│   ├── convert_gqa.py
│   └── build_splits.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_ontology.py
│   └── test_smoke.py
└── docs/
    ├── design_vi.md
    └── research_notes.md
```

## 1. Cài đặt

Tạo môi trường Python 3.10 hoặc mới hơn, sau đó cài PyTorch phù hợp với CUDA trước. Ví dụ:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Cài torch/torchvision theo CUDA từ https://pytorch.org/get-started/locally/
pip install -e .
```

Nếu muốn dùng text encoder CLIP từ Hugging Face:

```bash
pip install -e ".[vlm]"
```

Nếu chỉ cần chạy unit test và fallback smoke test, phần dependencies cơ bản là đủ.

## 2. Chuẩn bị ontology

Ontology nằm trong `ontology/ontology_v1.json`. Mỗi object leaf có `parents`, `aliases` và `siblings`. Ví dụ `man`, `woman`, `girl`, `boy` cùng có ancestor `person`; `chair` và `table` không dùng chung group con. Đây là điểm bảo vệ mô hình khỏi việc gom tất cả vật thể nội thất vào một cụm không phân biệt.

Bạn có thể chỉnh ontology bằng tay hoặc dùng converter để tạo label report trước. Không nên tự động coi hai label gần nhau trong embedding là synonym. Alias chỉ dùng khi cùng một annotation policy.

## 3. Chuyển đổi dataset

Tất cả converter đều đưa annotation về unified JSONL schema:

```json
{
  "image_id": "000001",
  "image_path": "/path/to/image.jpg",
  "width": 640,
  "height": 480,
  "objects": [
    {
      "id": 0,
      "bbox": [10, 20, 200, 300],
      "label": "man",
      "group_labels": ["person"],
      "attributes": [],
      "is_group": false,
      "source": "visual_genome"
    }
  ],
  "relations": [
    {
      "subject_id": 0,
      "object_id": 1,
      "predicate": "holding",
      "predicate_group": "contact_action",
      "source": "visual_genome"
    }
  ],
  "annotation_scope": "partial"
}
```

### Visual Genome

Tải `image_data.json`, `objects.json`, `relationships.json`, `object_alias.txt` hoặc file tương đương từ trang chính thức Visual Genome. Sau đó chạy:

```bash
python tools/convert_visual_genome.py \
  --images-root /data/VG_100K \
  --image-data /data/image_data.json \
  --objects /data/objects.json \
  --relationships /data/relationships.json \
  --ontology ontology/ontology_v1.json \
  --output data/vg_unified.jsonl
```

Visual Genome có annotation scope không hoàn toàn exhaustive. Converter mặc định đánh dấu `annotation_scope=partial`; training loader sẽ không biến mọi cặp không có relation thành negative.

### Open Images V6

Open Images cần các file CSV chính thức: boxes, relationships, class descriptions, image IDs và thư mục ảnh. Converter hỗ trợ quan hệ bằng MIDs hoặc tên class tùy phiên bản CSV:

```bash
python tools/convert_open_images.py \
  --images-root /data/openimages \
  --boxes-csv /data/oidv6-train-annotations-bbox.csv \
  --relations-csv /data/oidv6-train-annotations-vrd.csv \
  --class-descriptions /data/class-descriptions-boxable.csv \
  --ontology ontology/ontology_v1.json \
  --output data/oi_train_unified.jsonl \
  --annotation-scope exhaustive
```

Chỉ dùng negative pair từ Open Images khi relation annotation và object annotation của image/split được xác định là exhaustive. Với image-level negative label, có thể đưa vào `negative_labels` nếu bạn muốn mở rộng hard-negative mining.

### GQA

GQA thường cung cấp scene graph JSON có cấu trúc object/relations khác nhau giữa các bản phát hành. Converter có cơ chế đọc một số biến thể phổ biến:

```bash
python tools/convert_gqa.py \
  --scene-graphs /data/gqa/train_sceneGraphs.json \
  --images-root /data/gqa/images \
  --ontology ontology/ontology_v1.json \
  --output data/gqa_train_unified.jsonl
```

GQA được khuyến nghị như auxiliary scene-graph data hoặc clean validation, không nên trộn trực tiếp với VG nếu chưa kiểm tra mapping relation direction.

## 4. Xây dựng split base/novel

Tạo split `SS`, `NS`, `SN`, `NN` và novel-composition:

```bash
python tools/build_splits.py \
  --input data/vg_unified.jsonl data/oi_train_unified.jsonl \
  --ontology ontology/ontology_v1.json \
  --output-dir data/splits \
  --object-novel-ratio 0.20 \
  --relation-novel-ratio 0.20 \
  --seed 42
```

Converter split theo canonical label sau khi alias đã được chuẩn hóa. Nếu dùng strict zero-shot, cần loại novel label khỏi mọi object/relation annotation của train; caption pretraining chứa novel label phải được ghi thành protocol riêng.

## 5. Training

Cấu hình mặc định nằm trong `configs/hovr_sg.yaml`. Training gồm các stage:

| Stage | Mục tiêu | Backbone |
|---|---|---|
| `detector_warmup` | Box, objectness, leaf/group prototype alignment | Freeze |
| `hierarchical` | Ancestor consistency và sibling separation | Freeze hoặc LoRA |
| `relation` | Sparse pair, relationness, predicate embedding | Freeze detector ban đầu |
| `joint` | Fine-tune end-to-end, giảm train–test mismatch | LoRA/unfreeze block cuối |

Chạy smoke training với fallback CNN trên unified JSONL:

```bash
python scripts/train.py \
  --config configs/hovr_sg.yaml \
  --train-jsonl data_sample/train.jsonl \
  --val-jsonl data_sample/val.jsonl \
  --ontology ontology/ontology_v1.json \
  --output-dir runs/smoke
```

Chạy dataset thật:

```bash
python scripts/train.py \
  --config configs/hovr_sg.yaml \
  --train-jsonl data/splits/train.jsonl \
  --val-jsonl data/splits/val.jsonl \
  --ontology ontology/ontology_v1.json \
  --image-root /data/images \
  --output-dir runs/hovr_v1 \
  --backbone pretrained_vlm
```

`pretrained_vlm` trong skeleton là interface để nối encoder thực tế. Fallback `tiny_cnn` chỉ chứng minh data/model/loss plumbing; nó không đại diện cho open-vocabulary performance.

## 6. Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint runs/hovr_v1/best.pt \
  --jsonl data/splits/test_nn.jsonl \
  --ontology ontology/ontology_v1.json \
  --output runs/hovr_v1/test_nn_metrics.json
```

Metric tối thiểu cần báo cáo gồm object AP theo base/novel, harmonic mean, relation Recall@20/50/100, mean Recall, zero-shot triplet recall, group consistency, sibling confusion và calibration. Không dùng riêng UMAP/t-SNE để kết luận latent space đã tốt.

## 7. Gắn Grounding DINO/CLIP/SigLIP

Skeleton tách `visual_features` khỏi text prototypes. Để gắn VLM thực tế, triển khai adapter trả về:

```python
visual_features: Tensor  # [B, S, visual_dim]
leaf_text: Tensor        # [num_leaf, d_latent]
group_text: Tensor       # [num_group, d_latent]
relation_text: Tensor    # [num_predicates, d_latent]
```

Text prototypes phải được encode cùng text encoder với region projection hoặc được map qua một projection layer đã học. Không trộn trực tiếp CLIP text embedding với SigLIP region embedding nếu chưa có alignment layer.

## 8. Giới hạn hiện tại

Package này là một nền tảng nghiên cứu có thể chạy và mở rộng, nhưng chưa bao gồm Hungarian matcher tối ưu hóa đầy đủ, COCO-style evaluator hoàn chỉnh và checkpoint VLM. Các phần này phụ thuộc phiên bản dataset, backbone và GPU/CUDA của dự án. Converter và unified schema đã được viết để giảm phần công việc thay đổi cấu hình dataset.

## References

[1]: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html "Visual Genome official dataset page"

[2]: https://storage.googleapis.com/openimages/web/factsfigures.html "Open Images V6 official facts and figures"

[3]: https://cs.stanford.edu/people/dorarad/gqa/about.html "GQA official dataset page"

[4]: https://arxiv.org/html/2303.05499v5 "Grounding DINO"

[5]: https://arxiv.org/html/2404.00906v2 "From Pixels to Graphs"
