# Sinh ontology cho Visual Genome, Open Images V6 và GQA

Các dataset có format khác nhau nên repository cung cấp ba generator riêng nhưng dùng chung policy/canonicalization và output schema. Mỗi generator xuất một ontology JSON và một coverage report JSON. Nên review report trước khi convert dữ liệu và training.

## Open Images V6

Open Images dùng CSV boxes, class descriptions và relationship annotations:

```bash
python tools/build_open_images_ontology.py \
  --boxes-csv /data/openimages/oidv6-train-annotations-bbox.csv \
  --relations-csv /data/openimages/oidv6-train-annotations-vrd.csv \
  --class-descriptions /data/openimages/class-descriptions-boxable.csv \
  --policy configs/open_images_ontology_policy.example.yaml \
  --ontology-output ontology/ontology_open_images_v6_v1.json \
  --report-output reports/ontology_open_images_v6_v1.json
```

`--class-descriptions` map MID/LabelName sang display name. Nếu không cung cấp, generator dùng trực tiếp `ClassName` hoặc `LabelName` trong boxes CSV. Predicate được đọc từ `RelationshipLabel`, `Predicate`, `relation` hoặc `LabelName`.

Sau đó convert unified JSONL:

```bash
python tools/convert_open_images.py \
  --images-root /data/openimages/images \
  --boxes-csv /data/openimages/oidv6-train-annotations-bbox.csv \
  --relations-csv /data/openimages/oidv6-train-annotations-vrd.csv \
  --class-descriptions /data/openimages/class-descriptions-boxable.csv \
  --ontology ontology/ontology_open_images_v6_v1.json \
  --output data/open_images_unified.jsonl \
  --annotation-scope exhaustive
```

## GQA

GQA scene graphs thường là JSON object mapping image ID sang graph; generator cũng hỗ trợ list records và object/relation list hoặc dict:

```bash
python tools/build_gqa_ontology.py \
  --scene-graphs /data/gqa/train_sceneGraphs.json \
  --policy configs/gqa_ontology_policy.example.yaml \
  --ontology-output ontology/ontology_gqa_v1.json \
  --report-output reports/ontology_gqa_v1.json
```

Convert unified JSONL:

```bash
python tools/convert_gqa.py \
  --scene-graphs /data/gqa/train_sceneGraphs.json \
  --images-root /data/gqa/images \
  --ontology ontology/ontology_gqa_v1.json \
  --output data/gqa_unified.jsonl
```

GQA và Visual Genome thường có annotation scope partial; không biến mọi cặp object thiếu relation thành negative nếu annotation không exhaustive.

## Output và review

Cả ba generator đều tạo các field `object_groups`, `object_leaves`, `predicate_groups` và `predicates` tương thích với `Ontology`. Leaf có `parents`, `aliases`, `siblings`; predicate có `parents`, `aliases` và `symmetric`. Các field `frequency` và `image_frequency` được giữ trong JSON để audit nhưng không ảnh hưởng loader hiện tại.

Coverage report gồm số label unique thô, số label được giữ, số instance, coverage theo instance và top frequency labels. `min_object_count`, `min_predicate_count`, `max_object_leaves` và `max_predicates` là các tham số dataset-specific; không nên dùng một threshold cố định cho mọi benchmark.

Generator dùng regex group inference và alias policy, vì vậy output tự động là **điểm khởi đầu cần review**, không phải ontology ngữ nghĩa hoàn hảo. Sau khi review, giữ nguyên file ontology trong train/validation/test và lưu hash của file cùng checkpoint.
