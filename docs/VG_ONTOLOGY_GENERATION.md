# Sinh ontology từ Visual Genome

Repository cung cấp `tools/build_vg_ontology.py` để đọc `objects.json` và `relationships.json` của Visual Genome, thống kê frequency, canonicalize số ít/số nhiều và alias, suy luận object/predicate group, tạo sibling lists, rồi sinh ontology JSON cùng coverage report.

## Chạy nhanh

```bash
python tools/build_vg_ontology.py \
  --objects /data/visual_genome/objects.json \
  --relationships /data/visual_genome/relationships.json \
  --policy configs/vg_ontology_policy.example.yaml \
  --ontology-output ontology/ontology_vg_v1.json \
  --report-output reports/ontology_vg_v1.json
```

Nếu chỉ muốn chạy pilot mà không dùng policy file:

```bash
python tools/build_vg_ontology.py \
  --objects /data/visual_genome/objects.json \
  --relationships /data/visual_genome/relationships.json \
  --min-object-count 100 \
  --min-predicate-count 100 \
  --max-object-leaves 500 \
  --max-predicates 100 \
  --ontology-output ontology/ontology_vg_v1.json \
  --report-output reports/ontology_vg_v1.json
```

CLI cũng có `--max-aliases-per-label`. Các giá trị trong command line ghi đè policy file.

## Ý nghĩa coverage report

Report JSON chứa số label unique thô, số label được chọn, số instance thô, số instance được giữ lại, coverage và top frequency labels cho objects và predicates. `coverage` được tính theo instance, không chỉ theo số class. Nếu coverage thấp, ontology đang lọc phần lớn annotation VG và cần xem lại ngưỡng, alias hoặc ignore list trước khi training.

Một ontology tự động sinh **không nên được xem là đã đúng ngữ nghĩa ngay lập tức**. Group rules hiện là heuristic dựa trên regex; alias có thể được cấu hình thủ công; sibling list được tạo theo group. Cần review các lớp có tần suất cao, lớp `other_object`/`other_relation`, synonym và các predicate direction trước khi sử dụng cho benchmark chính thức.

## Dùng ontology đã sinh để convert VG

```bash
python tools/convert_visual_genome.py \
  --images-root /data/visual_genome/VG_100K \
  --image-data /data/visual_genome/image_data.json \
  --objects /data/visual_genome/objects.json \
  --relationships /data/visual_genome/relationships.json \
  --ontology ontology/ontology_vg_v1.json \
  --output data/vg_unified.jsonl
```

Sau đó inspect unified JSONL và kiểm tra một vài ảnh thủ công. Giữ nguyên `ontology_vg_v1.json` trong suốt train/validation/test; file ontology phải được lưu cùng checkpoint vì nó quyết định index của leaf, group và predicate.

## Ghi chú về canonicalization

Generator normalize chữ thường, thay `_` và `-` bằng khoảng trắng, rút gọn một số plural phổ biến và áp dụng `aliases` trong policy. Nó không dùng fuzzy matching vì fuzzy matching có thể gộp nhầm các lớp khác nghĩa. Những mapping như `bike → bicycle`, `tv → television`, `cellphone → cell phone` nên được khai báo rõ trong policy.
