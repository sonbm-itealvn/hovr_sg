# Dataset mapping guide

## Unified schema

Mọi adapter phải tạo một JSON object trên mỗi dòng. `objects[].bbox` dùng dạng pixel `[x1,y1,x2,y2]`; loader mới chuẩn hóa về `[0,1]`. `objects[].id` phải ổn định trong từng ảnh vì relation dùng các ID này.

`annotation_scope` có ba giá trị nên dùng:

| Giá trị | Ý nghĩa | Cách dùng negative |
|---|---|---|
| `exhaustive` | Dataset đã liệt kê đầy đủ instance/relation trong phạm vi annotation | Có thể dùng cặp không có relation làm negative nếu object pair nằm trong scope |
| `partial` | Chỉ các positive được biết, cặp thiếu có thể là unknown | Không được tự động coi thiếu là negative |
| `unknown` | Chưa xác định policy | Mask khỏi relation negative loss |

## Visual Genome

`objects.json` thường chứa `image_id` và danh sách `objects`; mỗi object có `object_id`, `x`, `y`, `w`, `h`, `names`, `attributes`. `relationships.json` thường chứa `subject.object_id`, `object.object_id` và `predicate`. Converter canonicalize `names` và `predicate` qua ontology. Alias/synset chưa có trong ontology sẽ bị bỏ qua và được thống kê trong một bước audit riêng.

## Open Images V6

Box CSV thường có `ImageID`, `LabelName`, `XMin`, `XMax`, `YMin`, `YMax`, `IsGroupOf`; tùy bản tải có thể có thêm `ImageWidth`/`ImageHeight`. Converter nhận cả tọa độ normalized và pixel. Class description CSV map `LabelName`/MID sang `DisplayName`.

Relationship CSV có nhiều biến thể theo bản phát hành. Converter cố gắng đọc `ImageID`, `RelationshipLabel`, `SubjectIndex` và `ObjectIndex`. Nếu file chỉ cung cấp subject/object box coordinates thay vì index, cần bổ sung một bước matching IoU vào adapter trước khi training. Không nên nối relation vào object theo thứ tự dòng nếu file không đảm bảo cùng ordering.

Open Images có thể đánh dấu `group-of`; converter giữ cờ này để loss/evaluation quyết định có loại khỏi instance detection hay không.

## GQA

GQA scene graph thường biểu diễn object dưới dạng dictionary keyed by object ID, còn relation nằm trong mỗi node dưới `relations` hoặc `relationships`. Converter giữ directed edge theo node source. Cần kiểm tra các relation như `left`, `right`, `in`, `inside`, `around` trước khi canonicalize vì hướng và synonym có thể khác giữa release.

## Canonicalization policy

1. Normalize lowercase, whitespace và underscore.
2. Tìm exact canonical ID.
3. Tìm alias đã được review trong ontology.
4. Nếu không match, loại khỏi supervised target nhưng ghi vào report `unseen_object_labels`/`unseen_predicates`.
5. Không dùng embedding nearest-neighbor để tự động canonicalize label trong production.

## Recommended data audit

```bash
python scripts/inspect_dataset.py \
  --jsonl data/vg_unified.jsonl data/oi_train_unified.jsonl \
  --ontology ontology/ontology_v1.json \
  --output reports/data_audit.json
```

Trước khi train, cần kiểm tra: số object/relation bị loại, tỷ lệ `partial/exhaustive`, số label không có ancestor, relation có subject/object ID không tồn tại và mức độ mất cân bằng theo group.
