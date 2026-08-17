# Validation report

## Đã kiểm tra trong sandbox

| Check | Trạng thái | Ghi chú |
|---|---|---|
| `python3 -m compileall -q .` | PASS | Toàn bộ source compile được |
| Unified sample audit | PASS | Đã thống kê object/group/relation và annotation scope |
| Visual Genome fixture converter | PASS | `male → man`, `coffee cup → cup`, `grasping → holding` |
| Sample images generation | PASS | Ba JPEG 256×256 được tạo trong `data_sample/` |

## Chưa chạy trong sandbox

| Check | Trạng thái | Lý do |
|---|---|---|
| PyTorch forward smoke test | BLOCKED | Sandbox hiện chưa cài `torch` |
| `pytest` unit tests | BLOCKED | Sandbox hiện chưa cài `pytest` |
| Real dataset conversion | NOT RUN | Chưa có file annotation/dataset của người dùng |
| Full training | NOT RUN | Cần PyTorch + CUDA/CPU backend và dữ liệu thật |

Sau khi cài dependencies, chạy:

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/train.py --config configs/hovr_sg.yaml \
  --train-jsonl data_sample/train.jsonl --val-jsonl data_sample/val.jsonl \
  --ontology ontology/ontology_v1.json --image-root data_sample \
  --output-dir runs/smoke --epochs 1
```
