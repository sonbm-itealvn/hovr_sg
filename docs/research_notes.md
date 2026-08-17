# Ghi chú nghiên cứu OV-SGG

## Nguồn và dữ liệu

- Visual Genome v1.4 cung cấp objects, relationships, object/relationship aliases và synsets; trang chính thức: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- Open Images V6: khoảng 9M ảnh; 16M bounding boxes trên 1.9M ảnh, 600 lớp object; 3.3M visual-relation annotations, 1,466 triplets, 288 lớp object/attribute liên quan; có hierarchy lớp 600 boxable classes, human-verified positive/negative image-level labels và 61 zero-shot triplets trong validation/test. Nguồn: https://storage.googleapis.com/openimages/web/factsfigures.html
- GQA: 22M câu hỏi; mỗi ảnh có scene graph gồm objects, attributes, relations, là phiên bản làm sạch dựa trên Visual Genome. Nguồn: https://cs.stanford.edu/people/dorarad/gqa/about.html

## Baseline/ý tưởng liên quan

- He et al., ECCV 2022, Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning: OV-SGG sử dụng VLM/prompt để mở rộng label space, đánh giá trên Visual Genome, GQA, Open Images; setup base/novel.
- Salzmann et al., ECCV 2024, Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection: decoder-free Transformer, liên kết image tokens với object/relation và dự đoán quan hệ mở end-to-end.
- Li et al., 2024, From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models (PGSG): image-to-sequence generative VLM; scene graph prompt với token [ENT]/[REL], entity grounding bằng cross-attention để dự đoán boxes, category conversion từ vocabulary score sang target category space, loss LM + position regression. Nguồn HTML: https://arxiv.org/html/2404.00906v2

## Hàm ý cho thiết kế

- Nên tách rõ open-vocabulary object classification và open-vocabulary predicate/relation classification; không dùng chỉ một centroid cho category gốc vì dễ làm 'person' nuốt 'man/woman/girl' hoặc làm lẫn 'table/chair'.
- Cần biểu diễn phân cấp: leaf label (man, woman, girl, cup, bottle, table, chair) và semantic group/superclass (person, container/drinkware, furniture), với loss giữ khoảng cách giữa nhóm và margin giữa các nhóm.
- Dữ liệu cần canonicalization bằng aliases/synsets, taxonomy có quan hệ is-a và sibling-negative; các cặp không có relation dương nên được dùng hard negative chỉ khi annotation đủ exhaustive.
- Với mô hình end-to-end thực dụng, nên dùng detector dạng DETR/MaskDINO hoặc backbone VLM frozen/LoRA, sau đó relation decoder trên object queries; có thể thêm nhánh generative VLM làm teacher/auxiliary semantic supervision.
- Đánh giá phải báo cáo object AP/base/novel/harmonic mean và SGG Recall@K/mR@K/zero-shot triplet; thêm group-consistency, sibling-confusion và calibration để đo đúng mục tiêu phân cấp.

## Các trích dẫn cần dùng trong tài liệu cuối

[1] Visual Genome official API/download page: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
[2] Open Images V6 official facts and figures: https://storage.googleapis.com/openimages/web/factsfigures.html
[3] GQA official dataset page: https://cs.stanford.edu/people/dorarad/gqa/about.html
[4] Li et al. 2024 PGSG HTML: https://arxiv.org/html/2404.00906v2
[5] He et al. 2022 ECCV PDF: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880055.pdf
[6] Salzmann et al. 2024 ECCV PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11211.pdf

## Cảnh báo dữ liệu

- Visual Genome có nhiễu và long-tail; GQA scene graphs sạch hơn nhưng thiên về ảnh/quan hệ phục vụ VQA.
- Open Images có quan hệ đa dạng và nhãn negative hữu ích, nhưng ontology object/relation không đồng nhất trực tiếp với VG; cần ontology mapping và split theo image để tránh leakage.
- Text/VLM teacher có thể sinh quan hệ hợp lý ngôn ngữ nhưng sai thị giác; chỉ dùng pseudo-label với confidence/consistency filtering, không coi mọi caption/teacher output là ground truth.

## Ghi chú hình người dùng

- Hình minh họa thể hiện hai mục tiêu cần tách: (i) latent space chưa gom nhóm, (ii) latent space có các cụm semantic rõ hơn; (iii) khoảng cách cùng nhóm nhỏ hơn khác nhóm; (iv) loss smooth và KL/structure term cùng giảm. Không đọc lại file ảnh theo yêu cầu của người dùng.

## Ngày ghi chú

2026-08-17 (GMT+7)

## Author

Manus AI

## References

[1]: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html "Visual Genome Dataset"
[2]: https://storage.googleapis.com/openimages/web/factsfigures.html "Open Images V6 facts and figures"
[3]: https://cs.stanford.edu/people/dorarad/gqa/about.html "GQA Dataset"
[4]: https://arxiv.org/html/2404.00906v2 "From Pixels to Graphs"
[5]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880055.pdf "Towards Open-vocabulary Scene Graph Generation"
[6]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11211.pdf "Scene-Graph ViT"

---

Các ghi chú trên là bản tóm lược làm việc, không phải deliverable cuối.

## Bổ sung: fully open vocabulary và hierarchical prompt

- OvSGTR / Expanding Scene Graph Boundaries phân biệt bốn setting: Closed-set SGG, OvD-SGG (object detection mở), OvR-SGG (relation mở), và OvD+R-SGG (cả object và relation mở). Framework là end-to-end Transformer học visual-concept alignment cho nodes và edges; relation-involved setting dùng relation-aware pretraining trên image-caption và knowledge distillation để giữ alignment. Nguồn: https://arxiv.org/html/2311.10988v2
- RAHP (AAAI 2025) dùng entity clustering để tạo super entities; supplementary mô tả dùng WordNet/POS trước, encode cluster bằng VLM text encoder rồi K-means, sau đó dùng LLM đặt tên superclass. Ví dụ có male/female/children, seating furniture, table, container, beverage, household item. RAHP cũng sinh region-aware prompts mô tả parts của subject/object trong quan hệ. Nguồn: https://arxiv.org/html/2412.19021v1
- Hướng cải tiến cho đề bài: không nên để K-means thuần túy quyết định ontology; cần taxonomy có kiểm duyệt và sibling-negative. K-means/WordNet/LLM nên chỉ khởi tạo candidate superclasses, còn ontology cuối phải được xác nhận bằng label policy và hard-negative matrix.

[7]: https://arxiv.org/html/2311.10988v2 "Fully open-vocabulary SGG via visual-concept alignment"
[8]: https://arxiv.org/html/2412.19021v1 "Relation-aware hierarchical prompt"
