# HOVR-SG: Mô hình end-to-end Open-Vocabulary Object–Relationship Detection với latent space phân cấp

**Tác giả:** Manus AI  
**Ngôn ngữ:** Tiếng Việt  
**Phạm vi:** Thiết kế nghiên cứu và blueprint triển khai, chưa phải mã nguồn chạy sẵn.

## 1. Tóm tắt đề xuất

Bài toán phù hợp nhất với ý tưởng của bạn không chỉ là object detection hay visual relationship detection riêng lẻ, mà là **fully open-vocabulary scene graph detection**: từ ảnh đầu vào, mô hình phải phát hiện các node đối tượng, định vị bounding box, gán nhãn vật thể mở, sau đó dự đoán các cạnh có hướng dạng `(subject, predicate, object)`. Setting này cần mở đồng thời không gian node và edge; trong tài liệu gần đây nó thường được phân biệt thành object-open, relation-open và fully open object-plus-relation SGG [7].

Tôi đề xuất kiến trúc **HOVR-SG (Hierarchical Open-Vocabulary Relational Scene Graph)**. Mô hình có bốn nguyên tắc chính:

1. **Detector VLM-conditioned kiểu DETR** tạo object queries và embedding vùng ảnh trong cùng latent space với text prototypes.
2. Mỗi vật thể được biểu diễn ở **hai mức đồng thời**: nhãn lá cụ thể như `man`, `woman`, `girl`, `cup`, `bottle`, `table`, `chair`; và nhãn nhóm như `person`, `drinkware/container`, `furniture`. Nhánh nhóm giúp chuyển giao ngữ nghĩa, còn nhánh lá giữ ranh giới giữa các sibling.
3. Relation decoder không dựa chỉ trên cặp nhãn dự đoán. Nó kết hợp embedding của chủ thể, đối tượng, union-region, hình học tương đối và đặc trưng tương tác để dự đoán predicate trong open vocabulary.
4. Loss được tách thành **alignment, hierarchy-consistency và sibling-separation**. Vì vậy, `man` gần `woman` ở group space nhưng không bị ép trùng trong leaf space; `table` và `chair` không bị gom vào cùng một cụm chỉ vì đều là đồ nội thất.

Điểm mấu chốt là không biến các nhóm semantic thành nhãn thay thế cho nhãn lá. Nhóm chỉ là một **tầng giám sát và prior**. Khi inference, mô hình vẫn phải xuất nhãn lá cụ thể và có thể xuất thêm superclass nếu ứng dụng cần graph phân cấp.

## 2. Phân định đúng bài toán

Với một ảnh `I`, đầu ra mong muốn là một scene graph:

\[
G=(V,E), \qquad V=\{v_i=(b_i,c_i,g_i,z_i)\}_{i=1}^{N},
\]

trong đó `b_i` là bounding box, `c_i` là leaf object label, `g_i` là superclass hoặc nhóm ngữ nghĩa, và `z_i` là object latent embedding. Mỗi cạnh có hướng là:

\[
e_{ij}=(i,r_{ij},j,s_{ij}),
\]

trong đó `r_ij` là predicate mở, còn `s_ij` là confidence. Ví dụ mô hình cần có khả năng biểu diễn đồng thời:

```text
(man, holding, cup)
(girl, holding, bottle)
(person, sitting-on, chair)
(dog, inside, car)
```

`person holding cup` không nên là một lớp nguyên tử duy nhất. Nó nên được factorize thành object subject `person/man/girl`, predicate `holding`, và object `cup/bottle`. Cách factorization này cho phép mô hình học compositional generalization: nếu đã thấy `man holding cup` và `girl holding bottle`, nó vẫn có cơ sở để suy luận `woman holding bottle` khi hình ảnh và text prototype phù hợp.

Cần phân biệt bốn setting đánh giá. Đây là điểm quan trọng vì nhiều kết quả được gọi là open vocabulary thực chất chỉ mở object hoặc chỉ mở predicate.

| Setting | Object vocabulary | Relation vocabulary | Ý nghĩa |
|---|---:|---:|---|
| Closed SGG | Seen | Seen | Baseline kiểm tra khả năng học scene graph thông thường |
| OvD-SGG | Novel object có thể xuất hiện | Seen | Kiểm tra open-vocabulary object detection |
| OvR-SGG | Seen | Novel predicate | Kiểm tra open-vocabulary relation detection |
| OvD+R-SGG | Novel object | Novel predicate | Mục tiêu đầy đủ của đề xuất |

Trong thực nghiệm chính, nên tách thêm **novel composition**: từng object và relation đều đã thấy riêng lẻ trong training nhưng tổ hợp triplet chưa từng xuất hiện. Đây là bài test trực tiếp cho mục tiêu của bạn, tốt hơn việc chỉ chia ngẫu nhiên các triplet.

## 3. Kiến trúc HOVR-SG đề xuất

### 3.1. Sơ đồ tổng thể

```text
Image I
  │
  ├── VLM image encoder + multi-scale neck ───────────────┐
  │                                                      │
  └── Text encoder: leaf labels, superclasses, predicates │
                         │                                │
               Vision-language feature enhancer          │
                         │                                │
        Language-guided DETR object query selection       │
                         │                                │
        Object decoder: N queries                         │
          ├── box/objectness                              │
          ├── leaf object embedding z_leaf                │
          ├── superclass embedding z_group               │
          └── object distributions p(leaf), p(group)     │
                         │                                │
       matched object slots + top-M candidate objects    │
                         │                                │
       Sparse relation proposal module                   │
          ├── subject/object object embeddings            │
          ├── union-region feature                        │
          ├── relative geometry                           │
          └── interaction/context attention               │
                         │                                │
       Relation decoder                                  │
          ├── relationness / no-relation                 │
          ├── predicate latent z_rel                     │
          └── open-vocabulary predicate scores           │
                         │                                │
           Scene graph + optional hierarchy graph        │
```

### 3.2. Backbone và language interface

Có hai lựa chọn triển khai.

**Lựa chọn nghiên cứu khuyến nghị:** khởi tạo từ một detector VLM-conditioned kiểu Grounding DINO, rồi thay classification head bằng hierarchical prototype head và bổ sung relation decoder. Grounding DINO đưa language vào ba vị trí: feature enhancer, language-guided query selection và cross-modality decoder; nó dùng grounded pre-training trên detection, grounding và caption data [9]. Đây là lựa chọn tốt nếu mục tiêu là tận dụng khả năng grounding sẵn có nhưng vẫn huấn luyện relation head end-to-end.

**Lựa chọn gọn và dễ kiểm soát:** dùng ViT-L/Swin-L image encoder đã căn chỉnh với CLIP hoặc SigLIP, text encoder tương ứng được freeze, rồi xây dựng DETR decoder từ đầu. Lựa chọn này phù hợp khi cần kiểm soát hoàn toàn loss và taxonomy, nhưng chất lượng box open-vocabulary ban đầu thường cần nhiều dữ liệu grounding hơn.

Trong cả hai trường hợp, text encoder sinh prototype cho ba loại token:

\[
T^{leaf}=\{t_c:c\in\mathcal C_{leaf}\},\quad
T^{group}=\{t_g:g\in\mathcal C_{group}\},\quad
T^{rel}=\{t_r:r\in\mathcal R\}.
\]

Mỗi label nên được encode với nhiều template, sau đó lấy trung bình có chuẩn hóa:

```text
"a photo of a {label}"
"a cropped image of a {label}"
"an object that is a {label}"
"a {label} in a visual scene"
```

Với quan hệ, template phải giữ hướng:

```text
"a person is {predicate} a cup"
"subject {predicate} object"
"the subject has the relation {predicate} to the object"
```

Không nên dùng một embedding duy nhất cho những từ đa nghĩa như `on`, `in`, `by`, `with`. Nên giữ predicate canonical và danh sách paraphrase, ví dụ `holding` có thể có `grasping`, `carrying-in-hand`, nhưng không gộp bừa `carrying` nếu annotation policy coi đó là một quan hệ khác.

### 3.3. Object encoder và hai không gian latent

Object query `q_i` sau decoder được đưa qua hai projection head:

\[
z_i^{leaf}=\mathrm{norm}(W_{leaf}q_i), \qquad
z_i^{group}=\mathrm{norm}(W_{group}q_i).
\]

Hai head này có chủ đích **không dùng chung hoàn toàn trọng số**. Nếu chỉ dùng một embedding và kéo tất cả các nhãn cùng superclass về một centroid, `man`, `woman`, `girl` có thể bị collapse. Ngược lại, nếu chỉ tối ưu nhãn lá, mô hình thiếu cấu trúc để chuyển giao từ `person` sang các subtype chưa thấy.

Điểm phân loại cosine với nhiệt độ học được:

\[
s^{leaf}_{ic}=\frac{z_i^{leaf}\cdot t_c^{leaf}}{\tau_{leaf}},
\qquad
s^{group}_{ig}=\frac{z_i^{group}\cdot t_g^{group}}{\tau_{group}}.
\]

Nhãn lá novel không cần một fully-connected classifier mới; chỉ cần thêm text prototype vào vocabulary ở inference. Tuy nhiên, khả năng zero-shot của prototype phụ thuộc mạnh vào chất lượng region–text alignment và ontology normalization.

### 3.4. Hierarchical object head

Mỗi leaf label phải có một hoặc nhiều ancestor group. Tối thiểu nên bắt đầu bằng một taxonomy hai tầng:

| Leaf labels | Superclass |
|---|---|
| `man`, `woman`, `boy`, `girl`, `child` | `person` |
| `cup`, `bottle`, `mug`, `glass` | `drinkware/container` |
| `chair`, `sofa`, `bench` | `seating furniture` |
| `table`, `desk`, `dining table` | `table furniture` |
| `car`, `bus`, `truck`, `motorcycle` | `ground vehicle` |
| `dog`, `cat`, `horse` | `animal` hoặc các nhóm con phù hợp |

`chair` và `table` cố ý được đặt ở **hai group khác nhau** nếu đó là label policy của miền dữ liệu. Không nên suy ra group chỉ bằng proximity của text embedding. Các group như `furniture` có thể là ancestor chung ở tầng cao hơn, nhưng phải tồn tại các node con `seating furniture` và `table furniture`.

Với mỗi object, mô hình dự đoán cả leaf và group. Nếu một leaf có nhiều ancestor, dùng multi-label BCE cho toàn bộ ancestor; nếu taxonomy là cây đơn, dùng softmax ở từng tầng. Với bản đầu tiên, nên dùng DAG/multi-label vì một `mug` có thể vừa thuộc `drinkware`, `container`, `household item` tùy ontology.

Một score leaf có điều kiện có thể dùng khi inference:

\[
S(c\mid z_i)=S_{leaf}(c\mid z_i)+
\lambda_g\log\big(P(g(c)\mid z_i)+\epsilon\big).
\]

Thành phần group chỉ làm prior, không được phép thay thế leaf score. Để tránh superclass quá phổ biến lấn át subtype, giới hạn `λ_g` nhỏ và calibrate theo từng group trên validation.

### 3.5. Sparse relation proposal

Nếu có `M` object, xét toàn bộ cặp có hướng sẽ có `M(M-1)` cặp. Với `M=100`, con số đã là 9.900; do đó cần một relation proposal module.

Với mỗi cặp `(i,j)`, tạo đặc trưng:

\[
h_{ij}=\mathrm{MLP}([
 z_i^{obj},z_j^{obj},z_i^{obj}\odot z_j^{obj},
 \phi(\Delta x,\Delta y,\Delta w,\Delta h,IoU),
 u_{ij},
 c_{ij}]).
\]

Trong đó `u_ij` là union-region feature lấy từ feature map của vùng bao phủ hai box; `φ` là positional encoding của hình học tương đối; `c_ij` là context feature do một vài layer cross-attention giữa query subject/object và image tokens tạo ra.

Relationness head dự đoán:

\[
p_{ij}^{rel}=\sigma(w_{rel}^{T}h_{ij}),
\]

sau đó chỉ giữ top `K_pair` cặp, ví dụ 256 hoặc 512. Trong training, không chỉ giữ cặp có relation dương; phải bổ sung hard negatives là các cặp gần nhau hoặc có cùng group nhưng không có relation, nếu annotation của dataset đủ exhaustive.

### 3.6. Relation decoder và predicate open vocabulary

Relation decoder dùng một hoặc hai tầng Transformer trên `h_ij`. Đầu ra:

\[
z_{ij}^{rel}=\mathrm{norm}(W_{rel}h_{ij}),
\qquad
s_{ijr}=\frac{z_{ij}^{rel}\cdot t_r^{rel}}{\tau_{rel}}.
\]

Có thể thêm relation hierarchy:

| Predicate lá | Relation group |
|---|---|
| `holding`, `grasping`, `carrying` | `human-object action/contact` |
| `inside`, `contains`, `covered-by` | `containment` |
| `on`, `under`, `above`, `below` | `vertical/spatial` |
| `left-of`, `right-of`, `in-front-of` | `directional` |
| `near`, `far`, `overlapping` | `topological/distance` |

Quan hệ có hướng nên được xử lý như cặp có thứ tự. Ví dụ `person holds cup` không tương đương `cup holds person`; với các symmetric predicate như `near`, annotation policy nên đặt canonical order hoặc cho phép cả hai hướng nhưng metric phải xử lý nhất quán.

### 3.7. Output và inference

Một output triplet chỉ hợp lệ khi đồng thời thỏa:

\[
Score(i,r,j)=
P_{obj}(i)P_{obj}(j)P_{rel}(i,j)P_{pred}(r\mid i,j).
\]

Không nên nhân trực tiếp tất cả xác suất nếu calibration chưa tốt vì sẽ làm triplet dài bị tụt điểm. Có thể dùng log-score có nhiệt độ riêng:

\[
\log S=\alpha\log P_i+\alpha\log P_j+
\beta\log P_{rel}+\gamma s_{ijr}.
\]

Sau đó áp dụng box NMS hoặc query deduplication, pair-level suppression và top-K scene graph output. Với open vocabulary, user có thể truyền vocabulary động; model sẽ encode text label mới rồi chấm điểm mà không thay đổi classifier matrix.

## 4. Ontology và annotation policy

### 4.1. Vì sao không dùng clustering thuần túy

Clustering trên text embedding là cách tốt để đề xuất group, nhưng không đủ để tạo ground truth. Hai label có thể gần nhau về ngôn ngữ nhưng khác nhau theo nghiệp vụ; ngược lại, hai label cùng superclass có thể khác xa vì khác hình dạng. Vì vậy, dùng WordNet, text encoder và K-means chỉ để khởi tạo candidate hierarchy là hợp lý; tài liệu RAHP cũng mô tả quy trình tạo super entity bằng lexical structure, VLM text embedding và K-means [8]. Nhưng ontology cuối phải qua label policy do con người kiểm duyệt.

Cần lưu ba loại quan hệ giữa label:

| Quan hệ ontology | Ví dụ | Dùng trong loss |
|---|---|---|
| `is-a` / ancestor | `man → person` | Group positive, hierarchy consistency |
| `alias/synonym` | `automobile ↔ car` nếu cùng policy | Positive augmentation, không phải negative |
| `sibling-negative` | `chair ↔ table`, `cup ↔ bottle` | Hard negative, angular margin |

Một label không chắc chắn hoặc không được annotation phải ghi là `unknown`, không được tự động ghi `negative`. Đây là điều đặc biệt quan trọng với Visual Genome vì annotation không exhaustive như Open Images trong mọi trường hợp.

### 4.2. Schema annotation đề xuất

Mỗi instance nên có bản ghi:

```json
{
  "image_id": "...",
  "instance_id": 17,
  "bbox": [x1, y1, x2, y2],
  "leaf_label": "cup",
  "super_labels": ["drinkware", "container"],
  "aliases": ["coffee cup"],
  "visibility": "visible|occluded|truncated",
  "label_confidence": 1.0,
  "source": "human|verified|teacher|weak"
}
```

Mỗi relation nên có:

```json
{
  "subject_id": 5,
  "object_id": 17,
  "predicate": "holding",
  "predicate_group": "contact_action",
  "directional": true,
  "visibility": "visible|ambiguous",
  "confidence": 1.0
}
```

Nếu xây dữ liệu mới, nên annotate leaf label **và** group label ngay tại thời điểm gán leaf. Không để hệ thống tự suy luận group bằng nearest text label sau khi training, vì khi đó label noise sẽ lan sang group loss và làm hỏng latent structure.

### 4.3. Chuẩn hóa nhiều dataset

Visual Genome cung cấp objects, relationships, aliases và synsets tải được từ trang dữ liệu chính thức [1]. Open Images V6 cung cấp đồng thời bounding boxes, visual relationships, hierarchy lớp, verified positive/negative image-level labels và quan hệ zero-shot; trang chính thức ghi nhận 600 lớp boxable, 3,3 triệu relation annotations và 1.466 triplets [2]. GQA gắn mỗi ảnh với scene graph gồm objects, attributes và relations, là phiên bản làm sạch dựa trên Visual Genome [3].

Nên xây một ontology registry thay vì nối thẳng label string:

| Trường | Mục đích |
|---|---|
| `dataset_label` | Tên gốc của dataset |
| `canonical_label` | Tên chuẩn dùng trong model |
| `synset_id` | Liên kết WordNet/ontology nếu có |
| `level` | Leaf, intermediate, root |
| `parent_ids` | Ancestor trong DAG |
| `aliases` | Các cách gọi tương đương |
| `sibling_ids` | Hard-negative candidates |
| `relation_policy` | Mô tả label có hướng/symmetric |
| `annotation_scope` | Exhaustive, partial hoặc unknown |

Các label như `man`, `woman`, `boy`, `girl` có thể map vào `person` nhưng vẫn phải giữ leaf riêng. `cup` và `bottle` có thể cùng group `container/drinkware`, nhưng không được dùng group để thay cho leaf. `table` và `chair` nên được giữ sibling-negative trong ontology con nếu yêu cầu ứng dụng coi đây là hai nhóm khác nhau.

## 5. Dataset recommendation và split

### 5.1. Bộ dữ liệu nên dùng

| Vai trò | Dataset | Lý do | Hạn chế và cách xử lý |
|---|---|---|---|
| Object open-vocabulary/grounding pretrain | Open Images V6, thêm các grounding/caption corpus được cấp phép | Có box, hierarchy, label positive/negative và visual relations; quy mô lớn [2] | Ontology rộng nhưng relation và box không phủ đồng đều; dùng annotation scope mask |
| Relation chính | Open Images V6 | Visual relationship annotations có nhiều triplet và quan hệ zero-shot; negative pair đáng tin hơn khi annotation exhaustive [2] | Relation distribution long-tail; cần class-balanced sampler |
| Lexical diversity/object–relation pretrain | Visual Genome | Có object, relationship, aliases, synsets và region descriptions [1] | Nhiễu, long-tail, không phải mọi cặp không được annotate đều là negative |
| Scene reasoning/clean graph auxiliary | GQA | Scene graph sạch hơn VG và có attributes/relations [3] | Thiên về VQA; dùng như auxiliary graph/text supervision, không nhất thiết làm benchmark chính |
| Custom domain | Dataset nội bộ có leaf+group+relation annotation | Cần thiết nếu miền triển khai khác ảnh đời thường | Phải ghi rõ license, annotation scope và unseen split |

Tôi khuyến nghị pipeline dữ liệu ba tầng. Tầng một học image–text/region–text grounding từ dữ liệu lớn; tầng hai học object detection với label hierarchy; tầng ba học relation decoder trên Open Images + Visual Genome + một phần GQA. Nếu tài nguyên hạn chế, có thể bỏ GQA ở phiên bản đầu và dùng Open Images làm relation chính, Visual Genome làm semantic augmentation.

### 5.2. Split base/novel không gây leakage

Không chia ngẫu nhiên theo triplet vì cùng object và predicate có thể xuất hiện trong nhiều tổ hợp, khiến zero-shot bị đánh giá quá dễ. Nên công bố bốn split:

| Split | Training | Test | Mục tiêu |
|---|---|---|---|
| `S/S` | object seen, relation seen | object seen, relation seen | Closed-set sanity check |
| `N/S` | novel object không xuất hiện trong object labels của train | novel object, relation seen | Open object |
| `S/N` | novel predicate không xuất hiện trong relation labels của train | object seen, novel predicate | Open relation |
| `N/N` | novel object + novel predicate bị giữ lại | cả hai novel | Fully open |

Thêm `Novel-Composition`: cấm một số triplet `(subject group/leaf, predicate, object group/leaf)` trong train nhưng vẫn cho phép từng thành phần xuất hiện riêng lẻ. Với object classes long-tail, nên lọc novel theo ngưỡng số instance tối thiểu để test không bị chi phối bởi label cực hiếm.

Một protocol thực tế là giữ 80% leaf classes phổ biến làm base và 20% làm novel, nhưng tỷ lệ này chỉ là đề xuất; cần báo cáo đầy đủ danh sách class, số instance và số triplet. Novel class phải được hold out khỏi detection/relation labels của train, kể cả qua synonym đã canonicalize. Caption pretraining chứa từ novel cần được tách rõ thành **strict zero-shot** hoặc **language-assisted zero-shot**; không trộn hai kết quả.

## 6. Loss function

Tổng loss đề xuất:

\[
\mathcal L=
\lambda_{det}\mathcal L_{det}+
\lambda_{align}\mathcal L_{align}+
\lambda_{hier}\mathcal L_{hier}+
\lambda_{sib}\mathcal L_{sib}+
\lambda_{rel}\mathcal L_{rel}+
\lambda_{dist}\mathcal L_{dist}.
\]

### 6.1. Object detection loss

Sau Hungarian matching giữa object queries và ground-truth instances:

\[
\mathcal L_{det}=\lambda_{cls}\mathcal L_{leaf}+
\lambda_{grp}\mathcal L_{group}+
\lambda_{box}\|b-\hat b\|_1+
\lambda_{giou}(1-GIoU(b,\hat b))+
\lambda_{noobj}\mathcal L_{noobj}.
\]

`L_leaf` nên là sigmoid focal/BCE trên open-vocabulary cosine logits hoặc cross-entropy trên candidate labels của batch. `L_group` là multi-label BCE nếu một leaf có nhiều ancestor. Với background/no-object, dùng DETR-style matching và focal weighting để tránh số query rỗng áp đảo.

### 6.2. Region–text alignment

Với object dương `i,c`, dùng InfoNCE:

\[
\mathcal L_{align}^{obj}=-
\log\frac{\exp(sim(z_i^{leaf},t_c)/\tau)}
{\sum_{c'\in\mathcal C_{batch}}\exp(sim(z_i^{leaf},t_{c'})/\tau)}.
\]

Group branch dùng positive là ancestor và negative là group khác. Có thể distill từ VLM teacher ở mức region–text, nhưng teacher chỉ cung cấp soft target khi box/label có confidence; không dùng teacher để phủ định human label.

### 6.3. Hierarchy consistency

Có hai điều kiện cần giữ.

Thứ nhất, leaf dương phải có group dương:

\[
\mathcal L_{anc}=\mathrm{BCE}(p^{group}_{i,g},1),\quad g\in Ancestor(c).
\]

Thứ hai, xác suất leaf không nên vượt quá ancestor của nó. Có thể dùng penalty:

\[
\mathcal L_{mono}=\sum_{c,g(c)}
\max(0,p^{leaf}_{i,c}-p^{group}_{i,g(c)}).
\]

Nếu taxonomy có nhiều leaf cùng group, group probability không nhất thiết bằng tổng xác suất leaf vì có thể còn label chưa biết. Dùng soft aggregation ổn định hơn:

\[
\tilde p_g=1-\prod_{c\in Children(g)}(1-p_c),
\]

sau đó phạt sai lệch giữa `p_g` và `tilde p_g` bằng KL hoặc MSE với trọng số nhỏ.

### 6.4. Giữ gần trong group nhưng không collapse sibling

Đây là loss trực tiếp cho mục tiêu trong hình minh họa.

**Group compactness:** các object cùng group kéo gần trong `z_group`:

\[
\mathcal L_{compact}=\sum_{(i,j):g_i=g_j}
\max(0,sim(z_i^{group},z_j^{group})-m_{within})
\]

thực tế nên viết theo dạng kéo similarity lên, ví dụ `max(0, m_pos - sim(...))`.

**Inter-group separation:** các group khác nhau có margin:

\[
\mathcal L_{group\text{-}margin}=
\max(0,m_g-sim(z_i^{group},z_j^{group}))
\quad g_i\neq g_j.
\]

**Sibling separation:** ở leaf space, sibling không được collapse:

\[
\mathcal L_{sib}=
\max\left(0,m_s-sim(z_i^{leaf},t_{c_i})+sim(z_i^{leaf},t_{c'})\right),
\]

với `c'` là sibling hard negative như `table` khi ground truth là `chair`, hoặc `bottle` khi ground truth là `cup`. `m_s` nên lớn hơn margin trong group space. Như vậy cùng group được compact ở `z_group`, nhưng leaf prototypes vẫn có angular margin.

**Decorrelation:** tránh hai head học cùng một latent bằng penalty nhẹ trên covariance giữa `z_leaf` và `z_group`. Không nên dùng trọng số lớn vì hai head vẫn cần chia sẻ thông tin hình ảnh.

### 6.5. Relation loss

Với cặp dương và negative:

\[
\mathcal L_{rel}=
\lambda_{ness}\mathcal L_{relationness}+
\lambda_{pred}\mathcal L_{predicate}+
\lambda_{hier-r}\mathcal L_{relation-hierarchy}.
\]

`L_relationness` là focal/BCE cho có hoặc không có edge. `L_pred` là open-vocabulary contrastive loss giữa `z_ij^rel` và predicate text prototype. Nếu relation label có paraphrase, tất cả paraphrase là positive; sibling predicate khác nghĩa là hard negative.

Relation branch nên nhận soft object distribution thay vì chỉ nhận argmax leaf:

\[
\bar z_i=\sum_c p(c\mid i)\,e_c + W_z z_i^{leaf}.
\]

Cách này giúp `person` truyền thông tin sang `man/woman/girl` mà không làm relation decoder phụ thuộc vào một nhãn sai duy nhất.

### 6.6. Distillation và pseudo-label

Có thể dùng VLM/generative VLM teacher để tạo candidate label và region description, tương tự hướng image-to-graph và category conversion của PGSG [4]. Tuy nhiên, teacher text có thể sinh quan hệ hợp lý nhưng không hiện diện trong ảnh. Chỉ giữ pseudo-label khi thỏa ít nhất hai điều kiện: teacher score cao, và box/region hoặc relation ổn định qua hai prompt/paraphrase hoặc hai augmentation. Pseudo-label phải mang `source=teacher`, `confidence`, và trọng số loss thấp hơn human label.

## 7. Quy trình training đề xuất

### Giai đoạn 0: Ontology và data audit

Trước khi train, xây label registry, kiểm tra alias, thống nhất hướng predicate, đánh dấu annotation scope và sinh hard-negative matrix. Đây là giai đoạn bắt buộc; nếu bỏ qua, group loss sẽ học nhãn sai và kết quả clustering đẹp nhưng không có ý nghĩa.

### Giai đoạn 1: Warm-up detector

Freeze image/text backbone, chỉ train neck, object decoder, box head và object projection heads. Dùng Open Images/VG object boxes. Mục tiêu là detector học box và region–text alignment mà không bị relation branch gây nhiễu. Có thể dùng 10–20 epochs tùy quy mô; validation theo open-vocabulary AP và group/leaf accuracy.

### Giai đoạn 2: Hierarchical alignment

Mở group loss, leaf contrastive loss, ancestor consistency và sibling hard-negative. Sampling phải cân bằng theo group trước, sau đó theo leaf. Nếu sample ngẫu nhiên toàn dataset, group `person` hoặc `vehicle` có thể áp đảo và khiến hình ảnh latent space trông như nhóm lớn nhưng các nhóm nhỏ bị mất.

### Giai đoạn 3: Relation decoder

Freeze detector trong vài epoch đầu, train relation proposal và relation decoder trên các object matching chính xác. Dùng ground-truth pair boxes hoặc matched predicted slots có teacher forcing ở giai đoạn đầu, sau đó chuyển dần sang predicted boxes để tránh train–test mismatch.

### Giai đoạn 4: Joint end-to-end fine-tuning

Mở LoRA/adapters hoặc unfreeze các block cuối của vision backbone và cross-modal neck. Train joint với learning rate backbone nhỏ hơn head từ 5–10 lần. Dùng curriculum: relation pair sampling từ easy positive trước, sau đó tăng hard negatives gần nhau và sibling-confusable.

### Giai đoạn 5: Open-vocabulary adaptation

Thêm image-caption/grounding batches và pseudo-label có lọc. Giữ một phần batch có human SGG label ở mỗi bước để tránh catastrophic forgetting. Knowledge distillation nên áp dụng lên region–text logits và relation–text logits, không chỉ lên output label cuối.

Một cấu hình khởi đầu hợp lý cho prototype là:

| Thành phần | Giá trị khởi đầu |
|---|---:|
| Image size | short side 800, max long side 1333 |
| Object queries | 300; tăng lên 900 nếu dùng Grounding DINO style |
| Objects giữ cho relation | top `M=100` |
| Relation pairs | top `K_pair=256–512` |
| Transformer relation layers | 2–4 |
| Head learning rate | `1e-4` |
| Backbone learning rate | `1e-5` hoặc freeze + LoRA |
| Optimizer | AdamW, weight decay `1e-4` |
| Warm-up | 1–2% tổng bước |
| Gradient clipping | 0.1–1.0 |
| Temperature | học được, khởi tạo `0.07` cho contrastive |
| Loss schedule | tăng dần `λ_hier`, `λ_sib`, `λ_rel` sau warm-up |

Đây là initial hyperparameter chứ không phải giá trị tối ưu cố định. Cần tune bằng validation split giữ nguyên danh sách novel classes.

### Pseudo-code huấn luyện

```python
for batch in loader:
    image, gt_objects, gt_edges, ontology = batch

    image_tokens = vision_encoder(image)
    text_leaf, text_group, text_rel = encode_prototypes(ontology)
    fused = vl_feature_enhancer(image_tokens, text_leaf, text_group, text_rel)

    slots = object_decoder(fused, language_guided_queries=True)
    boxes = box_head(slots)
    z_leaf = normalize(leaf_proj(slots))
    z_group = normalize(group_proj(slots))

    matches = hungarian_match(boxes, z_leaf, gt_objects, ontology)
    det_loss = object_detection_loss(boxes, z_leaf, z_group,
                                     gt_objects, matches)
    hier_loss = hierarchy_consistency(z_leaf, z_group,
                                      gt_objects, ontology)
    sibling_loss = sibling_margin(z_leaf, gt_objects, ontology)
    align_loss = region_text_contrastive(z_leaf, z_group,
                                         gt_objects, text_leaf, text_group)

    selected = select_top_objects(slots, boxes, z_leaf, top_m=100)
    pair_features = build_sparse_pairs(selected, fused)
    pair_features = relation_context_encoder(pair_features)
    relness = relationness_head(pair_features)
    z_rel = normalize(relation_proj(pair_features))

    rel_loss = relation_loss(relness, z_rel, gt_edges,
                             matches, text_rel, ontology)

    loss = (lam_det * det_loss + lam_hier * hier_loss +
            lam_sib * sibling_loss + lam_align * align_loss +
            lam_rel * rel_loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 8. Đánh giá và ablation

### 8.1. Object detection

Báo cáo `AP`, `AP50`, `AP75` theo ba nhóm: base, novel và toàn bộ. Tính harmonic mean:

\[
H=\frac{2AP_{base}AP_{novel}}
{AP_{base}+AP_{novel}}.
\]

Bổ sung `group accuracy`, `leaf accuracy given correct box`, `hierarchical F1`, `open-set unknown detection` và calibration error. AP tổng thể không cho biết mô hình có đang đoán `person` thay cho `girl` hay không.

### 8.2. Relationship/scene graph

Dùng các metric SGG chuẩn như `R@K` và `mR@K`, nhưng phải tách `seen/novel` và `S/S, N/S, S/N, N/N`. Với long-tail, `mR@K` quan trọng hơn R@K vì R@K dễ bị chi phối bởi vài relation phổ biến.

Đề xuất thêm các metric riêng cho mục tiêu phân cấp:

| Metric | Ý nghĩa |
|---|---|
| `Zero-shot triplet recall` | Recall các triplet chưa từng xuất hiện trong train |
| `Group consistency` | Tỷ lệ leaf dự đoán có ancestor đúng |
| `Sibling confusion rate` | Tỷ lệ `table→chair`, `cup→bottle`, v.v. |
| `Cross-group false merge` | Tỷ lệ hai object khác group bị gom cùng group |
| `Novel calibration` | ECE hoặc Brier score trên novel labels |
| `Relation direction accuracy` | Kiểm tra `holding` không bị đảo subject/object |
| `Pair localization recall` | Cặp box đúng trước khi đánh giá predicate |

### 8.3. Latent space

UMAP/t-SNE như hình minh họa có giá trị trực quan nhưng không được dùng làm bằng chứng duy nhất. Cần đo:

1. intra-group cosine distance;
2. inter-group margin;
3. sibling leaf margin;
4. k-NN group purity;
5. silhouette score ở group space;
6. zero-shot nearest-prototype accuracy ở leaf space.

Nên vẽ riêng `z_group` và `z_leaf`. Kỳ vọng đúng là `z_group` có cụm `person`, `container`, `table furniture`, `seating furniture`; còn `z_leaf` trong cụm `person` vẫn tách được `man/woman/girl`. Nếu cả hai không gian đều collapse về group thì mô hình chưa đạt mục tiêu.

### 8.4. Ablation bắt buộc

| Ablation | Câu hỏi kiểm chứng |
|---|---|
| Không có group head | Group supervision có giúp novel leaf không? |
| Không có sibling loss | `table/chair`, `cup/bottle` có bị lẫn không? |
| Chỉ text clustering, không human ontology | K-means/LLM có tạo nhóm sai không? |
| Không có union-region | Relation có phụ thuộc quá nhiều vào nhãn object không? |
| Không có geometry | Quan hệ spatial/contact giảm bao nhiêu? |
| Hard negative thường vs sibling-aware | Nhãn nhóm có thực sự được bảo vệ không? |
| GT pair boxes vs predicted boxes | Train–test mismatch ảnh hưởng thế nào? |
| Freeze VLM vs LoRA vs full fine-tune | Chi phí và catastrophic forgetting |
| Không có pseudo-label | VLM teacher giúp novel relation đến đâu? |

## 9. Rủi ro kỹ thuật và cách xử lý

**Rủi ro 1: group semantic lấn át leaf semantic.** Biểu hiện là AP group cao nhưng leaf accuracy thấp, sibling confusion cao. Xử lý bằng hai projection head, sibling margin, group loss nhỏ hơn leaf loss và đánh giá riêng hai không gian.

**Rủi ro 2: text embedding sai do polysemy hoặc prompt bias.** Dùng canonical label, nhiều prompt template, paraphrase có kiểm duyệt và prototype ensemble. Không để LLM tự quyết định rằng hai label là synonym chỉ vì gần nhau trong ngôn ngữ.

**Rủi ro 3: relation hallucination từ VLM.** Pseudo-label phải có confidence/consistency filter; human relation label luôn có trọng số cao hơn teacher. Relation decoder phải nhìn union-region và geometry, không chỉ đọc text object labels.

**Rủi ro 4: partial annotation bị xem là negative.** Chỉ lấy cặp không có relation làm negative khi dataset ghi rõ exhaustive scope. Với Visual Genome, dùng annotation mask; với unknown pair, bỏ khỏi BCE/focal loss thay vì phạt.

**Rủi ro 5: long-tail và class prior.** Dùng group-balanced sampler, repeat factor cho leaf hiếm, logit adjustment hoặc class-balanced focal. Không oversample quá mạnh các label novel đã giữ lại cho test.

**Rủi ro 6: quadratic relation computation.** Dùng top-M objects, relationness proposal, sparse pair attention và cache union features. Khi cần nhiều quan hệ, có thể dùng two-stage ranking: relationness trước, predicate classification sau.

**Rủi ro 7: train–test mismatch.** Relation branch phải được chuyển từ GT/matched boxes sang predicted boxes theo curriculum. Nếu chỉ train trên GT pair, kết quả SGDet sẽ giảm mạnh dù PredCls đẹp.

**Rủi ro 8: ontology thay đổi làm hỏng checkpoint.** Lưu ontology version cùng checkpoint, prototype cache và mapping hash. Khi thêm label mới, không sửa tên cũ âm thầm; tạo version mới và đánh giá backward compatibility.

## 10. Lộ trình triển khai thực tế

### MVP nghiên cứu

Bắt đầu với 20–40 leaf object labels và 10–20 predicate labels, nhưng giữ taxonomy rõ ràng: `person → man/woman/girl/boy`, `container → cup/bottle`, `furniture → chair/table` với sibling-negative. Train detector trên một subset Open Images/VG, dùng CLIP/SigLIP text prototypes, thêm 2-layer relation decoder và báo cáo S/S, N/S, S/N, N/N.

### Bản đầy đủ

Sau khi MVP ổn định, mở lên ontology nhiều tầng, Open Images V6 cho relation, Visual Genome cho lexical diversity, GQA cho auxiliary scene-graph reasoning và caption/grounding data cho alignment. Dùng Grounding DINO initialization nếu ưu tiên zero-shot box quality; dùng custom DETR nếu ưu tiên nghiên cứu loss/architecture minh bạch.

### Tiêu chí đạt trước khi mở rộng dữ liệu

Không nên mở rộng vocabulary chỉ vì UMAP đẹp. Một checkpoint chỉ nên được coi là tiến bộ khi đồng thời đạt: novel AP tăng hoặc harmonic mean tăng; sibling confusion không tăng; zero-shot triplet recall tăng; group consistency cao; và calibration trên novel không xấu đi quá mức. Nếu group compactness tăng nhưng leaf AP giảm, cần giảm `λ_compact` hoặc tăng sibling margin.

## 11. Kết luận thiết kế

Mô hình đúng với ý tưởng của bạn là một **factorized hierarchical open-vocabulary scene graph detector**, không phải một classifier gom tất cả nhãn vào một latent centroid. Tầng group dùng để học sự tương đồng semantic và chuyển giao từ `person` sang `man/woman/girl`, hoặc từ `container` sang `cup/bottle`. Tầng leaf giữ khả năng phân biệt các nhãn cụ thể. Relation branch học compositional interaction từ hai object latent, union-region và hình học; vì vậy `person holding cup` có thể mở rộng theo cả subject và object mà không cần tạo một class nguyên tử cho từng triplet.

Khuyến nghị triển khai theo thứ tự: **ontology audit → detector warm-up → hierarchical alignment → relation decoder → joint fine-tuning → strict open-vocabulary evaluation**. Nếu phải chọn một phiên bản đầu tiên, hãy dùng Open Images V6 làm nguồn relation chính, Visual Genome làm nguồn lexical/region augmentation, khởi tạo detector theo Grounding DINO hoặc DETR-VLM, và tập trung chứng minh ba kết quả: `novel object`, `novel relation`, và `novel composition`.

## References

[1]: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html "The Visual Genome Dataset: official API and downloads"

[2]: https://storage.googleapis.com/openimages/web/factsfigures.html "Open Images V6: official facts and figures"

[3]: https://cs.stanford.edu/people/dorarad/gqa/about.html "GQA Dataset: Visual Reasoning in the Real World"

[4]: https://arxiv.org/html/2404.00906v2 "From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models"

[5]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880055.pdf "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning"

[6]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11211.pdf "Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection"

[7]: https://arxiv.org/html/2311.10988v2 "Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention"

[8]: https://arxiv.org/html/2412.19021v1 "Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation: supplementary details"

[9]: https://arxiv.org/html/2303.05499v5 "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"

---

**Lưu ý:** Các con số và mô tả dataset trong tài liệu được lấy từ trang chính thức hoặc bài báo được dẫn ở References. Các hyperparameter, loss weight, taxonomy minh họa và protocol split là đề xuất kỹ thuật cần được hiệu chỉnh trên dữ liệu thực tế của dự án.


## Ghi chú kiểm thử skeleton

File `hovr_sg_model.py` đã vượt qua kiểm tra cú pháp bằng `python3 -m py_compile`. Smoke test forward pass chưa chạy được vì môi trường sandbox hiện không cài PyTorch (`ModuleNotFoundError: No module named 'torch'`); đây là giới hạn môi trường, không phải lỗi cú pháp của skeleton. Khi triển khai dự án, cài PyTorch phù hợp CUDA trước khi chạy test shape và training.
