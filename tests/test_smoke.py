import torch

from hovr_sg.models import HOVRSG, deterministic_text_embeddings


def test_forward_shapes():
    model = HOVRSG(visual_dim=32, d_model=64, num_queries=16, d_latent=32)
    visual = torch.randn(2, 40, 32)
    leaf = deterministic_text_embeddings(["man", "woman", "cup"], 32)
    groups = deterministic_text_embeddings(["person", "container"], 32)
    rel = deterministic_text_embeddings(["holding", "on"], 32)
    output = model(visual, leaf, groups, rel, top_m=8, top_k_pairs=16)
    assert output.boxes.shape == (2, 16, 4)
    assert output.leaf_logits.shape == (2, 16, 3)
    assert output.group_logits.shape == (2, 16, 2)
    assert output.relations["relation_logits"].shape == (2, 16, 2)
