import torch

from model import Model, SiameseModel


def test_model_forward_output_shape():
    """Model.forward should return (batch, embedding_dim)."""
    batch = 5
    embedding_dim = 64
    model = Model(embedding_dim=embedding_dim)
    model.eval()

    dummy_input = torch.randn(batch, 7, 10)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch, embedding_dim)


def test_siamese_forward_shape_and_similarity_ordering():
    """
    SiameseModel.forward should return (batch, 1), and identical pairs should
    produce higher similarity logits (i.e., less negative distance) than random pairs.
    """
    torch.manual_seed(0)
    batch = 6

    feature_extractor = Model(embedding_dim=32)
    siamese = SiameseModel(feature_extractor)
    siamese.eval()

    x = torch.randn(batch, 7, 10)
    x_random = torch.randn(batch, 7, 10)

    with torch.no_grad():
        identical_logits = siamese(x, x)
        random_logits = siamese(x, x_random)

    assert identical_logits.shape == (batch, 1)
    assert random_logits.shape == (batch, 1)

    # For identical inputs, distance should be zero => logit close to 0.
    assert torch.allclose(
        identical_logits,
        torch.zeros_like(identical_logits),
        atol=1e-6,
    )

    # Since logits are negative distances, higher means more similar.
    assert torch.mean(identical_logits) > torch.mean(random_logits)


def test_build_edge_index_connectivity_regression():
    """Regression guard for graph topology size/connectivity."""
    model = Model()
    edge_index = model._build_edge_index()

    # Expect shape (2, E) with stable directed edge count from architecture.
    # 4 orientation nodes <-> node 7: 8 edges
    # 3 position nodes <-> node 8: 6 edges
    # 7 <-> 8: 2 edges
    # 7 <-> 9 and 8 <-> 9: 4 edges
    # Total: 20 directed edges
    assert edge_index.shape == (2, 20)

    # Also keep node index range stable (0..9).
    assert int(edge_index.min()) == 0
    assert int(edge_index.max()) == 9
