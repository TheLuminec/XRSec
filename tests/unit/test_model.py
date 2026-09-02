import torch

from extractors.paper_gnn_bilstm import Model
from model import SiameseModel


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


def test_siamese_forward_shape_and_identical_pair_invariance():
    """
    SiameseModel.forward should return (batch, 1).

    The head classifies |e1 - e2| through a learned linear layer, so an identical
    pair has a zero difference vector and must collapse to exactly the classifier
    bias for every row. Distinct pairs must not, which guards against embedding
    collapse or a head that ignores its input.

    Note: the logit is *not* a negative distance, so an untrained model carries no
    ordering guarantee between identical and random pairs. Ordering is something
    training has to produce, not an architectural invariant.
    """
    torch.manual_seed(0)
    batch = 6
    embedding_dim = 32

    feature_extractor = Model(embedding_dim=embedding_dim)
    siamese = SiameseModel(feature_extractor, embedding_dim=embedding_dim)
    siamese.eval()

    x = torch.randn(batch, 7, 10)
    x_random = torch.randn(batch, 7, 10)

    with torch.no_grad():
        identical_logits = siamese(x, x)
        random_logits = siamese(x, x_random)

    assert identical_logits.shape == (batch, 1)
    assert random_logits.shape == (batch, 1)

    # Zero difference vector => every row equals the classifier bias.
    assert torch.allclose(
        identical_logits,
        siamese.classifier.bias.expand_as(identical_logits),
        atol=1e-6,
    )

    # The head must actually respond to the inputs it is given.
    assert random_logits.std() > 1e-6
    assert not torch.allclose(random_logits, identical_logits, atol=1e-6)


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
