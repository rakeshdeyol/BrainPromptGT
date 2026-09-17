import torch

from transformer_directional_pe.model import DirectionalTransformerClassifier


def make_directed_graph(batch_size, num_nodes):
    adjacency = torch.zeros(batch_size, num_nodes, num_nodes)
    source = torch.arange(num_nodes - 1)
    target = source + 1
    adjacency[:, source, target] = 1.0
    adjacency[:, -1, 0] = 0.5
    adjacency[:, 0, 2] = 2.0
    return adjacency


if __name__ == "__main__":
    torch.manual_seed(7)
    batch_size = 4
    num_nodes = 8
    model = DirectionalTransformerClassifier(
        input_dim=5,
        num_nodes=num_nodes,
        d_model=32,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        walk_steps=3,
        dropout=0.1,
    )

    node_features = torch.randn(batch_size, num_nodes, 5)
    adjacency = make_directed_graph(batch_size, num_nodes)
    logits, attentions = model(node_features, adjacency, return_attention=True)

    assert logits.shape == (batch_size, 2)
    assert len(attentions) == 2
    assert attentions[0].shape == (batch_size, 4, num_nodes, num_nodes)
    print("logits:", tuple(logits.shape))
    print("attention:", tuple(attentions[0].shape))
    print("standalone directional Transformer: OK")
