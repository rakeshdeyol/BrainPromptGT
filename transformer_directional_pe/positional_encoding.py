import torch
import torch.nn as nn


class DirectionalRandomWalkPE(nn.Module):
    """Encode directed incoming and outgoing random-walk structure per node."""

    def __init__(self, num_nodes, d_model, steps=3, dropout=0.0):
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")

        self.num_nodes = num_nodes
        self.steps = steps
        self.projection = nn.Linear(2 * steps * num_nodes, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _row_normalize(adjacency):
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency / degree

    def forward(self, x, adjacency):
        """
        Args:
            x: Node tokens with shape [batch, nodes, d_model].
            adjacency: Directed weighted adjacency with shape [batch, nodes, nodes].
        """
        if x.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("x and adjacency must both be 3D tensors")
        batch_size, node_count, _ = x.shape
        if node_count != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {node_count}")
        if adjacency.shape != (batch_size, node_count, node_count):
            raise ValueError("adjacency must have shape [batch, nodes, nodes]")

        outgoing = self._row_normalize(adjacency)
        incoming = self._row_normalize(adjacency.transpose(-1, -2))

        outgoing_power = outgoing
        incoming_power = incoming
        features = []
        for _ in range(self.steps):
            features.extend((outgoing_power, incoming_power))
            outgoing_power = outgoing_power @ outgoing
            incoming_power = incoming_power @ incoming

        # Each node receives one row from every walk-power matrix.
        positional_features = torch.cat(features, dim=-1)
        return x + self.dropout(self.projection(positional_features))
