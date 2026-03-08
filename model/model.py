"""
Section 3.1: Advanced Machine Learning Architecture for Biometric Sensing

Architecture (from NSF ERI 2023 paper):
    Input M (7x10) -> GNN Aggregation (Ga, Gp) -> <M, M', b>
    -> BiLSTM -> Attention -> BiLSTM -> Attention -> Dense + Softmax

Steps:
    - Input: 7x10 matrix (7 data channels x 10 time samples per second)
    - Ga (GATConv, concat): Graph attention over data channels -> M'
    - Gp (Sum aggregation): Graph sum pooling over data channels -> b
    - BiLSTM 1: Bidirectional LSTM for temporal feature extraction
    - Attention 1: Self-attention (Eq 1-3) to re-weight sequence
    - BiLSTM 2: Second bidirectional LSTM layer
    - Attention 2: Self-attention followed by pooling
    - Dense: Classification layer mapping to num_users classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv
from torch_geometric.data import Data, Batch


class SelfAttention(nn.Module):
    """
    Attention mechanism from paper Equations 1-3:
        Q  = tanh(O * Wa + ba)
        a  = softmax(Q * O^T)
        Oa = a * O
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)

    def forward(self, O):
        """
        Args:
            O: (batch, seq_len, hidden_size) - BiLSTM output
        Returns:
            Oa: (batch, seq_len, hidden_size) - attention-weighted output
        """
        Q = torch.tanh(self.Wa(O))                          # Eq 1
        scores = torch.bmm(Q, O.transpose(1, 2))            # Eq 2 (pre-softmax)
        a = F.softmax(scores, dim=-1)                        # Eq 2
        Oa = torch.bmm(a, O)                                # Eq 3
        return Oa


class Model(nn.Module):
    """
    Neural network model for XR user biometric identification.

    Implements Thrust I from Section 3.1:
        - Task 1 (§3.1.2): GNN for relationship-based data aggregation
        - Task 2 (§3.1.3): Attention mechanism for improved identification
        - Preliminary (§3.1.1): BiLSTM backbone for temporal patterns

    Args:
        num_users:    Number of users to classify
        seq_len:      Number of time samples per window (default: 10)
        num_channels: Number of data channels (default: 7 = qx,qy,qz,qw,Hx,Hy,Hz)
        gnn_hidden:   Hidden dimension for GNN layers
        lstm_hidden:  Hidden dimension for LSTM layers
        gat_heads:    Number of attention heads in GATConv
    """
    def __init__(self, num_users=None, seq_len=10, num_channels=7,
                 gnn_hidden=32, lstm_hidden=64, gat_heads=4, embedding_dim=None):
        super().__init__()
        self.num_users = num_users
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.gnn_hidden = gnn_hidden
        self.lstm_hidden = lstm_hidden
        self.gat_heads = gat_heads
        self.num_nodes = num_channels + 3  # 7 data + orientation + position + root

        # Fixed graph structure (§3.1.2 Figure 6)
        self.register_buffer('edge_index', self._build_edge_index())

        # --- Ga: Graph Attention Network (concatenation aggregation) ---
        # GATConv with multi-head attention, concat=True concatenates head outputs
        self.ga_conv1 = GATConv(seq_len, gnn_hidden, heads=gat_heads, concat=True)
        self.ga_conv2 = GATConv(gnn_hidden * gat_heads, seq_len, heads=1, concat=False)

        # --- Gp: GNN with sum aggregation ---
        self.gp_conv1 = GraphConv(seq_len, gnn_hidden, aggr='add')
        self.gp_conv2 = GraphConv(gnn_hidden, seq_len, aggr='add')

        # --- BiLSTM layers (§3.1.1) ---
        # Input: <M, M', b> concatenated = 3 * num_channels per time step
        lstm_input_size = num_channels * 3
        self.lstm1 = nn.LSTM(lstm_input_size, lstm_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                             batch_first=True, bidirectional=True)

        # --- Attention layers (§3.1.3 Equations 1-3) ---
        self.attn1 = SelfAttention(lstm_hidden * 2)
        self.attn2 = SelfAttention(lstm_hidden * 2)

        # --- Dense classification/embedding layer ---
        if num_users is not None:
            self.fc = nn.Linear(lstm_hidden * 2, num_users)
        elif embedding_dim is not None:
            self.fc = nn.Linear(lstm_hidden * 2, embedding_dim)
        else:
            self.fc = nn.Identity()

    def _build_edge_index(self):
        """
        Build the graph structure from Figure 6:
            - Nodes 0-3: qx, qy, qz, qw (orientation data)
            - Nodes 4-6: Hx, Hy, Hz (position data)
            - Node 7: orientation aggregate (dark blue dot)
            - Node 8: position aggregate (orange dot)
            - Node 9: root / overall headset movement (red dot)

        Edges connect data nodes to their group, groups to each other,
        and both groups to the root node.
        """
        edges = []
        # qx,qy,qz,qw <-> orientation node
        for i in range(4):
            edges.extend([[i, 7], [7, i]])
        # Hx,Hy,Hz <-> position node
        for i in range(4, 7):
            edges.extend([[i, 8], [8, i]])
        # orientation <-> position
        edges.extend([[7, 8], [8, 7]])
        # orientation -> root, position -> root
        edges.extend([[7, 9], [9, 7], [8, 9], [9, 8]])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _run_gnn(self, M, conv1, conv2):
        """
        Run a 2-layer GNN on the graph built from input M.

        Args:
            M: (batch, 7, seq_len) - input data matrix
            conv1, conv2: GNN convolutional layers

        Returns:
            (batch, 7, out_dim) - updated features for data nodes only
        """
        batch_size = M.size(0)
        device = M.device

        # Pad with 3 aggregate nodes (zeros) -> (batch, 10, seq_len)
        pad = torch.zeros(batch_size, 3, self.seq_len, device=device)
        node_features = torch.cat([M, pad], dim=1)

        # Create PyG batch (all graphs share the same edge structure)
        data_list = [
            Data(x=node_features[i], edge_index=self.edge_index)
            for i in range(batch_size)
        ]
        batch = Batch.from_data_list(data_list)
        batch = batch.to(device)

        # Two-layer GNN forward pass
        h = F.relu(conv1(batch.x, batch.edge_index))
        h = conv2(h, batch.edge_index)

        # Reshape and extract only the 7 data node features
        h = h.view(batch_size, self.num_nodes, -1)
        return h[:, :self.num_channels, :]

    def forward(self, M):
        """
        Forward pass.

        Args:
            M: (batch, 7, 10) - input matrix where rows are data channels
               (qx, qy, qz, qw, Hx, Hy, Hz) and columns are 10 time samples

        Returns:
            (batch, num_users) - classification logits
        """
        # === GNN Data Aggregation (§3.1.2) ===
        # M' = Ga(data) - graph attention augmented features
        M_prime = self._run_gnn(M, self.ga_conv1, self.ga_conv2)  # (batch, 7, 10)
        # b = Gp(data) - sum-aggregated importance
        b = self._run_gnn(M, self.gp_conv1, self.gp_conv2)       # (batch, 7, 10)

        # Prepare LSTM input: <M, M', b> concatenated along channel dimension
        # Transpose each from (batch, 7, 10) to (batch, 10, 7) for time-step-major
        x = torch.cat([
            M.permute(0, 2, 1),           # (batch, 10, 7)
            M_prime.permute(0, 2, 1),      # (batch, 10, 7)
            b.permute(0, 2, 1),            # (batch, 10, 7)
        ], dim=2)                          # (batch, 10, 21)

        # === BiLSTM + Attention (§3.1.1 & §3.1.3) ===
        # BiLSTM 1: extract forward + backward temporal features
        x, _ = self.lstm1(x)               # (batch, 10, 2*lstm_hidden)

        # Attention 1: re-weight sequence via self-attention
        x = self.attn1(x)                  # (batch, 10, 2*lstm_hidden)

        # BiLSTM 2: further temporal pattern extraction
        x, _ = self.lstm2(x)               # (batch, 10, 2*lstm_hidden)

        # Attention 2: re-weight sequence via self-attention
        x = self.attn2(x)                  # (batch, 10, 2*lstm_hidden)

        # Pool over time dimension -> fixed-size vector
        x = x.mean(dim=1)                  # (batch, 2*lstm_hidden)

        # === Dense Classification/Embedding ===
        x = self.fc(x)                     # (batch, num_users) or (batch, embedding_dim)
        return x


class SiameseModel(nn.Module):
    """
    Siamese wrapper around the Model feature extractor.
    Given two sequences, it computes their distance following Eq (6):
        D(Vs, Vi) = 1 / (1 + exp(||Vs - Vi||_2))
    
    This outputs the negative distance as a logit, which can be directly
    used with BCEWithLogitsLoss to optimize Eq (7).
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2):
        # Extract features (embeddings)
        e1 = self.feature_extractor(x1)
        e2 = self.feature_extractor(x2)
        
        # Compute L2 distance ||Vs - Vi||_2
        # keepdim=True ensures shape (batch, 1) to match with target labels
        dist = torch.norm(e1 - e2, p=2, dim=1, keepdim=True)
        
        # We return the logit which is the negative distance.
        # This way, sigmoid(-dist) = 1 / (1 + exp(dist)) matches Eq (6)
        return -dist


if __name__ == "__main__":
    # Quick smoke test for Model
    model = Model(num_users=48)
    dummy_input = torch.randn(4, 7, 10)   # batch of 4, 7 channels, 10 time steps
    output = model(dummy_input)
    print(f"Base Model Output shape: {output.shape}")

    # Quick smoke test for SiameseModel
    feature_extractor = Model(embedding_dim=128)
    siamese_model = SiameseModel(feature_extractor)
    dummy_input2 = torch.randn(4, 7, 10)
    output_siamese = siamese_model(dummy_input, dummy_input2)
    print(f"Siamese Output shape:  {output_siamese.shape}")
    print(f"Siamese probabilities: {torch.sigmoid(output_siamese).squeeze().detach().cpu().numpy()}")
    print(f"Num params:   {sum(p.numel() for p in siamese_model.parameters()):,}")
