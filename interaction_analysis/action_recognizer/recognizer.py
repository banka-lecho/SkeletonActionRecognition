import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for processing skeletal points data.

    Args:
        input_dim (int): Dimension of input node features. Default: 3 (x,y,confidence).
        hidden_dim (int): Dimension of hidden layer. Default: 64.
        output_dim (int): Dimension of output features. Default: 32.
    """

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # <-- Добавлено
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)  # <-- Добавлено

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the GCN.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim]
            edge_index (Tensor): Graph connectivity in COO format [2, num_edges]

        Returns:
            Tensor: Output node features [num_nodes, output_dim]
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x


class SuperFormer(nn.Module):
    """Multimodal action recognition model combining mask, optical flow and skeleton data."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Mask processing stream
        self.mask_stream = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

        # Optical flow processing stream
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

        # Skeletal points processing
        self.gcn = GCN(input_dim=3, hidden_dim=64, output_dim=256)

        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):  # Для слоев GCN из torch_geometric
                nn.init.kaiming_normal_(m.lin.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mask: Tensor, optical_flow: Tensor,
                skeleton_points: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the SuperFormer model.

                Args:
                    mask (Tensor): Object mask tensor [batch_size, 1, H, W]
                    optical_flow (Tensor): Optical flow tensor [batch_size, 2, H, W]
                    skeleton_points (Tensor): Skeletal points coordinates [num_nodes, 3] (x,y,confidence)
                    edge_index (Tensor): Skeleton graph edges [2, num_edges]

                Returns:
                    Tensor: Classification logits [batch_size, num_classes]
                """
        # Process mask stream
        mask_features = self.mask_stream(mask).flatten(1)  # [B, 256]

        # Process optical flow stream
        flow_features = self.temporal_stream(optical_flow).flatten(1)  # [B, 256]

        # Process skeleton points
        B, N, C = skeleton_points.shape
        # Обрабатываем каждый скелетон в батче
        skeleton_features = []
        for i in range(B):
            # [N, C] -> [N, 256] через GCN
            feat = self.gcn(skeleton_points[i], edge_index)
            # Global max pooling для каждого скелетона
            feat = torch.max(feat, dim=0)[0]  # [256]
            skeleton_features.append(feat)

        skeleton_features = torch.stack(skeleton_features)  # [B, 256]

        # Attention-weighted feature fusion
        combined = torch.cat([mask_features, flow_features, skeleton_features], dim=1)
        result = (combined[:, 0:1] * mask_features +
                  combined[:, 1:2] * flow_features +
                  combined[:, 2:3] * skeleton_features)
        return self.fc(result)
