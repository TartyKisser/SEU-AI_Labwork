import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=64, num_classes=None):
        super().__init__()
        self.name = "GCN"
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # 保存类别数量
        self.num_classes = num_classes

        # 添加分类器层
        # 对于多分类，num_classes应该是类别的数量
        # 对于二分类，num_classes可以是1（使用sigmoid）或2（使用softmax）
        if num_classes is not None:
            self.classifier = torch.nn.Linear(out_channels, num_classes)

    def encode(self, x, edge_index):
        """编码输入特征为节点嵌入"""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """计算边的得分用于链接预测（保持向后兼容性）"""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def classify(self, z):
        """对节点进行分类"""
        if not hasattr(self, 'classifier'):
            raise AttributeError("模型未配置分类器。在创建模型时请指定num_classes参数。")

        # 返回分类结果
        # 对于多分类：输出形状为[num_nodes, num_classes]的logits
        # 对于二分类：如果num_classes=1，输出形状为[num_nodes]的logits
        return self.classifier(z)

    def forward(self, x, edge_index):
        """前向传播，结合编码和分类"""
        z = self.encode(x, edge_index)
        if hasattr(self, 'classifier'):
            return self.classify(z)
        return z