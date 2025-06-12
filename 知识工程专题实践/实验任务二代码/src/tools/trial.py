from typing import List, Union, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Data, HeteroData
from torch_geometric import transforms as T
import os
import numpy as np


class FocalLoss(torch.nn.Module):
    """Focal Loss implementation to handle class imbalance"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  # 添加类别权重支持

    def forward(self, inputs, targets):
        # 对于多分类任务
        if inputs.dim() > 1 and inputs.size(1) > 1:  # 多分类情况
            # 确保targets是长整型用于计算交叉熵
            if not torch.is_integral(targets):
                targets = targets.long()

            # 根据是否提供类别权重选择不同的交叉熵函数
            if self.class_weights is not None:
                # 确保权重在正确的设备上
                weights = self.class_weights
                if weights.device != inputs.device:
                    weights = weights.to(inputs.device)
                CE_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
            else:
                CE_loss = F.cross_entropy(inputs, targets, reduction='none')

            # 计算focal loss
            pt = torch.exp(-CE_loss)
            F_loss = (1 - pt) ** self.gamma * CE_loss

            # 应用alpha平衡因子（如果提供）
            if self.alpha is not None:
                F_loss = self.alpha * F_loss

        else:  # 二分类情况
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = (1 - pt) ** self.gamma * BCE_loss

            if self.alpha is not None:
                F_loss = self.alpha * F_loss

        # 根据reduction模式返回
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class Study:
    def _unpack_data(self, data):
        """提取图数据中的特征、边索引、标签和掩码"""
        device = next(self.model.parameters()).device  # 获取模型设备

        if isinstance(data, Data):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            # 优先使用parcel_types作为节点标签，如果没有则使用y
            if hasattr(data, 'parcel_types'):
                node_labels = data.parcel_types.to(device)
            else:
                node_labels = data.y.to(device)

            # 获取相应的mask
            if hasattr(data, 'train_mask'):
                mask = data.train_mask.to(device)
            else:
                mask = None
        else:
            # 异构图的情况
            x = {k: v.to(device) for k, v in data.x_dict.items()}
            edge_index = {k: v.to(device) for k, v in data.edge_index_dict.items()}

            # 获取parcel节点的标签
            node_labels = data["parcel"].y.to(device)
            mask = data["parcel"].train_mask.to(device) if hasattr(data["parcel"], 'train_mask') else None

        return x, edge_index, node_labels, mask

    @torch.no_grad()
    def _eval(self, data):
        """评估模型在验证/测试集上的准确率"""
        self.model.eval()

        x, edge_index, node_labels, mask = self._unpack_data(data)

        # 检查模型是否有classify方法
        has_classify = hasattr(self.model, 'classify')

        # 获取节点嵌入
        if isinstance(data, Data):
            z = self.model.encode(x, edge_index)

            if has_classify:
                # 使用classify方法获取预测
                out = self.model.classify(z)

                # 处理掩码
                if mask is not None:
                    # 确保node_labels是长整型用于argmax比较
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    pred = out[mask].argmax(dim=1) if out.dim() > 1 and out.size(1) > 1 else (out[mask] > 0).float()
                    y = node_labels[mask]
                else:
                    # 确保node_labels是长整型用于argmax比较
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    pred = out.argmax(dim=1) if out.dim() > 1 and out.size(1) > 1 else (out > 0).float()
                    y = node_labels
            else:
                # 对于没有classify方法的模型，使用decode方法
                if mask is not None:
                    nodes = torch.nonzero(mask).squeeze(-1)
                else:
                    nodes = torch.arange(z.size(0), device=z.device)

                edge_label_index = torch.stack([nodes, nodes], dim=0)
                out = self.model.decode(z, edge_label_index)
                pred = (out > 0).float()

                if mask is not None:
                    y = node_labels[mask]
                else:
                    y = node_labels
        else:
            # 异构图的情况
            z_dict = self.model.encode(x, edge_index)

            if has_classify:
                out = self.model.classify(z_dict["parcel"])

                if mask is not None:
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    pred = out[mask].argmax(dim=1) if out.dim() > 1 and out.size(1) > 1 else (out[mask] > 0).float()
                    y = node_labels[mask]
                else:
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    pred = out.argmax(dim=1) if out.dim() > 1 and out.size(1) > 1 else (out > 0).float()
                    y = node_labels
            else:
                # 对于没有classify方法的异构图
                if mask is not None:
                    nodes = torch.nonzero(mask).squeeze(-1)
                else:
                    nodes = torch.arange(z_dict["parcel"].size(0), device=z_dict["parcel"].device)

                edge_label_index = torch.stack([nodes, nodes], dim=0)
                parcel_emb = z_dict["parcel"]
                out = torch.sum(parcel_emb[edge_label_index[0]] * parcel_emb[edge_label_index[1]], dim=1)
                pred = (out > 0).float()

                if mask is not None:
                    y = node_labels[mask]
                else:
                    y = node_labels

        # 计算准确率
        return accuracy_score(y.cpu().numpy(), pred.cpu().numpy())

    def _train(self, optimizer, criterion, scheduler=None):
        """训练模型一个epoch"""
        self.model.train()
        optimizer.zero_grad()

        x, edge_index, node_labels, mask = self._unpack_data(self.train_data)

        # 获取节点嵌入和预测
        if isinstance(self.train_data, Data):
            z = self.model.encode(x, edge_index)
            out = self.model.classify(z)

            # 处理掩码
            if mask is not None:
                # 多分类情况
                if out[mask].dim() > 1 and out[mask].size(1) > 1:
                    # 确保node_labels是长整型用于CrossEntropyLoss
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    loss = criterion(out[mask], node_labels[mask])
                else:
                    # 二分类情况
                    loss = criterion(out[mask], node_labels[mask].float())
            else:
                # 多分类情况
                if out.dim() > 1 and out.size(1) > 1:
                    # 确保node_labels是长整型用于CrossEntropyLoss
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    loss = criterion(out, node_labels)
                else:
                    # 二分类情况
                    loss = criterion(out, node_labels.float())
        else:
            # 异构图的情况
            z_dict = self.model.encode(x, edge_index)
            out = self.model.classify(z_dict["parcel"])

            if mask is not None:
                if out[mask].dim() > 1 and out[mask].size(1) > 1:
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    loss = criterion(out[mask], node_labels[mask])
                else:
                    loss = criterion(out[mask], node_labels[mask].float())
            else:
                if out.dim() > 1 and out.size(1) > 1:
                    if not torch.is_integral(node_labels):
                        node_labels = node_labels.long()
                    loss = criterion(out, node_labels)
                else:
                    loss = criterion(out, node_labels.float())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return loss

    def _backward_compatible_train(self, optimizer, criterion, scheduler=None):
        """为没有classify方法的模型提供向后兼容的训练方法"""
        self.model.train()
        optimizer.zero_grad()

        x, edge_index, node_labels, mask = self._unpack_data(self.train_data)

        z = self.model.encode(x, edge_index)

        if mask is not None:
            nodes = torch.nonzero(mask).squeeze(-1)
        else:
            nodes = torch.arange(z.size(0), device=z.device)

        edge_label_index = torch.stack([nodes, nodes], dim=0)
        out = self.model.decode(z, edge_label_index)

        if isinstance(criterion, torch.nn.BCELoss):
            out = torch.sigmoid(out)

        if mask is not None:
            target = node_labels[mask]
        else:
            target = node_labels

        # 注意：这种方式主要适用于二分类，对多分类支持有限
        if target.dim() > 1 and target.size(1) > 1:
            pass
        else:
            target = target.float()

            if out.dim() > 1 and out.size(1) > 1:
                if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    out = out[:, 0]

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss

    def __init__(
            self, model, train_graph: Union[Data, HeteroData], node_split=None
    ) -> None:
        """
        初始化Study类用于节点分类

        参数:
            model: 节点分类模型
            train_graph: 图数据
            node_split: 可选的节点分割变换
        """
        self.model = model

        if node_split is not None:
            self.train_data, self.val_data, self.test_data = node_split(train_graph)
        else:
            # 如果没有提供分割，则对训练/验证/测试使用相同的图
            self.train_data = self.val_data = self.test_data = train_graph

    def train(
            self, epoch, optimizer, criterion=None, scheduler=None, verbose=0, save_dir=None,
            class_weights=None
    ):
        """
        训练节点分类模型

        参数:
            epoch: 训练的轮数
            optimizer: PyTorch优化器
            criterion: 损失函数（默认：FocalLoss处理类别不平衡）
            scheduler: 可选的学习率调度器
            verbose: 详细程度级别
            save_dir: 保存模型检查点的目录
            class_weights: 可选的类别权重张量
        """
        # 检查模型是否有classify方法
        has_classify = hasattr(self.model, 'classify')

        # 默认使用FocalLoss处理类别不平衡
        if criterion is None:
            # 如果提供了类别权重，使用它们
            if class_weights is not None:
                criterion = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
            else:
                # 检查是否可以从训练数据计算类别权重
                if hasattr(self.train_data, 'parcel_types') and hasattr(self.train_data, 'train_mask'):
                    # 仅使用训练掩码中的节点计算类别权重
                    labels = self.train_data.parcel_types[self.train_data.train_mask].cpu()
                    classes = torch.unique(labels)
                    class_counts = torch.bincount(labels, minlength=len(classes))
                    # 反比于类别频率的权重
                    weights = 1.0 / class_counts.float()
                    # 归一化权重
                    weights = weights / weights.sum() * len(classes)
                    weights = weights.to(next(self.model.parameters()).device)
                    criterion = FocalLoss(alpha=0.25, gamma=2.0, class_weights=weights)
                else:
                    criterion = FocalLoss(alpha=0.25, gamma=2.0)
        # 如果使用CrossEntropyLoss并提供了类别权重
        elif isinstance(criterion, torch.nn.CrossEntropyLoss) and class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.loss = []
        self.valid_acc = []
        self.test_acc = []

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_valid_acc = 0
        final_test_acc = 0

        for e in range(epoch):
            # 使用适当的训练方法
            if has_classify:
                loss = self._train(optimizer, criterion, scheduler)
            else:
                loss = self._backward_compatible_train(optimizer, criterion, scheduler)

            valid_acc = self._eval(self.val_data)
            test_acc = self._eval(self.test_data)

            self.loss.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            self.valid_acc.append(valid_acc)
            self.test_acc.append(test_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                final_test_acc = test_acc

                # 保存最佳模型
                if save_dir is not None:
                    torch.save(
                        self.model.state_dict(), f"{save_dir}/{self.model.name}_best.pt"
                    )

            if verbose == 0:
                print(
                    f"Epoch: {e + 1:d}/{epoch:d}, Loss: {loss:.4f}, Val Acc: {valid_acc:.4f}, "
                    f"Test Acc: {test_acc:.4f}"
                )

            if save_dir is not None:
                torch.save(
                    self.model.state_dict(), f"{save_dir}/{self.model.name}_{e + 1}.pt"
                )

        if verbose <= 1:
            print(f"Final test accuracy: {final_test_acc:.4f}")

    @torch.no_grad()
    def test(self, data, return_metrics=True, zero_division=0):
        """
        测试节点分类模型

        参数:
            data: 测试图数据
            return_metrics: 是否返回额外的评估指标
            zero_division: 当没有预测某个类别时的精确率/召回率值

        返回:
            accuracy: 分类准确率
            pred_classes: 每个节点的预测类别
            metrics: 额外评估指标的字典（如果return_metrics=True）
        """
        assert type(self.train_data) == type(data), "Data type mismatch"
        self.model.eval()

        x, edge_index, node_labels, mask = self._unpack_data(data)

        # 获取节点嵌入和预测
        if isinstance(data, Data):
            z = self.model.encode(x, edge_index)
            out = self.model.classify(z)

            # 处理掩码
            if mask is not None:
                # 确保node_labels是长整型用于argmax比较
                if not torch.is_integral(node_labels):
                    node_labels = node_labels.long()
                pred = out[mask].argmax(dim=1)
                y = node_labels[mask]
            else:
                # 确保node_labels是长整型用于argmax比较
                if not torch.is_integral(node_labels):
                    node_labels = node_labels.long()
                pred = out.argmax(dim=1)
                y = node_labels
        else:
            # 异构图的情况
            z_dict = self.model.encode(x, edge_index)
            out = self.model.classify(z_dict["parcel"])

            if mask is not None:
                if not torch.is_integral(node_labels):
                    node_labels = node_labels.long()
                pred = out[mask].argmax(dim=1)
                y = node_labels[mask]
            else:
                if not torch.is_integral(node_labels):
                    node_labels = node_labels.long()
                pred = out.argmax(dim=1)
                y = node_labels

        # 转换为NumPy数组用于评估指标计算
        y_true = y.cpu().numpy()
        y_pred = pred.cpu().numpy()

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)

        if return_metrics:
            # 获取类别数量
            num_classes = len(torch.unique(torch.cat([y, pred])))

            # 计算详细指标
            metrics = {'accuracy': accuracy}

            try:
                # 计算宏平均指标
                metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro',
                                                             zero_division=zero_division)
                metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=zero_division)
                metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=zero_division)

                # 计算加权平均指标
                metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted',
                                                                zero_division=zero_division)
                metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted',
                                                          zero_division=zero_division)
                metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=zero_division)

                # 计算每个类别的指标
                metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None,
                                                                 zero_division=zero_division)
                metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=zero_division)
                metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=zero_division)

                # 计算混淆矩阵
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

                # 统计每个类别的样本数量
                metrics['class_counts'] = np.bincount(y_true, minlength=num_classes)

                # 统计每个类别的预测数量
                metrics['prediction_counts'] = np.bincount(y_pred, minlength=num_classes)

            except Exception as e:
                # 处理计算指标时可能出现的错误
                print(f"警告: 无法计算所有指标。错误: {e}")

            return accuracy, pred, metrics

        return accuracy, pred

    def find_best(self, ckpt_dir, test_graph, n_trials, model_ids):
        """
        使用Optuna找到最佳模型检查点

        参数:
            ckpt_dir: 包含模型检查点的目录
            test_graph: 测试图数据
            n_trials: Optuna的试验次数
            model_ids: 要尝试的模型ID列表（轮数编号）

        返回:
            best_params: 找到的最佳参数
            best_accuracy: 达到的最佳准确率
        """
        import optuna

        def objective(trial):
            epoch = trial.suggest_categorical("model_id", model_ids)
            ckpt = torch.load(f"{ckpt_dir}/{self.model.name}_{epoch}.pt")
            self.model.load_state_dict(ckpt)
            accuracy, _ = self.test(test_graph, return_metrics=False)

            return accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("Best accuracy:", study.best_value)
        print("Optimal parameters:", study.best_params)

        return study.best_params, study.best_value