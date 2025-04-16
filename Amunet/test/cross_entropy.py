import torch
import torch.nn.functional as F

# 假设 edge_1 和 edge_2 是你的输入张量
# edge_1 的形状是 (N, C, H, W)
# edge_2 的形状是 (N, H, W)

# 示例数据
edge_1 = torch.randn(10, 1, 40, 40, requires_grad=True)  # 模型的输出，假设有两个类别
edge_2 = torch.randint(0, 2, (10, 40, 40))  # 目标标签

# 计算交叉熵损失
loss = F.cross_entropy(edge_1, edge_2)
print(loss.item())