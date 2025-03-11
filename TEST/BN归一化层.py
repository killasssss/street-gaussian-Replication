import torch
import torch.nn as nn
import torch.nn.functional as F


class BnLayer(nn.Module):
    """
    自定义 Batch Normalization (BN) 层
    适用于 4D 输入 (batch_size, channels, height, width)
    """

    def __init__(self, feat_nums):
        """
        初始化 BN 层
        :param feat_nums: 特征通道数 (C)
        """
        super(BnLayer, self).__init__()

        # 存储通道数
        self.feat_nums = feat_nums

        # 避免除零错误的小常数 epsilon
        self.eps = 1e-5

        # 指数加权移动平均的动量
        self.momentum = 0.9

        # 形状：(1, C, 1, 1)，用于通道级别的广播
        shape = (1, feat_nums, 1, 1)

        # 训练过程中可学习的缩放因子 gamma，初始化为 1
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)

        # 训练过程中可学习的偏移因子 beta，初始化为 0
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)

        # 训练时维护的移动平均均值和方差（初始化）
        self.moving_mean = torch.zeros(shape)  # 初始均值为 0
        self.moving_var = torch.ones(shape)  # 初始方差为 1

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 (batch_size, C, H, W)
        :return: 归一化后的张量，形状与 x 相同
        """
        # 确保 moving_mean 和 moving_var 运行在相同设备（避免 GPU/CPU 设备不匹配错误）
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        # 进行批归一化计算
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.momentum, self.eps,
            self.moving_mean, self.moving_var, is_train=self.training
        )
        return y


def batch_norm(x, gamma, beta, momentum, eps, moving_mean, moving_var, is_train=True):
    """
    执行批归一化操作
    :param x: 输入张量 (batch_size, C, H, W)
    :param gamma: 训练过程中可学习的缩放因子
    :param beta: 训练过程中可学习的偏移因子
    :param momentum: 移动平均动量参数
    :param eps: 避免除零错误的小常数
    :param moving_mean: 训练时的移动平均均值
    :param moving_var: 训练时的移动平均方差
    :param is_train: 是否处于训练模式
    :return: 归一化后的输出张量，更新后的均值、方差
    """

    # 计算 batch 维度（N）、空间维度（H, W）的均值
    x_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # 均值 (1, C, 1, 1)
    x_var = torch.mean((x - x_mean) ** 2, dim=(0, 2, 3), keepdim=True)  # 方差 (1, C, 1, 1)

    if is_train:
        # 训练模式：使用 batch 统计量进行归一化
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)

        # 计算并更新移动平均均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * x_mean
        moving_var = momentum * moving_var + (1 - momentum) * x_var

    else:
        # 推理模式：使用训练过程中保存的 moving_mean 和 moving_var
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)

    # 计算最终 BN 输出 y = gamma * x_hat + beta
    out = gamma * x_hat + beta

    return out, moving_mean, moving_var


# 测试代码
if __name__ == '__main__':
    # 创建一个测试输入张量，形状为 (batch=2, channels=3, height=10, width=10)
    tensor = torch.ones(2, 3, 10, 10)

    # 创建 Batch Normalization 层，通道数为 3
    bn = BnLayer(3)

    # 进行 BN 计算
    output = bn(tensor)

    # 输出结果
    print("输出张量形状：", output.shape)  # 期待结果：(2, 3, 10, 10)
    print("输出张量：", output)
