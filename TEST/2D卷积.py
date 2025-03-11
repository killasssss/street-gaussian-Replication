import numpy as np


def Conv2d(input_tensor, in_channels, out_channels, kernel, stride, padding):
    """
    实现 2D 卷积操作（不带偏置）。

    参数:
    input_tensor : np.ndarray
        输入特征图，形状为 (batch_size, in_channels, input_height, input_width)
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    kernel : np.ndarray
        卷积核，形状为 (out_channels, in_channels, kernel_height, kernel_width)
    stride : int
        步长（stride）
    padding : int
        填充（padding）

    返回:
    output_tensor : np.ndarray
        卷积后的输出特征图，形状为 (batch_size, out_channels, output_height, output_width)
    """

    # 获取输入形状
    batch_size, _, input_height, input_width = input_tensor.shape
    kernel_out_channels, _, kernel_height, kernel_width = kernel.shape

    # 计算输出特征图的高度和宽度
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    # 对输入进行填充
    input_padded = np.pad(input_tensor,
                          [(0, 0), (0, 0), (padding, padding), (padding, padding)],
                          mode='constant')

    # 初始化输出张量
    output_tensor = np.zeros((batch_size, out_channels, output_height, output_width))

    # 遍历 batch 维度
    for batch in range(batch_size):
        # 遍历输出通道数
        for out_channel in range(out_channels):
            # 遍历输出特征图的高度
            for out_y in range(output_height):
                # 遍历输出特征图的宽度
                for out_x in range(output_width):
                    # 遍历输入通道数
                    for in_channel in range(in_channels):
                        # 遍历卷积核的高度
                        for k_y in range(kernel_height):
                            # 遍历卷积核的宽度
                            for k_x in range(kernel_width):
                                # 按照卷积操作计算加权求和
                                output_tensor[batch, out_channel, out_y, out_x] += (
                                        input_padded[batch, in_channel, out_y * stride + k_y, out_x * stride + k_x]
                                        * kernel[out_channel, in_channel, k_y, k_x]
                                )

    return output_tensor
